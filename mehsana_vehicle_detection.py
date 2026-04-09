import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque, defaultdict
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== LOAD ROI CONFIGURATION ==========
def load_roi_config():
    """Load ROI configuration from JSON file"""
    config_file = os.path.join(BASE_DIR, "roi_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            mode = config.get('mode', 'LINE')
            print(f"OK Loaded ROI configuration ({mode} mode) from {config_file}")
            return config
        except Exception as e:
            print(f"ERROR loading config: {e}. Using defaults.")
    else:
        print(f"INFO {config_file} not found. Using default configuration.")
    return None

# ========== CONFIGURATION ==========
class Config:
    # Video Settings
    SPEED_MULTIPLIER = 4
    VIDEO_PATH = "input.MP4"  # Video in Mehsana folder

    # Detection Settings
    YOLO_MODEL = "yolov8n.pt"  # Using yolov8n.pt as it's in the workspace
    CONFIDENCE_THRESHOLD = 0.25  # Lowered for better detection
    VEHICLE_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    PEDESTRIAN_IDS = [0]  # person

    # Calibration
    PIXELS_PER_METER = 30

    # Thresholds
    CONGESTION_DENSITY_THRESHOLD = 50  # veh/km
    MIN_TRACKING_DISTANCE = 150  # pixels
    LINE_CROSSING_BUFFER_PX = 4
    TRACK_MAX_AGE_SECONDS = 0.8
    TRACK_POINT_BOTTOM_OFFSET_RATIO = 0.15

    # NEW: Accident & Safety Thresholds
    COLLISION_DISTANCE_THRESHOLD = 25  # pixels
    COLLISION_MIN_CLOSING_PIXELS = 10
    COLLISION_MIN_MOTION_PIXELS = 12
    COLLISION_CONFIRM_FRAMES = 3
    STOPPED_VEHICLE_THRESHOLD_SECONDS = 8  # seconds
    STOPPED_MOVEMENT_THRESHOLD = 8  # pixels
    STOPPED_ALERT_DEDUP_SECONDS = 6
    STOPPED_ALERT_DEDUP_DISTANCE = 90
    PEDESTRIAN_CONFIDENCE_THRESHOLD = 0.45
    PEDESTRIAN_MIN_BOX_HEIGHT = 28
    PEDESTRIAN_MIN_ASPECT_RATIO = 1.2
    PEDESTRIAN_CONFIRM_FRAMES = 3
    PEDESTRIAN_MAX_VEHICLE_OVERLAP = 0.35
    PEDESTRIAN_TRACKING_DISTANCE = 90
    PEDESTRIAN_TRACK_MAX_AGE_SECONDS = 1.0
    PEDESTRIAN_EVENT_DEDUP_SECONDS = 3.0
    PEDESTRIAN_EVENT_DEDUP_DISTANCE = 120
    QUEUE_DETECTION_DISTANCE = 200  # pixels before line
    SPEED_LIMIT_KMH = 60  # km/h
    WRONG_WAY_ANGLE_THRESHOLD = 120  # degrees
    WRONG_WAY_MIN_TRAVEL_PIXELS = 25
    WRONG_WAY_CONFIRM_FRAMES = 3
    WRONG_WAY_ALLOWED_FLOW = "TOP_TO_BOTTOM"
    WRONG_WAY_VERTICAL_DOMINANCE = 1.1

    # Time Windows
    TIME_WINDOWS = [20, 40, 60, 120]  # seconds
    FLOW_WINDOW_SECONDS = 60

    # Export Settings
    EXPORT_DATA = True
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    ROI_MODE = "LINE"
    PROCESS_SIDE = "RIGHT"
    ACTIVE_AOI_POINTS = None
    DIVIDER_LINE_START = None
    DIVIDER_LINE_END = None

    # Load ROI from config file or use defaults
    _roi_config = load_roi_config()
    if _roi_config:
        _config_mode = _roi_config.get('mode', 'LINE')
        ROI_MODE = _config_mode
        PROCESS_SIDE = _roi_config.get('process_side', 'RIGHT').upper()
        if _config_mode == 'RECTANGLE':
            # For RECTANGLE mode, use the rectangle ROI as AOI
            rect_roi = _roi_config.get('rectangle_roi', None)
            if rect_roi:
                AOI_POINTS = np.array(rect_roi, dtype=np.int32)
                # Use first edge as detection line
                DETECTION_LINE_START = tuple(rect_roi[0])
                DETECTION_LINE_END = tuple(rect_roi[1])
            else:
                raise ValueError("Rectangle ROI not found in config")
        elif _config_mode == 'SIDE_LINE':
            divider_line = _roi_config.get('divider_line')
            detection_line = _roi_config.get('detection_line') or divider_line
            if divider_line and detection_line:
                AOI_POINTS = np.empty((0, 2), dtype=np.int32)
                DIVIDER_LINE_START = tuple(divider_line['start'])
                DIVIDER_LINE_END = tuple(divider_line['end'])
                DETECTION_LINE_START = tuple(detection_line['start'])
                DETECTION_LINE_END = tuple(detection_line['end'])
            else:
                raise ValueError("Divider line and/or detection line not found in config")
        else:
            # LINE mode - original behavior
            AOI_POINTS = np.array(_roi_config['aoi_polygon'], dtype=np.int32)
            DETECTION_LINE_START = tuple(_roi_config['detection_line']['start'])
            DETECTION_LINE_END = tuple(_roi_config['detection_line']['end'])
    else:
        # Default Area of Interest (AOI) polygon points - between L1 and L2
        AOI_POINTS = np.array([
            [4, 1019],    # L1 start
            [1188, 219],  # L1 end
            [1872, 1099], # L2 end
            [1218, 240]   # L2 start
        ], np.int32)
        DETECTION_LINE_START = (4, 1019)
        DETECTION_LINE_END = (1218, 240)

    if DIVIDER_LINE_START is None:
        DIVIDER_LINE_START = DETECTION_LINE_START
    if DIVIDER_LINE_END is None:
        DIVIDER_LINE_END = DETECTION_LINE_END

def ensure_output_dir():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    return Config.OUTPUT_DIR


def build_output_path(filename):
    return os.path.join(ensure_output_dir(), filename)


VEHICLE_LABELS = {
    0: "Pedestrian",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


def line_midpoint(start, end):
    return np.array([(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0], dtype=np.float32)


def estimate_aoi_length_pixels(aoi_points):
    """Approximate AOI length from the midpoint distance of opposite edges."""
    start_mid = line_midpoint(aoi_points[0], aoi_points[1])
    end_mid = line_midpoint(aoi_points[3], aoi_points[2])
    return float(np.linalg.norm(end_mid - start_mid))


def estimate_line_length_pixels(line_start, line_end):
    return float(np.linalg.norm(np.asarray(line_end, dtype=np.float32) - np.asarray(line_start, dtype=np.float32)))


def polygon_area_pixels(points):
    return float(cv2.contourArea(np.asarray(points, dtype=np.float32)))


def signed_distance_to_line(point, line_start, line_end):
    """Signed perpendicular distance from a point to an oriented line."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    denominator = np.hypot(x2 - x1, y2 - y1)
    if denominator == 0:
        return 0.0
    return (((x2 - x1) * (y0 - y1)) - ((y2 - y1) * (x0 - x1))) / denominator


def bbox_tracking_point(x1, y1, x2, y2):
    """Use a point near the vehicle's road-contact area for crossing logic."""
    height = max(1, y2 - y1)
    cx = (x1 + x2) // 2
    cy = int(y2 - (height * Config.TRACK_POINT_BOTTOM_OFFSET_RATIO))
    return cx, cy


def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0
    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)


def is_valid_pedestrian_detection(box, confidence, vehicle_boxes):
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area = width * height

    if confidence < Config.PEDESTRIAN_CONFIDENCE_THRESHOLD:
        return False
    if height < Config.PEDESTRIAN_MIN_BOX_HEIGHT:
        return False
    if (height / float(width)) < Config.PEDESTRIAN_MIN_ASPECT_RATIO:
        return False

    for vehicle_box in vehicle_boxes:
        overlap = intersection_area(box, vehicle_box)
        if overlap <= 0:
            continue
        overlap_ratio = overlap / float(area)
        if overlap_ratio >= Config.PEDESTRIAN_MAX_VEHICLE_OVERLAP:
            return False

    return True


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


def detection_line_forward_vector(line_start, line_end, direction_sign):
    dx = float(line_end[0] - line_start[0])
    dy = float(line_end[1] - line_start[1])
    normal = np.array([-dy, dx], dtype=np.float32)
    normal = normalize_vector(normal)
    if normal is None:
        return np.array([1.0, 0.0], dtype=np.float32)
    return normal * float(direction_sign)


def counting_direction_sign(aoi_points, line_start, line_end):
    """Choose the same directional crossing logic as the reference scripts."""
    upstream_mid = line_midpoint(aoi_points[0], aoi_points[1])
    downstream_mid = line_midpoint(aoi_points[3], aoi_points[2])
    upstream_dist = signed_distance_to_line(upstream_mid, line_start, line_end)
    downstream_dist = signed_distance_to_line(downstream_mid, line_start, line_end)

    if upstream_dist == 0 and downstream_dist == 0:
        return -1.0
    if upstream_dist == 0:
        upstream_dist = -downstream_dist if downstream_dist != 0 else -1.0

    return 1.0 if upstream_dist > 0 else -1.0


def frame_rectangle_polygon(frame_shape):
    h, w = frame_shape[:2]
    return np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)


def clip_polygon_to_half_plane(polygon, line_start, line_end, include_sign, epsilon=1e-6):
    if len(polygon) == 0:
        return np.empty((0, 2), dtype=np.float32)

    output = []
    prev_point = np.asarray(polygon[-1], dtype=np.float32)
    prev_dist = signed_distance_to_line(tuple(prev_point), line_start, line_end) * include_sign
    prev_inside = prev_dist >= -epsilon

    for point in polygon:
        curr_point = np.asarray(point, dtype=np.float32)
        curr_dist = signed_distance_to_line(tuple(curr_point), line_start, line_end) * include_sign
        curr_inside = curr_dist >= -epsilon

        if curr_inside != prev_inside:
            denominator = prev_dist - curr_dist
            if abs(denominator) > epsilon:
                ratio = prev_dist / denominator
                intersection = prev_point + ratio * (curr_point - prev_point)
                output.append(intersection)

        if curr_inside:
            output.append(curr_point)

        prev_point = curr_point
        prev_dist = curr_dist
        prev_inside = curr_inside

    return np.asarray(output, dtype=np.float32)


def resolve_process_side_sign(frame_shape, line_start, line_end, process_side):
    h, w = frame_shape[:2]
    if process_side == "LEFT":
        reference_points = [(0, 0), (0, h // 2), (0, h - 1), (w // 4, h // 2)]
    else:
        reference_points = [(w - 1, 0), (w - 1, h // 2), (w - 1, h - 1), ((3 * w) // 4, h // 2)]

    signed_refs = [signed_distance_to_line(point, line_start, line_end) for point in reference_points]
    strongest = max(signed_refs, key=lambda value: abs(value)) if signed_refs else 1.0
    if abs(strongest) < 1e-6:
        strongest = 1.0
    return 1.0 if strongest >= 0 else -1.0


def build_side_line_roi(frame_shape, line_start, line_end, process_side="RIGHT"):
    frame_polygon = frame_rectangle_polygon(frame_shape)
    include_sign = resolve_process_side_sign(frame_shape, line_start, line_end, process_side.upper())

    include_polygon = clip_polygon_to_half_plane(frame_polygon, line_start, line_end, include_sign)
    exclude_polygon = clip_polygon_to_half_plane(frame_polygon, line_start, line_end, -include_sign)

    if len(include_polygon) < 3:
        include_polygon = frame_polygon.copy()
    if len(exclude_polygon) < 3:
        exclude_polygon = np.empty((0, 2), dtype=np.float32)

    return {
        "mode": "SIDE_LINE",
        "process_side": process_side.upper(),
        "include_sign": include_sign,
        "include_polygon": np.round(include_polygon).astype(np.int32),
        "exclude_polygon": np.round(exclude_polygon).astype(np.int32),
    }


def prepare_roi_geometry(frame_shape):
    if Config.ROI_MODE == "SIDE_LINE":
        roi_geometry = build_side_line_roi(
            frame_shape,
            Config.DIVIDER_LINE_START,
            Config.DIVIDER_LINE_END,
            Config.PROCESS_SIDE
        )
        aoi_points = roi_geometry["include_polygon"]
    else:
        aoi_points = Config.AOI_POINTS.astype(np.int32)
        roi_geometry = {
            "mode": Config.ROI_MODE,
            "include_polygon": aoi_points,
            "exclude_polygon": np.empty((0, 2), dtype=np.int32),
            "include_sign": counting_direction_sign(
                aoi_points,
                Config.DETECTION_LINE_START,
                Config.DETECTION_LINE_END
            ),
            "process_side": None,
        }

    roi_geometry["count_direction_sign"] = counting_direction_sign(
        aoi_points,
        Config.DETECTION_LINE_START,
        Config.DETECTION_LINE_END
    ) if len(aoi_points) >= 4 else -1.0

    Config.ACTIVE_AOI_POINTS = aoi_points.copy()
    aoi_contour = aoi_points.reshape((-1, 1, 2))
    return roi_geometry, aoi_contour

# ========== NEW FEATURE 1: SPEED ESTIMATION ==========
class SpeedEstimator:
    def __init__(self, pixels_per_meter):
        self.pixels_per_meter = pixels_per_meter
        self.vehicle_speeds = {}  # vid -> deque of (time, speed)
        self.speed_history = defaultdict(lambda: deque(maxlen=10))
        
    def estimate_speed(self, vehicle_id, prev_point, curr_point, time_delta):
        """Estimate vehicle speed in km/h"""
        if time_delta == 0 or prev_point is None:
            return 0
            
        pixel_distance = np.linalg.norm(
            np.array(curr_point) - np.array(prev_point)
        )
        
        # Convert to real-world units
        meters = pixel_distance / self.pixels_per_meter
        speed_mps = meters / time_delta
        speed_kmh = speed_mps * 3.6
        
        # Store in history
        self.speed_history[vehicle_id].append(speed_kmh)
        
        return speed_kmh
    
    def get_average_speed(self, vehicle_id):
        """Get smoothed average speed for a vehicle"""
        if vehicle_id not in self.speed_history or len(self.speed_history[vehicle_id]) == 0:
            return 0
        return np.mean(list(self.speed_history[vehicle_id]))
    
    def is_speeding(self, speed_kmh, limit_kmh=None):
        """Check if vehicle is speeding"""
        if limit_kmh is None:
            limit_kmh = Config.SPEED_LIMIT_KMH
        return speed_kmh > limit_kmh

# ========== NEW FEATURE 2: STOPPED VEHICLE DETECTION ==========
class StoppedVehicleDetector:
    def __init__(self, stopped_threshold_seconds=3, movement_threshold=10):
        self.vehicle_positions = defaultdict(deque)
        self.stopped_threshold = stopped_threshold_seconds
        self.movement_threshold = movement_threshold
        self.stopped_vehicles = set()
        self.stopped_alerts = deque(maxlen=100)
        self.recent_alert_positions = deque(maxlen=100)
        
    def update(self, vehicle_id, position, current_time):
        """Update vehicle position and check if stopped"""
        self.vehicle_positions[vehicle_id].append((position, current_time))
        
        # Keep only recent positions
        while (len(self.vehicle_positions[vehicle_id]) > 0 and 
               current_time - self.vehicle_positions[vehicle_id][0][1] > self.stopped_threshold + 1):
            self.vehicle_positions[vehicle_id].popleft()
        
        is_stopped = self.check_stopped(vehicle_id, current_time)
        
        # Alert if newly stopped
        if is_stopped and vehicle_id not in self.stopped_vehicles:
            self.stopped_vehicles.add(vehicle_id)
            if not self._is_duplicate_alert(position, current_time):
                self.recent_alert_positions.append((current_time, position))
                self.stopped_alerts.append({
                    'time': current_time,
                    'vehicle_id': vehicle_id,
                    'position': position,
                    'duration': self.get_stopped_duration(vehicle_id, current_time)
                })
        elif not is_stopped and vehicle_id in self.stopped_vehicles:
            self.stopped_vehicles.discard(vehicle_id)
            
        return is_stopped

    def _is_duplicate_alert(self, position, current_time):
        for event_time, event_position in self.recent_alert_positions:
            if current_time - event_time > Config.STOPPED_ALERT_DEDUP_SECONDS:
                continue
            if np.linalg.norm(np.array(position) - np.array(event_position)) <= Config.STOPPED_ALERT_DEDUP_DISTANCE:
                return True
        return False
    
    def check_stopped(self, vehicle_id, current_time):
        """Check if vehicle is currently stopped"""
        if len(self.vehicle_positions[vehicle_id]) < 2:
            return False
            
        # Calculate movement
        positions = [p for p, _ in self.vehicle_positions[vehicle_id]]
        if len(positions) < 2:
            return False
            
        movement = max([np.linalg.norm(np.array(p1) - np.array(p2)) 
                       for p1, p2 in zip(positions[:-1], positions[1:])])
        
        time_stopped = current_time - self.vehicle_positions[vehicle_id][0][1]
        
        return movement < self.movement_threshold and time_stopped >= self.stopped_threshold
    
    def get_stopped_duration(self, vehicle_id, current_time):
        """Get how long vehicle has been stopped"""
        if vehicle_id not in self.vehicle_positions or len(self.vehicle_positions[vehicle_id]) == 0:
            return 0
        return current_time - self.vehicle_positions[vehicle_id][0][1]

# ========== NEW FEATURE 3: QUEUE LENGTH DETECTION ==========
class QueueLengthDetector:
    def __init__(self, detection_line_start, detection_line_end, queue_distance=200):
        self.detection_line_start = detection_line_start
        self.detection_line_end = detection_line_end
        self.queue_distance = queue_distance
        self.queue_history = deque(maxlen=500)
        
    def calculate_queue_length(self, current_centers, current_time):
        """Measure vehicle queue length before detection line"""
        queued_vehicles = []
        
        for vid, (cx, cy) in current_centers.items():
            dist = signed_distance_to_line((cx, cy), self.detection_line_start, self.detection_line_end)
            
            # Vehicles waiting before line (within threshold distance)
            if -self.queue_distance < dist < 0:
                queued_vehicles.append((vid, dist))
        
        if not queued_vehicles:
            self.queue_history.append((current_time, 0, 0))
            return 0, 0
            
        # Sort by distance to line
        queued_vehicles.sort(key=lambda x: x[1])
        queue_length_pixels = abs(queued_vehicles[0][1] - queued_vehicles[-1][1])
        queue_count = len(queued_vehicles)
        
        self.queue_history.append((current_time, queue_count, queue_length_pixels))
        
        return queue_count, queue_length_pixels
    
    def get_average_queue(self, current_time, window_seconds=60):
        """Get average queue length over time window"""
        recent_queues = [count for t, count, _ in self.queue_history 
                        if current_time - t <= window_seconds]
        if not recent_queues:
            return 0
        return np.mean(recent_queues)

# ========== NEW FEATURE 4: ACCIDENT/COLLISION DETECTION ==========
class AccidentDetector:
    def __init__(self, collision_threshold=30):
        self.collision_threshold = collision_threshold
        self.collision_history = deque(maxlen=100)
        self.active_collisions = {}
        self.collision_events = []
        self.candidate_frames = defaultdict(int)
        
    def detect_collision(self, current_centers, previous_centers, current_time):
        """Detect potential collisions only when vehicles are converging across multiple frames."""
        vehicle_ids = list(current_centers.keys())
        collisions_detected = []
        
        for i, vid1 in enumerate(vehicle_ids):
            for vid2 in vehicle_ids[i+1:]:
                pos1 = np.array(current_centers[vid1])
                pos2 = np.array(current_centers[vid2])
                prev1 = previous_centers.get(vid1)
                prev2 = previous_centers.get(vid2)
                
                distance = np.linalg.norm(pos1 - pos2)
                collision_key = tuple(sorted([vid1, vid2]))

                if prev1 is None or prev2 is None:
                    self.candidate_frames.pop(collision_key, None)
                    continue

                prev1 = np.array(prev1)
                prev2 = np.array(prev2)
                prev_distance = np.linalg.norm(prev1 - prev2)
                motion1 = np.linalg.norm(pos1 - prev1)
                motion2 = np.linalg.norm(pos2 - prev2)
                closing_pixels = prev_distance - distance

                is_candidate = (
                    distance < self.collision_threshold and
                    closing_pixels > Config.COLLISION_MIN_CLOSING_PIXELS and
                    motion1 > Config.COLLISION_MIN_MOTION_PIXELS and
                    motion2 > Config.COLLISION_MIN_MOTION_PIXELS
                )

                if not is_candidate:
                    self.candidate_frames.pop(collision_key, None)
                    continue

                self.candidate_frames[collision_key] += 1

                if (
                    self.candidate_frames[collision_key] >= Config.COLLISION_CONFIRM_FRAMES and
                    collision_key not in self.active_collisions
                ):
                    collision_event = {
                        'time': current_time,
                        'vehicles': list(collision_key),
                        'location': ((pos1 + pos2) / 2).tolist(),
                        'distance': float(distance),
                        'closing_pixels': float(closing_pixels),
                        'severity': self._calculate_severity(distance)
                    }
                    self.collision_history.append(collision_event)
                    self.collision_events.append(collision_event)
                    self.active_collisions[collision_key] = current_time
                    collisions_detected.append(collision_event)
        
        # Clean up old collisions
        expired = [k for k, t in self.active_collisions.items() 
                  if current_time - t > 2.0]
        for k in expired:
            del self.active_collisions[k]
        expired_candidates = [k for k in self.candidate_frames if k not in self.active_collisions and self.candidate_frames[k] <= 0]
        for k in expired_candidates:
            self.candidate_frames.pop(k, None)
                    
        return len(collisions_detected) > 0, collisions_detected
    
    def _calculate_severity(self, distance):
        """Calculate collision severity based on distance"""
        if distance < 10:
            return "CRITICAL"
        elif distance < 20:
            return "HIGH"
        else:
            return "MEDIUM"

# ========== NEW FEATURE 5: WRONG-WAY DETECTION ==========
class WrongWayDetector:
    def __init__(self, expected_direction_degrees=0, angle_threshold=90):
        self.expected_direction = np.radians(expected_direction_degrees)
        self.angle_threshold = np.radians(angle_threshold)
        self.wrong_way_vehicles = set()
        self.wrong_way_events = deque(maxlen=100)
        self.confirmation_counts = defaultdict(int)
        self.reference_motion = np.array([1.0, 0.0], dtype=np.float32)
        
    def set_reference_motion(self, motion_vector):
        normalized = normalize_vector(motion_vector)
        if normalized is not None:
            self.reference_motion = normalized
        
    def detect_wrong_way(self, vehicle_id, prev_point, curr_point, current_time):
        """Detect bottom-to-top motion as wrong-way for this one-way active region."""
        if prev_point is None:
            return False
        
        dx = curr_point[0] - prev_point[0]
        dy = curr_point[1] - prev_point[1]
        movement_mag = np.hypot(dx, dy)
        
        if movement_mag < Config.WRONG_WAY_MIN_TRAVEL_PIXELS:
            self.confirmation_counts[vehicle_id] = 0
            return False

        if Config.WRONG_WAY_ALLOWED_FLOW == "TOP_TO_BOTTOM":
            candidate_wrong_way = (
                dy < (-Config.WRONG_WAY_MIN_TRAVEL_PIXELS) and
                abs(dy) > (abs(dx) * Config.WRONG_WAY_VERTICAL_DOMINANCE)
            )
            angle_diff = 180.0 if candidate_wrong_way else 0.0
        else:
            motion_unit = normalize_vector((dx, dy))
            if motion_unit is None:
                self.confirmation_counts[vehicle_id] = 0
                return False

            dot = float(np.dot(motion_unit, self.reference_motion))
            angle_diff = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
            candidate_wrong_way = angle_diff > np.degrees(self.angle_threshold)

        if candidate_wrong_way:
            self.confirmation_counts[vehicle_id] += 1
        else:
            self.confirmation_counts[vehicle_id] = 0

        is_wrong_way = self.confirmation_counts[vehicle_id] >= Config.WRONG_WAY_CONFIRM_FRAMES

        if is_wrong_way and vehicle_id not in self.wrong_way_vehicles:
            self.wrong_way_vehicles.add(vehicle_id)
            self.wrong_way_events.append({
                'time': current_time,
                'vehicle_id': vehicle_id,
                'position': curr_point,
                'angle_diff_degrees': float(angle_diff)
            })
        elif not is_wrong_way and vehicle_id in self.wrong_way_vehicles:
            self.wrong_way_vehicles.discard(vehicle_id)
            
        return is_wrong_way

# ========== NEW FEATURE 6: LANE CHANGE DETECTION ==========
class LaneChangeDetector:
    def __init__(self, lane_divider_start, lane_divider_end):
        self.lane_divider_start = lane_divider_start
        self.lane_divider_end = lane_divider_end
        self.vehicle_lanes = {}  # vid -> 'LEFT'/'RIGHT'
        self.lane_change_events = deque(maxlen=500)
        
    def detect_lane_change(self, vehicle_id, position, current_time):
        """Detect lane changes"""
        current_side = self.get_side(position)
        
        changed = False
        change_info = None
        
        if vehicle_id in self.vehicle_lanes:
            prev_side = self.vehicle_lanes[vehicle_id]
            if current_side != prev_side:
                changed = True
                change_info = {
                    'time': current_time,
                    'vehicle_id': vehicle_id,
                    'from_lane': prev_side,
                    'to_lane': current_side,
                    'position': position
                }
                self.lane_change_events.append(change_info)
        
        self.vehicle_lanes[vehicle_id] = current_side
        return changed, change_info
    
    def get_side(self, position):
        """Determine which side of divider line the position is on"""
        dist = signed_distance_to_line(
            position, 
            self.lane_divider_start, 
            self.lane_divider_end
        )
        return 'RIGHT' if dist > 0 else 'LEFT'
    
    def get_lane_change_count(self, current_time, window_seconds=60):
        """Count lane changes in time window"""
        return sum(1 for event in self.lane_change_events 
                  if current_time - event['time'] <= window_seconds)

# ========== NEW FEATURE 7: VISIBILITY/WEATHER ASSESSMENT ==========
class VisibilityAssessor:
    def __init__(self):
        self.visibility_history = deque(maxlen=100)
        
    def assess_visibility(self, frame, current_time):
        """Assess frame brightness/contrast as visibility proxy"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))
        
        # Determine visibility condition
        visibility = "GOOD"
        color = (0, 255, 0)
        
        if mean_brightness < 60:
            visibility = "LOW_LIGHT"
            color = (0, 165, 255)
        elif std_brightness < 30:
            visibility = "POOR_CONTRAST"  # Possible fog/rain
            color = (0, 100, 255)
        
        self.visibility_history.append({
            'time': current_time,
            'condition': visibility,
            'brightness': mean_brightness,
            'contrast': std_brightness
        })
        
        return visibility, mean_brightness, std_brightness, color

# ========== ADVANCED LANE TRACKER (ENHANCED) ==========
class AdvancedLaneTracker:
    def __init__(self, lane_name, lane_index):
        self.lane_name = lane_name
        self.lane_index = lane_index

        # Basic tracking
        self.previous_centers = {}
        self.track_last_seen = {}
        self.next_id = 0
        self.vehicle_count = 0
        self.last_cross_time = None
        self.crossed_vehicle_ids = set()

        # Event and time-series data
        self.headways = deque(maxlen=2000)
        self.crossing_events = deque(maxlen=5000)
        self.density_history = deque(maxlen=1000)
        self.flow_history = deque(maxlen=1000)
        self.occupancy_history = deque(maxlen=1000)

        # Statistical tracking
        self.vehicles_in_roi = 0
        self.pedestrians_on_road = 0
        self.current_bbox_area = 0
        self.total_travel_time = []
        self.vehicle_entry_times = {}
        self.vehicle_class_by_id = {}
        self.vehicle_classes = defaultdict(int)
        self.pedestrian_previous_centers = {}
        self.pedestrian_track_last_seen = {}
        self.pedestrian_track_hits = {}
        self.pedestrian_next_id = 0
        self.pedestrian_events = deque(maxlen=1000)
        self.recent_pedestrian_event_positions = deque(maxlen=200)
        self.confirmed_pedestrian_ids = set()

        # Time series data for prediction
        self.time_series_data = []

        # Anomaly detection
        self.anomalies = []
        
        # NEW: Initialize all new feature detectors
        self.speed_estimator = SpeedEstimator(Config.PIXELS_PER_METER)
        self.stopped_detector = StoppedVehicleDetector(
            Config.STOPPED_VEHICLE_THRESHOLD_SECONDS,
            Config.STOPPED_MOVEMENT_THRESHOLD
        )
        self.queue_detector = QueueLengthDetector(
            Config.DETECTION_LINE_START,
            Config.DETECTION_LINE_END,
            Config.QUEUE_DETECTION_DISTANCE
        )
        self.accident_detector = AccidentDetector(Config.COLLISION_DISTANCE_THRESHOLD)
        self.wrong_way_detector = WrongWayDetector(expected_direction_degrees=0, angle_threshold=Config.WRONG_WAY_ANGLE_THRESHOLD)
        self.lane_change_detector = LaneChangeDetector(
            Config.DIVIDER_LINE_START,
            Config.DIVIDER_LINE_END
        )
        
        # Store previous positions for speed/direction calculation
        self.previous_positions = {}
        self.previous_time = {}

    def register_pedestrian_event(self, pedestrian_id, position, current_time):
        for event_time, event_position in self.recent_pedestrian_event_positions:
            if current_time - event_time > Config.PEDESTRIAN_EVENT_DEDUP_SECONDS:
                continue
            if np.linalg.norm(np.array(position) - np.array(event_position)) <= Config.PEDESTRIAN_EVENT_DEDUP_DISTANCE:
                return False

        self.recent_pedestrian_event_positions.append((current_time, position))
        self.pedestrian_events.append({
            'time': current_time,
            'pedestrian_id': pedestrian_id,
            'position': position
        })
        return True

    def get_vehicle_mix(self):
        total = sum(self.vehicle_classes.values())
        if total == 0:
            return {}

        return {
            VEHICLE_LABELS.get(cls_id, str(cls_id)): (count / total) * 100.0
            for cls_id, count in sorted(self.vehicle_classes.items())
        }

    def get_total_crossings(self):
        """Total vehicles counted at the red detection line."""
        return len(self.crossing_events)

    def calculate_avg_headway(self, current_time, window_seconds):
        """Calculate average headway for given time window"""
        valid_headways = [h for (t, h) in self.headways
                         if current_time - t <= window_seconds]
        if len(valid_headways) > 0:
            return sum(valid_headways) / len(valid_headways)
        return 0

    def calculate_headway_variance(self, current_time, window_seconds):
        """Calculate headway variance (measure of traffic stability)"""
        valid_headways = [h for (t, h) in self.headways
                         if current_time - t <= window_seconds]
        if len(valid_headways) > 1:
            return np.var(valid_headways)
        return 0

    def register_crossing(self, vehicle_id, cls_id, current_time):
        """Record a detection-line crossing event and its headway."""
        self.crossed_vehicle_ids.add(vehicle_id)
        self.vehicle_classes[cls_id] += 1

        travel_time = None
        if vehicle_id in self.vehicle_entry_times:
            travel_time = current_time - self.vehicle_entry_times[vehicle_id]
            if travel_time >= 0:
                self.total_travel_time.append(travel_time)

        headway = None
        if self.last_cross_time is not None:
            headway = current_time - self.last_cross_time
            if headway >= 0:
                self.headways.append((current_time, headway))

        self.last_cross_time = current_time
        self.crossing_events.append((current_time, vehicle_id, cls_id, headway, travel_time))
        self.vehicle_count = self.get_total_crossings()
        return True

    def update_metrics(self, current_time, aoi_length_pixels, aoi_area_pixels, pixels_per_meter):
        """Calculate traffic metrics using AOI occupancy and line-crossing events."""
        roi_length_km = aoi_length_pixels / pixels_per_meter / 1000
        density = self.vehicles_in_roi / roi_length_km if roi_length_km > 0 else 0
        self.density_history.append((current_time, density))

        recent_crosses = sum(
            1 for (t, *_rest) in self.crossing_events
            if current_time - t <= Config.FLOW_WINDOW_SECONDS
        )
        flow = (recent_crosses / Config.FLOW_WINDOW_SECONDS) * 3600 if recent_crosses > 0 else 0
        self.flow_history.append((current_time, flow))

        occupancy = (self.current_bbox_area / aoi_area_pixels * 100.0) if aoi_area_pixels > 0 else 0
        self.occupancy_history.append((current_time, occupancy))

        avg_travel_time = float(np.mean(self.total_travel_time)) if self.total_travel_time else 0
        last_headway = self.headways[-1][1] if self.headways else 0

        return {
            'density': density,
            'flow': flow,
            'occupancy': occupancy,
            'avg_travel_time': avg_travel_time,
            'last_headway': last_headway,
            'vehicles_in_roi': self.vehicles_in_roi,
            'pedestrians_on_road': self.pedestrians_on_road,
            'total_pedestrian_events': len(self.pedestrian_events),
            'vehicle_mix': self.get_vehicle_mix(),
        }

    def detect_anomaly(self, current_metrics, current_time):
        """Detect traffic anomalies using statistical methods"""
        if len(self.density_history) < 30:
            return False, None

        recent_densities = [d for (t, d) in self.density_history if current_time - t <= 120]

        if len(recent_densities) > 10:
            mean_density = np.mean(recent_densities)
            std_density = np.std(recent_densities)

            # Z-score anomaly detection
            if std_density > 0:
                z_score = abs((current_metrics['density'] - mean_density) / std_density)
                if z_score > 2.5:  # 2.5 standard deviations
                    anomaly_info = {
                        'time': current_time,
                        'type': 'density_spike' if current_metrics['density'] > mean_density else 'density_drop',
                        'severity': z_score,
                        'value': current_metrics['density']
                    }
                    return True, anomaly_info

        return False, None

    def predict_congestion(self, current_time, horizon_seconds=60):
        """Predict congestion using linear regression"""
        if len(self.density_history) < 20:
            return None, None

        # Get recent data
        recent_data = [(t, d) for (t, d) in self.density_history
                      if current_time - t <= 120]

        if len(recent_data) < 10:
            return None, None

        times = np.array([t for t, _ in recent_data]).reshape(-1, 1)
        densities = np.array([d for _, d in recent_data])

        # Normalize time
        times_normalized = times - times[0]

        # Linear regression
        model = LinearRegression()
        model.fit(times_normalized, densities)

        # Predict
        future_time = np.array([[times_normalized[-1][0] + horizon_seconds]])
        predicted_density = model.predict(future_time)[0]

        # Congestion probability
        congestion_prob = min(max(predicted_density / Config.CONGESTION_DENSITY_THRESHOLD, 0), 1.0)

        return predicted_density, congestion_prob

    def calculate_level_of_service(self, current_metrics):
        """Calculate Level of Service (A-F) based on HCM 2010"""
        density = current_metrics['density']

        if density < 11:
            return 'A', (0, 255, 0)      # Free flow
        elif density < 18:
            return 'B', (100, 255, 0)
        elif density < 26:
            return 'C', (200, 255, 0)
        elif density < 35:
            return 'D', (255, 200, 0)
        elif density < 45:
            return 'E', (255, 100, 0)
        else:
            return 'F', (255, 0, 0)       # Breakdown

# ========== TRAFFIC VISUALIZER (ENHANCED) ==========
class TrafficVisualizer:
    @staticmethod
    def draw_metrics_panel(frame, lane, x_off, y_start, current_time,
                           metrics, pred_density, congestion_prob,
                           queue_count, queue_length, stopped_count,
                           collision_detected, speeding_count, wrong_way_count,
                           lane_changes):
        headway_stats = {}
        for w in Config.TIME_WINDOWS:
            headway_stats[w] = (lane.calculate_avg_headway(current_time, w),
                                lane.calculate_headway_variance(current_time, w))

        los, los_color = lane.calculate_level_of_service(metrics)
        is_anomaly, anomaly_info = lane.detect_anomaly(metrics, current_time)

        panel_h = 670
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_off, y_start),
                      (x_off + 420, y_start + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (x_off, y_start),
                      (x_off + 420, y_start + panel_h), (100, 100, 100), 2)

        y = y_start + 30
        lh = 26

        def txt(text, color=(255, 255, 255), scale=0.6, bold=False):
            nonlocal y
            cv2.putText(frame, text, (x_off + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1)
            y += lh

        txt(f"=== {lane.lane_name} ===", (255, 255, 0), 0.8, True)
        y += 5
        txt(f"Total Count: {lane.get_total_crossings()}", (0, 255, 255), 0.7)
        txt(f"Current Vehicles: {metrics['vehicles_in_roi']}", (150, 255, 150), 0.65)
        txt(f"Pedestrians On Road: {metrics['pedestrians_on_road']}", (0, 165, 255), 0.65)
        txt(f"Pedestrian Events: {metrics['total_pedestrian_events']}", (0, 165, 255), 0.65)
        txt(f"Occupancy: {metrics['occupancy']:.1f}%", (255, 180, 120), 0.65)
        txt(f"Avg Travel Time: {metrics['avg_travel_time']:.2f}s", (255, 150, 150), 0.65)
        y += 3

        # NEW: Queue Information
        txt("--- Queue Analysis ---", (200, 200, 255), 0.65)
        queue_color = (0, 255, 0) if queue_count < 5 else (0, 165, 255) if queue_count < 10 else (0, 0, 255)
        txt(f"Queue: {queue_count} veh, {queue_length/Config.PIXELS_PER_METER:.1f}m", queue_color, 0.6)
        y += 3

        txt("--- Headway Analysis ---", (200, 200, 255), 0.65)
        txt(f"Current: {metrics['last_headway']:.2f}s", (255, 200, 100), 0.6)
        for w in [20, 40, 60]:
            a, v = headway_stats[w]
            txt(f"Avg({w}s): {a:.2f}s  (var:{v:.2f})", (150, 255, 200), 0.55)
        y += 3

        txt("--- Flow Fundamentals ---", (200, 200, 255), 0.65)
        txt(f"Density: {metrics['density']:.1f} veh/km", (150, 255, 200), 0.6)
        txt(f"Flow: {metrics['flow']:.0f} veh/h", (200, 150, 255), 0.6)
        txt(f"Level of Service: {los}", los_color, 0.7, True)
        y += 3

        # NEW: Safety & Incidents
        txt("--- Safety Monitoring ---", (200, 200, 255), 0.65)
        txt(f"Stopped Vehicles: {stopped_count}", 
            (0, 0, 255) if stopped_count > 0 else (0, 255, 0), 0.6)
        txt(f"Collisions: {'YES!' if collision_detected else 'None'}", 
            (0, 0, 255) if collision_detected else (0, 255, 0), 0.6, collision_detected)
        txt(f"Speeding: {speeding_count}", 
            (0, 165, 255) if speeding_count > 0 else (0, 255, 0), 0.6)
        txt(f"Wrong-Way: {wrong_way_count}", 
            (0, 0, 255) if wrong_way_count > 0 else (0, 255, 0), 0.6)
        txt(f"Lane Changes(60s): {lane_changes}", (200, 200, 100), 0.6)
        y += 3

        txt("--- Vehicle Mix ---", (200, 200, 255), 0.65)
        vehicle_mix = metrics['vehicle_mix']
        if vehicle_mix:
            for name, pct in sorted(vehicle_mix.items(), key=lambda item: item[1], reverse=True):
                txt(f"{name}: {pct:.1f}%", (180, 255, 180), 0.55)
        else:
            txt("No counted vehicles yet", (180, 180, 180), 0.55)
        y += 3

        pedestrian_anomaly = metrics['pedestrians_on_road'] > 0

        if pedestrian_anomaly:
            y += 3
            txt("!! ANOMALY DETECTED !!", (0, 0, 255), 0.7, True)
            txt("Pedestrian On Road", (0, 165, 255), 0.6, True)
        elif is_anomaly and anomaly_info:
            y += 3
            txt("!! ANOMALY DETECTED !!", (0, 0, 255), 0.7, True)
            txt(f"Type: {anomaly_info['type']}", (255, 100, 100), 0.55)
            txt(f"Severity: {anomaly_info['severity']:.2f} sigma", (255, 100, 100), 0.55)

        return metrics

    @staticmethod
    def draw_aoi_lines(frame, l1_start, l1_end, l2_start, l2_end):
        """Draw the Area of Interest lines"""
        cv2.line(frame, tuple(l1_start), tuple(l1_end), (0, 255, 255), 3)
        cv2.line(frame, tuple(l2_start), tuple(l2_end), (0, 255, 255), 3)
        cv2.putText(frame, "L1", tuple(l1_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "L2", tuple(l2_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    @staticmethod
    def draw_vehicle_info(frame, x1, y1, x2, y2, vehicle_id, cls_id, is_counted, 
                         speed_kmh=None, is_stopped=False, is_wrong_way=False, is_speeding=False):
        """Draw enhanced bounding box with status indicators"""
        # Determine color based on status
        if is_wrong_way:
            color = (0, 0, 255)  # Red for wrong way
        elif is_stopped:
            color = (255, 0, 255)  # Magenta for stopped
        elif is_speeding:
            color = (0, 165, 255)  # Orange for speeding
        elif is_counted:
            color = (0, 200, 255)  # Yellow for counted
        else:
            color = (0, 255, 0)  # Green for normal
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Build label
        label_parts = [f"{VEHICLE_LABELS.get(cls_id, 'Veh')} {vehicle_id}"]
        
        if is_counted:
            label_parts.append("COUNTED")
        if speed_kmh is not None and speed_kmh > 0:
            label_parts.append(f"{speed_kmh:.0f}km/h")
        if is_stopped:
            label_parts.append("STOPPED")
        if is_wrong_way:
            label_parts.append("WRONG WAY!")
        if is_speeding:
            label_parts.append("SPEEDING")
        
        label = " ".join(label_parts)
        
        # Draw label background
        (lw, lh_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh_ - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    @staticmethod
    def draw_pedestrian_info(frame, x1, y1, x2, y2, pedestrian_id):
        color = (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"PEDESTRIAN {pedestrian_id}"
        (lw, lh_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh_ - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ========== MAIN PROCESSING (ENHANCED) ==========
def process_aoi(frame, lane_tracker, current_time, aoi_contour, roi_geometry, prev_time):
    """Process the configured region with detection-line counting and all new features."""
    source_frame = frame.copy()
    include_polygon = roi_geometry["include_polygon"]

    if roi_geometry["mode"] == "SIDE_LINE":
        overlay = frame.copy()
        exclude_polygon = roi_geometry["exclude_polygon"]
        if len(exclude_polygon) >= 3:
            cv2.fillPoly(overlay, [exclude_polygon.astype(np.int32)], (20, 20, 20))
        if len(include_polygon) >= 3:
            cv2.fillPoly(overlay, [include_polygon.astype(np.int32)], (0, 80, 0))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        line_mid = (
            (Config.DIVIDER_LINE_START[0] + Config.DIVIDER_LINE_END[0]) // 2,
            (Config.DIVIDER_LINE_START[1] + Config.DIVIDER_LINE_END[1]) // 2,
        )
        det_mid = (
            (Config.DETECTION_LINE_START[0] + Config.DETECTION_LINE_END[0]) // 2,
            (Config.DETECTION_LINE_START[1] + Config.DETECTION_LINE_END[1]) // 2,
        )
        cv2.line(frame, Config.DIVIDER_LINE_START, Config.DIVIDER_LINE_END, (0, 0, 255), 3)
        cv2.putText(frame, "DIVIDER LINE", (line_mid[0] - 100, max(30, line_mid[1] - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.line(frame, Config.DETECTION_LINE_START, Config.DETECTION_LINE_END, (255, 0, 255), 3)
        cv2.putText(frame, "DETECTION LINE", (det_mid[0] - 110, max(30, det_mid[1] - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, "RIGHT SIDE ACTIVE", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, "LEFT SIDE IGNORED", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        direction_sign = None
    else:
        direction_sign = roi_geometry["include_sign"]

        overlay = frame.copy()
        cv2.fillPoly(overlay, [include_polygon.astype(np.int32)], (0, 80, 80))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        cv2.line(frame, Config.DETECTION_LINE_START, Config.DETECTION_LINE_END, (0, 0, 255), 3)
        cv2.putText(frame, "RED COUNT LINE", (Config.DETECTION_LINE_START[0] + 50, Config.DETECTION_LINE_START[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if len(Config.AOI_POINTS) >= 4:
            TrafficVisualizer.draw_aoi_lines(frame, Config.AOI_POINTS[0], Config.AOI_POINTS[1], Config.AOI_POINTS[2], Config.AOI_POINTS[3])

    # Run YOLO only on the configured ROI region.
    masked_frame = np.zeros_like(source_frame)
    cv2.fillPoly(masked_frame, [include_polygon.astype(np.int32)], (255, 255, 255))
    masked_frame = cv2.bitwise_and(source_frame, masked_frame)
    results = model(masked_frame, conf=Config.CONFIDENCE_THRESHOLD, imgsz=640, verbose=False)[0]

    current_centers = {}
    current_pedestrian_centers = {}
    lane_tracker.vehicles_in_roi = 0
    lane_tracker.pedestrians_on_road = 0
    lane_tracker.current_bbox_area = 0
    total_detections = 0
    aoi_detections = 0
    matched_previous_ids = set()
    matched_pedestrian_ids = set()
    
    # NEW: Tracking for new features
    time_delta = current_time - prev_time if prev_time > 0 else 0
    vehicle_statuses = {}  # vid -> {speed, stopped, wrong_way, speeding, lane_change}
    speeding_count = 0
    stopped_positions = []
    stopped_count = 0
    wrong_way_count = 0

    vehicle_boxes_in_aoi = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in Config.VEHICLE_IDS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        point = bbox_tracking_point(x1, y1, x2, y2)
        if cv2.pointPolygonTest(aoi_contour, point, False) >= 0:
            vehicle_boxes_in_aoi.append((x1, y1, x2, y2))

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in Config.VEHICLE_IDS and cls_id not in Config.PEDESTRIAN_IDS:
            continue

        total_detections += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = bbox_tracking_point(x1, y1, x2, y2)

        # Check if tracking point is inside AOI
        point = (cx, cy)
        if cv2.pointPolygonTest(aoi_contour, point, False) < 0:
            continue

        aoi_detections += 1

        if cls_id in Config.PEDESTRIAN_IDS:
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            pedestrian_box = (x1, y1, x2, y2)
            if not is_valid_pedestrian_detection(pedestrian_box, confidence, vehicle_boxes_in_aoi):
                continue

            pedestrian_id = None
            min_distance = float('inf')
            for pid, (px, py) in lane_tracker.pedestrian_previous_centers.items():
                if pid in matched_pedestrian_ids:
                    continue
                last_seen = lane_tracker.pedestrian_track_last_seen.get(pid, current_time)
                if current_time - last_seen > Config.PEDESTRIAN_TRACK_MAX_AGE_SECONDS:
                    continue
                distance = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if distance < Config.PEDESTRIAN_TRACKING_DISTANCE and distance < min_distance:
                    min_distance = distance
                    pedestrian_id = pid

            if pedestrian_id is None:
                pedestrian_id = lane_tracker.pedestrian_next_id
                lane_tracker.pedestrian_next_id += 1
                lane_tracker.pedestrian_track_hits[pedestrian_id] = 0

            lane_tracker.pedestrian_track_hits[pedestrian_id] = lane_tracker.pedestrian_track_hits.get(pedestrian_id, 0) + 1
            current_pedestrian_centers[pedestrian_id] = point
            lane_tracker.pedestrian_track_last_seen[pedestrian_id] = current_time
            matched_pedestrian_ids.add(pedestrian_id)

            if lane_tracker.pedestrian_track_hits[pedestrian_id] >= Config.PEDESTRIAN_CONFIRM_FRAMES:
                lane_tracker.pedestrians_on_road += 1
                if pedestrian_id not in lane_tracker.confirmed_pedestrian_ids:
                    lane_tracker.confirmed_pedestrian_ids.add(pedestrian_id)
                    lane_tracker.register_pedestrian_event(pedestrian_id, point, current_time)

                TrafficVisualizer.draw_pedestrian_info(frame, x1, y1, x2, y2, pedestrian_id)
                cv2.circle(frame, point, 4, (0, 165, 255), -1)
            continue

        lane_tracker.vehicles_in_roi += 1
        lane_tracker.current_bbox_area += max(0, x2 - x1) * max(0, y2 - y1)

        # ID assignment
        found_id = None
        min_distance = float('inf')
        for vid, (px, py) in lane_tracker.previous_centers.items():
            if vid in matched_previous_ids:
                continue
            last_seen = lane_tracker.track_last_seen.get(vid, current_time)
            if current_time - last_seen > Config.TRACK_MAX_AGE_SECONDS:
                continue
            distance = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if distance < Config.MIN_TRACKING_DISTANCE and distance < min_distance:
                min_distance = distance
                found_id = vid

        if found_id is None:
            found_id = lane_tracker.next_id
            lane_tracker.next_id += 1
            lane_tracker.vehicle_entry_times[found_id] = current_time

        lane_tracker.vehicle_class_by_id[found_id] = cls_id
        current_centers[found_id] = point
        lane_tracker.track_last_seen[found_id] = current_time
        matched_previous_ids.add(found_id)
        
        # NEW: Calculate speed and check for violations
        speed_kmh = 0
        prev_pos = lane_tracker.previous_positions.get(found_id)
        
        if prev_pos is not None and time_delta > 0:
            speed_kmh = lane_tracker.speed_estimator.estimate_speed(
                found_id, prev_pos, point, time_delta
            )
        
        # NEW: Check if stopped
        is_stopped = lane_tracker.stopped_detector.update(found_id, point, current_time)
        if is_stopped:
            stopped_positions.append(point)
        
        # NEW: Check wrong-way
        is_wrong_way = False
        if prev_pos is not None:
            is_wrong_way = lane_tracker.wrong_way_detector.detect_wrong_way(
                found_id, prev_pos, point, current_time
            )
        if is_wrong_way:
            wrong_way_count += 1
        
        # NEW: Check speeding
        is_speeding = lane_tracker.speed_estimator.is_speeding(speed_kmh)
        if is_speeding and speed_kmh > 0:
            speeding_count += 1
        
        # NEW: Check lane change
        lane_changed, change_info = lane_tracker.lane_change_detector.detect_lane_change(
            found_id, point, current_time
        )
        
        # Store status
        vehicle_statuses[found_id] = {
            'speed': speed_kmh,
            'stopped': is_stopped,
            'wrong_way': is_wrong_way,
            'speeding': is_speeding,
            'lane_change': lane_changed
        }
        
        # Update position history
        lane_tracker.previous_positions[found_id] = point
        lane_tracker.previous_time[found_id] = current_time

        # Reference-style crossing logic: previous point before line, current point at/after line.
        if found_id in lane_tracker.previous_centers and found_id not in lane_tracker.crossed_vehicle_ids:
            prev_center = lane_tracker.previous_centers[found_id]
            prev_dist = signed_distance_to_line(
                prev_center,
                Config.DETECTION_LINE_START,
                Config.DETECTION_LINE_END
            )
            curr_dist = signed_distance_to_line(
                (cx, cy),
                Config.DETECTION_LINE_START,
                Config.DETECTION_LINE_END
            )

            if roi_geometry["mode"] == "SIDE_LINE":
                crossed_line = (
                    (prev_dist < (-Config.LINE_CROSSING_BUFFER_PX) and curr_dist >= 0) or
                    (prev_dist > Config.LINE_CROSSING_BUFFER_PX and curr_dist <= 0)
                )
            else:
                prev_rel = prev_dist * direction_sign
                curr_rel = curr_dist * direction_sign
                crossed_line = prev_rel < (-Config.LINE_CROSSING_BUFFER_PX) and curr_rel >= 0

            if crossed_line:
                lane_tracker.register_crossing(found_id, cls_id, current_time)

        # Draw enhanced bounding box with status
        is_counted = found_id in lane_tracker.crossed_vehicle_ids
        status = vehicle_statuses.get(found_id, {})
        
        TrafficVisualizer.draw_vehicle_info(
            frame, x1, y1, x2, y2, found_id, cls_id, is_counted,
            speed_kmh=status.get('speed', 0),
            is_stopped=status.get('stopped', False),
            is_wrong_way=status.get('wrong_way', False),
            is_speeding=status.get('speeding', False)
        )
        
        # Draw tracking point
        cv2.circle(frame, point, 4, (255, 0, 0), -1)

    # NEW: Detect collisions
    collision_detected, collision_events = lane_tracker.accident_detector.detect_collision(
        current_centers, lane_tracker.previous_centers, current_time
    )
    
    # Draw collision alerts
    if collision_detected:
        for event in collision_events:
            loc = event['location']
            cv2.circle(frame, (int(loc[0]), int(loc[1])), 50, (0, 0, 255), 3)
            cv2.putText(frame, f"COLLISION! {event['severity']}", 
                       (int(loc[0]) - 80, int(loc[1]) - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # NEW: Calculate queue length
    queue_count, queue_length_px = lane_tracker.queue_detector.calculate_queue_length(
        current_centers, current_time
    )

    retained_centers = {}
    for vid, prev_point in lane_tracker.previous_centers.items():
        last_seen = lane_tracker.track_last_seen.get(vid, current_time)
        if vid not in current_centers and current_time - last_seen <= Config.TRACK_MAX_AGE_SECONDS:
            retained_centers[vid] = prev_point

    retained_centers.update(current_centers)
    lane_tracker.previous_centers = retained_centers

    stale_track_ids = [
        vid for vid, last_seen in lane_tracker.track_last_seen.items()
        if current_time - last_seen > Config.TRACK_MAX_AGE_SECONDS
    ]
    for vid in stale_track_ids:
        lane_tracker.track_last_seen.pop(vid, None)
        lane_tracker.previous_centers.pop(vid, None)
        lane_tracker.previous_positions.pop(vid, None)
        lane_tracker.previous_time.pop(vid, None)
    
    # NEW: Get lane change count
    lane_changes_60s = lane_tracker.lane_change_detector.get_lane_change_count(current_time, 60)

    unique_stopped_positions = []
    for position in stopped_positions:
        if not any(np.linalg.norm(np.array(position) - np.array(existing)) <= Config.STOPPED_ALERT_DEDUP_DISTANCE
                   for existing in unique_stopped_positions):
            unique_stopped_positions.append(position)
    stopped_count = len(unique_stopped_positions)

    retained_pedestrian_centers = {}
    for pid, prev_point in lane_tracker.pedestrian_previous_centers.items():
        last_seen = lane_tracker.pedestrian_track_last_seen.get(pid, current_time)
        if pid not in current_pedestrian_centers and current_time - last_seen <= Config.PEDESTRIAN_TRACK_MAX_AGE_SECONDS:
            retained_pedestrian_centers[pid] = prev_point

    retained_pedestrian_centers.update(current_pedestrian_centers)
    lane_tracker.pedestrian_previous_centers = retained_pedestrian_centers

    stale_pedestrian_ids = [
        pid for pid, last_seen in lane_tracker.pedestrian_track_last_seen.items()
        if current_time - last_seen > Config.PEDESTRIAN_TRACK_MAX_AGE_SECONDS
    ]
    for pid in stale_pedestrian_ids:
        lane_tracker.pedestrian_track_last_seen.pop(pid, None)
        lane_tracker.pedestrian_previous_centers.pop(pid, None)
        lane_tracker.pedestrian_track_hits.pop(pid, None)
        lane_tracker.confirmed_pedestrian_ids.discard(pid)

    if lane_tracker.pedestrians_on_road > 0:
        cv2.rectangle(frame, (460, 20), (980, 70), (0, 0, 255), -1)
        cv2.putText(frame, "ANOMALY DETECTED: PEDESTRIAN ON ROAD",
                    (475, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return (frame, total_detections, aoi_detections, queue_count, queue_length_px, 
            stopped_count, collision_detected, speeding_count, wrong_way_count, 
            lane_changes_60s)

# ========== DATA EXPORT (ENHANCED) ==========
class DataExporter:
    @staticmethod
    def export_results(lane, video_duration):
        """Export comprehensive analysis results with new features."""
        import os

        if not Config.EXPORT_DATA:
            return

        output_dir = ensure_output_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        density_values = [d for _, d in lane.density_history]
        flow_values = [f for _, f in lane.flow_history]
        occupancy_values = [o for _, o in lane.occupancy_history]
        active_aoi_points = Config.ACTIVE_AOI_POINTS if Config.ACTIVE_AOI_POINTS is not None else Config.AOI_POINTS
        vehicle_class_counts = {
            VEHICLE_LABELS.get(cls_id, str(cls_id)): count
            for cls_id, count in sorted(lane.vehicle_classes.items())
        }

        # ── Summary JSON (Enhanced) ──
        summary = {
            'timestamp': ts,
            'video_duration': video_duration,
            'roi_mode': Config.ROI_MODE,
            'process_side': Config.PROCESS_SIDE,
            'aoi_points': active_aoi_points.tolist() if len(active_aoi_points) > 0 else [],
            'divider_line': {
                'start': list(Config.DIVIDER_LINE_START),
                'end': list(Config.DIVIDER_LINE_END),
            },
            'detection_line': {
                'start': list(Config.DETECTION_LINE_START),
                'end': list(Config.DETECTION_LINE_END),
            },
            'lane': {
                'total_vehicles': lane.get_total_crossings(),
                'vehicle_classes': vehicle_class_counts,
                'vehicle_mix_percent': lane.get_vehicle_mix(),
                'avg_headways': {
                    w: lane.calculate_avg_headway(video_duration, w)
                    for w in Config.TIME_WINDOWS
                },
                'avg_travel_time': float(np.mean(lane.total_travel_time)) if lane.total_travel_time else 0,
                'avg_density': float(np.mean(density_values)) if density_values else 0,
                'peak_density': float(np.max(density_values)) if density_values else 0,
                'avg_flow': float(np.mean(flow_values)) if flow_values else 0,
                'peak_flow': float(np.max(flow_values)) if flow_values else 0,
                'avg_occupancy': float(np.mean(occupancy_values)) if occupancy_values else 0,
                'peak_occupancy': float(np.max(occupancy_values)) if occupancy_values else 0,
            },
            # NEW: Safety statistics
            'safety': {
                'total_collisions': len(lane.accident_detector.collision_events),
                'total_stopped_alerts': len(lane.stopped_detector.stopped_alerts),
                'total_pedestrian_events': len(lane.pedestrian_events),
                'total_wrong_way_events': len(lane.wrong_way_detector.wrong_way_events),
                'total_lane_changes': len(lane.lane_change_detector.lane_change_events),
            }
        }

        with open(build_output_path(f"mehsana_analysis_summary_{ts}.json"), 'w') as f:
            json.dump(summary, f, indent=4)

        # ── Time-series CSV ──
        if lane.headways:
            pd.DataFrame(list(lane.headways), columns=['time', 'headway']) \
              .to_csv(build_output_path(f"mehsana_headways_{ts}.csv"), index=False)
        if lane.density_history:
            pd.DataFrame(list(lane.density_history), columns=['time', 'density']) \
              .to_csv(build_output_path(f"mehsana_density_{ts}.csv"), index=False)
        if lane.flow_history:
            pd.DataFrame(list(lane.flow_history), columns=['time', 'flow']) \
              .to_csv(build_output_path(f"mehsana_flow_{ts}.csv"), index=False)
        if lane.occupancy_history:
            pd.DataFrame(list(lane.occupancy_history), columns=['time', 'occupancy']) \
              .to_csv(build_output_path(f"mehsana_occupancy_{ts}.csv"), index=False)
        if lane.crossing_events:
            events_df = pd.DataFrame(
                list(lane.crossing_events),
                columns=['time', 'vehicle_id', 'class_id', 'headway', 'travel_time']
            )
            events_df['vehicle_type'] = events_df['class_id'].map(
                lambda cls_id: VEHICLE_LABELS.get(cls_id, str(cls_id))
            )
            events_df = events_df[['time', 'vehicle_id', 'class_id', 'vehicle_type', 'headway', 'travel_time']]
            events_df.to_csv(build_output_path(f"mehsana_vehicle_events_{ts}.csv"), index=False)
        
        # NEW: Export safety events
        if lane.accident_detector.collision_events:
            pd.DataFrame(lane.accident_detector.collision_events) \
              .to_csv(build_output_path(f"mehsana_collisions_{ts}.csv"), index=False)
        
        if lane.stopped_detector.stopped_alerts:
            pd.DataFrame(list(lane.stopped_detector.stopped_alerts)) \
              .to_csv(build_output_path(f"mehsana_stopped_vehicles_{ts}.csv"), index=False)

        if lane.pedestrian_events:
            pd.DataFrame(list(lane.pedestrian_events)) \
              .to_csv(build_output_path(f"mehsana_pedestrians_{ts}.csv"), index=False)
        
        if lane.wrong_way_detector.wrong_way_events:
            pd.DataFrame(list(lane.wrong_way_detector.wrong_way_events)) \
              .to_csv(build_output_path(f"mehsana_wrong_way_{ts}.csv"), index=False)
        
        if lane.lane_change_detector.lane_change_events:
            pd.DataFrame(list(lane.lane_change_detector.lane_change_events)) \
              .to_csv(build_output_path(f"mehsana_lane_changes_{ts}.csv"), index=False)
        
        if lane.queue_detector.queue_history:
            pd.DataFrame(list(lane.queue_detector.queue_history), 
                        columns=['time', 'queue_count', 'queue_length_pixels']) \
              .to_csv(build_output_path(f"mehsana_queue_{ts}.csv"), index=False)

        print(f"\n[+] Data exported to {output_dir}")

# ========== MAIN EXECUTION (ENHANCED) ==========
def main():
    global model

    print("Loading YOLO model...")
    model = YOLO(Config.YOLO_MODEL)

    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video {Config.VIDEO_PATH}")
        return

    # Ensure full resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Single lane tracker for AOI
    aoi_lane = AdvancedLaneTracker("MEHSANA AOI", 0)
    
    analysis_start_time = time.time()
    frame_count = 0
    processed_frames = 0
    roi_geometry = None
    aoi_contour = None
    aoi_length_pixels = 0.0
    aoi_area_pixels = 0.0
    prev_time = 0

    print("\n" + "=" * 70)
    print("  MEHSANA TRAFFIC VEHICLE DETECTION - ENHANCED AOI ANALYSIS")
    print("=" * 70)
    print(f"  Video          : {Config.VIDEO_PATH}")
    print(f"  Total Frames   : {total_frames}")
    print(f"  FPS            : {fps}")
    print(f"  Speed Mult     : {Config.SPEED_MULTIPLIER}x")
    print(f"  ROI Mode       : {Config.ROI_MODE}")
    if Config.ROI_MODE == "SIDE_LINE":
        print(f"  Active Side    : {Config.PROCESS_SIDE}")
        print(f"  Divider Line   : {Config.DIVIDER_LINE_START} -> {Config.DIVIDER_LINE_END}")
        print(f"  Detect Line    : {Config.DETECTION_LINE_START} -> {Config.DETECTION_LINE_END}")
    print(f"  Output Folder  : {ensure_output_dir()}")
    print("\n  NEW FEATURES ENABLED:")
    print("    ✓ Speed Estimation & Speeding Detection")
    print("    ✓ Stopped Vehicle Detection")
    print("    ✓ Queue Length Monitoring")
    print("    ✓ Accident/Collision Detection")
    print("    ✓ Wrong-Way Vehicle Detection")
    print("    ✓ Lane Change Detection")
    print("    ✓ Pedestrian On-Road Detection")
    print("\n  Press 'q' to quit, 's' to save screenshot")
    print("=" * 70 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % Config.SPEED_MULTIPLIER != 0:
            continue

        processed_frames += 1
        h, w = frame.shape[:2]
        current_time = frame_count / fps if fps > 0 else processed_frames

        if roi_geometry is None:
            roi_geometry, aoi_contour = prepare_roi_geometry(frame.shape)
            if Config.ROI_MODE == "SIDE_LINE":
                aoi_length_pixels = estimate_line_length_pixels(
                    Config.DETECTION_LINE_START,
                    Config.DETECTION_LINE_END
                )
            else:
                aoi_length_pixels = estimate_aoi_length_pixels(roi_geometry["include_polygon"])
            aoi_area_pixels = polygon_area_pixels(roi_geometry["include_polygon"])
            aoi_lane.wrong_way_detector.set_reference_motion(
                detection_line_forward_vector(
                    Config.DETECTION_LINE_START,
                    Config.DETECTION_LINE_END,
                    roi_geometry.get("count_direction_sign", 1.0)
                )
            )

        # Process AOI with all new features
        (frame, total_detections, aoi_detections, queue_count, queue_length_px,
         stopped_count, collision_detected, speeding_count, wrong_way_count,
         lane_changes_60s) = process_aoi(
            frame,
            aoi_lane,
            current_time,
            aoi_contour,
            roi_geometry,
            prev_time
        )
        
        prev_time = current_time

        # ── Compute metrics & predictions ──
        metrics = aoi_lane.update_metrics(
            current_time,
            aoi_length_pixels,
            aoi_area_pixels,
            Config.PIXELS_PER_METER
        )
        pred_density, congestion_prob = aoi_lane.predict_congestion(current_time)

        # ── Draw enhanced metrics panel ──
        TrafficVisualizer.draw_metrics_panel(
            frame, aoi_lane, 10, 50, current_time,
            metrics, pred_density, congestion_prob,
            queue_count, queue_length_px, stopped_count,
            collision_detected, speeding_count, wrong_way_count,
            lane_changes_60s
        )

        # Progress bar
        progress = (frame_count / total_frames) * 100
        bar_w = w - 40
        cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 40),
                      (20 + int(bar_w * progress / 100), h - 20), (0, 255, 0), -1)
        
        status_text = (f"Progress: {progress:.1f}% | Det: {total_detections}/{aoi_detections} | "
                      f"Q:{queue_count} | Stop:{stopped_count} | Ped:{metrics['pedestrians_on_road']} | Speed:{speeding_count} | "
                      f"WW:{wrong_way_count}")
        if collision_detected:
            status_text += " | COLLISION!"
        
        cv2.putText(frame, status_text,
                    (w // 2 - 300, h - 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Resize for display if too large
        display_frame = cv2.resize(frame, (1920, 1080)) if w > 1920 or h > 1080 else frame

        cv2.imshow("Mehsana Traffic - Enhanced Monitoring", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            sname = build_output_path(f"screenshot_{int(current_time)}.jpg")
            cv2.imwrite(sname, frame)
            print(f"Screenshot saved: {sname}")

    # ── Cleanup ──
    processing_time = time.time() - analysis_start_time
    final_video_time = frame_count / fps if fps > 0 else processed_frames
    cap.release()
    cv2.destroyAllWindows()

    # ══════════════════════════════════════════════════════════════
    #  FINAL STATISTICS (ENHANCED)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL STATISTICS - MEHSANA AOI (ENHANCED)")
    print("=" * 70)
    print(f"  Video Duration   : {final_video_time:.2f} s")
    print(f"  Processing Time  : {processing_time:.2f} s")
    print(f"  Frames Processed : {processed_frames}/{total_frames}")
    print("=" * 70)

    lane = aoi_lane
    counted_classes = {
        VEHICLE_LABELS.get(k, str(k)): v
        for k, v in lane.vehicle_classes.items()
    }
    print(f"\n  {lane.lane_name}")
    print("  " + "-" * 66)
    print(f"    Total Vehicles Counted : {lane.get_total_crossings()}")
    print(f"    Vehicle Classes        : {counted_classes}")
    if counted_classes:
        print(f"    Vehicle Mix            : {lane.get_vehicle_mix()}")

    print("\n    Headway Analysis:")
    for w in Config.TIME_WINDOWS:
        a = lane.calculate_avg_headway(final_video_time, w)
        v = lane.calculate_headway_variance(final_video_time, w)
        print(f"      {w:>4}s window : Avg={a:.2f}s, Var={v:.2f}")

    if lane.total_travel_time:
        print(f"\n    Travel Time:")
        print(f"      Average : {np.mean(lane.total_travel_time):.2f}s")

    if lane.density_history:
        densities = [d for _, d in lane.density_history]
        print(f"\n    Density:")
        print(f"      Average : {np.mean(densities):.2f} veh/km")
        print(f"      Max     : {np.max(densities):.2f} veh/km")

    if lane.flow_history:
        flows = [f for _, f in lane.flow_history if f > 0]
        if flows:
            print(f"\n    Flow:")
            print(f"      Average : {np.mean(flows):.0f} veh/h")
            print(f"      Max     : {np.max(flows):.0f} veh/h")

    if lane.occupancy_history:
        occupancies = [o for _, o in lane.occupancy_history]
        if occupancies:
            print(f"\n    Occupancy:")
            print(f"      Average : {np.mean(occupancies):.2f}%")
            print(f"      Max     : {np.max(occupancies):.2f}%")
    
    # NEW: Safety statistics
    print("\n    === SAFETY & INCIDENTS ===")
    print(f"      Total Collisions      : {len(lane.accident_detector.collision_events)}")
    print(f"      Stopped Vehicle Alerts: {len(lane.stopped_detector.stopped_alerts)}")
    print(f"      Pedestrian Events     : {len(lane.pedestrian_events)}")
    print(f"      Wrong-Way Events      : {len(lane.wrong_way_detector.wrong_way_events)}")
    print(f"      Lane Changes          : {len(lane.lane_change_detector.lane_change_events)}")
    
    if lane.queue_detector.queue_history:
        queue_counts = [c for _, c, _ in lane.queue_detector.queue_history]
        print(f"\n    Queue Analysis:")
        print(f"      Average Queue : {np.mean(queue_counts):.1f} vehicles")
        print(f"      Max Queue     : {np.max(queue_counts):.0f} vehicles")

    print("\n" + "=" * 70)

    # Export
    DataExporter.export_results(aoi_lane, final_video_time)
    print("\nEnhanced Analysis complete!")

if __name__ == "__main__":
    main()
