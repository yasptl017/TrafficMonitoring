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

    # Calibration
    PIXELS_PER_METER = 30

    # Thresholds
    CONGESTION_DENSITY_THRESHOLD = 50  # veh/km
    MIN_TRACKING_DISTANCE = 150  # pixels
    LINE_CROSSING_BUFFER_PX = 4
    TRACK_MAX_AGE_SECONDS = 0.8
    TRACK_POINT_BOTTOM_OFFSET_RATIO = 0.15

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

    Config.ACTIVE_AOI_POINTS = aoi_points.copy()
    aoi_contour = aoi_points.reshape((-1, 1, 2))
    return roi_geometry, aoi_contour

# ========== ADVANCED LANE TRACKER ==========
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
        self.current_bbox_area = 0
        self.total_travel_time = []
        self.vehicle_entry_times = {}
        self.vehicle_class_by_id = {}
        self.vehicle_classes = defaultdict(int)

        # Time series data for prediction
        self.time_series_data = []

        # Anomaly detection
        self.anomalies = []

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

# ========== TRAFFIC VISUALIZER ==========
class TrafficVisualizer:
    @staticmethod
    def draw_metrics_panel(frame, lane, x_off, y_start, current_time,
                           metrics, pred_density, congestion_prob):
        headway_stats = {}
        for w in Config.TIME_WINDOWS:
            headway_stats[w] = (lane.calculate_avg_headway(current_time, w),
                                lane.calculate_headway_variance(current_time, w))

        los, los_color = lane.calculate_level_of_service(metrics)
        is_anomaly, anomaly_info = lane.detect_anomaly(metrics, current_time)

        panel_h = 520
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_off, y_start),
                      (x_off + 400, y_start + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (x_off, y_start),
                      (x_off + 400, y_start + panel_h), (100, 100, 100), 2)

        y = y_start + 30
        lh = 28

        def txt(text, color=(255, 255, 255), scale=0.6, bold=False):
            nonlocal y
            cv2.putText(frame, text, (x_off + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1)
            y += lh

        txt(f"=== {lane.lane_name} ===", (255, 255, 0), 0.8, True)
        y += 5
        txt(f"Total Count: {lane.get_total_crossings()}", (0, 255, 255), 0.7)
        txt(f"Current Vehicles: {metrics['vehicles_in_roi']}", (150, 255, 150), 0.65)
        txt(f"Occupancy: {metrics['occupancy']:.1f}%", (255, 180, 120), 0.65)
        txt(f"Avg Travel Time: {metrics['avg_travel_time']:.2f}s", (255, 150, 150), 0.65)
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

        txt("--- Vehicle Mix ---", (200, 200, 255), 0.65)
        vehicle_mix = metrics['vehicle_mix']
        if vehicle_mix:
            for name, pct in sorted(vehicle_mix.items(), key=lambda item: item[1], reverse=True):
                txt(f"{name}: {pct:.1f}%", (180, 255, 180), 0.55)
        else:
            txt("No counted vehicles yet", (180, 180, 180), 0.55)
        y += 3

        if pred_density is not None and congestion_prob is not None:
            txt("--- Trend Prediction (60s) ---", (200, 200, 255), 0.65)
            pc = (0, 255, 0) if congestion_prob < 0.5 else \
                 (0, 165, 255) if congestion_prob < 0.8 else (0, 0, 255)
            txt(f"Pred Density: {pred_density:.1f} veh/km", (200, 255, 255), 0.6)
            txt(f"Congestion Prob: {congestion_prob*100:.0f}%", pc, 0.65)

        if is_anomaly and anomaly_info:
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

# ========== MAIN PROCESSING ==========
def process_aoi(frame, lane_tracker, current_time, aoi_contour, roi_geometry):
    """Process the configured region with detection-line counting."""
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
    lane_tracker.vehicles_in_roi = 0
    lane_tracker.current_bbox_area = 0
    total_detections = 0
    aoi_detections = 0
    matched_previous_ids = set()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in Config.VEHICLE_IDS:
            continue

        total_detections += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = bbox_tracking_point(x1, y1, x2, y2)

        # Check if tracking point is inside AOI
        point = (cx, cy)
        if cv2.pointPolygonTest(aoi_contour, point, False) < 0:
            continue

        aoi_detections += 1
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

        # Draw bounding box
        color = (0, 200, 255) if found_id in lane_tracker.crossed_vehicle_ids else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{VEHICLE_LABELS.get(cls_id, 'Veh')} {found_id}"
        if found_id in lane_tracker.crossed_vehicle_ids:
            label += " COUNTED"

        (lw, lh_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh_ - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(frame, point, 4, (255, 0, 0), -1)

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

    return frame, total_detections, aoi_detections

# ========== DATA EXPORT ==========
class DataExporter:
    @staticmethod
    def export_results(lane, video_duration):
        """Export comprehensive analysis results."""
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

        # ── Summary JSON ──
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

        print(f"\n[+] Data exported to {output_dir}")

# ========== MAIN EXECUTION ==========
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

    print("\n" + "=" * 70)
    print("  MEHSANA TRAFFIC VEHICLE DETECTION - AOI ANALYSIS")
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
    print("  Press 'q' to quit, 's' to save screenshot")
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

        # Process AOI
        frame, total_detections, aoi_detections = process_aoi(
            frame,
            aoi_lane,
            current_time,
            aoi_contour,
            roi_geometry
        )

        # ── Compute metrics & predictions ──
        metrics = aoi_lane.update_metrics(
            current_time,
            aoi_length_pixels,
            aoi_area_pixels,
            Config.PIXELS_PER_METER
        )
        pred_density, congestion_prob = aoi_lane.predict_congestion(current_time)

        # ── Draw metrics panel ──
        TrafficVisualizer.draw_metrics_panel(
            frame, aoi_lane, 10, 50, current_time,
            metrics, pred_density, congestion_prob
        )

        # Progress bar
        progress = (frame_count / total_frames) * 100
        bar_w = w - 40
        cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 40),
                      (20 + int(bar_w * progress / 100), h - 20), (0, 255, 0), -1)
        cv2.putText(frame, f"Progress: {progress:.1f}% | Total Det: {total_detections} | AOI Det: {aoi_detections}",
                    (w // 2 - 150, h - 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Resize for display if too large
        display_frame = cv2.resize(frame, (1920, 1080)) if w > 1920 or h > 1080 else frame

        cv2.imshow("Mehsana Traffic Vehicle Detection - AOI", display_frame)

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
    #  FINAL STATISTICS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL STATISTICS - MEHSANA AOI")
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

    print("\n" + "=" * 70)

    # Export
    DataExporter.export_results(aoi_lane, final_video_time)
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
