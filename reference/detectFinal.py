import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque, defaultdict
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import json

# ========== CONFIGURATION ==========
class Config:
    # Video Settings
    SPEED_MULTIPLIER = 4
    VIDEO_PATH = "v5.mp4"

    # Detection Settings
    YOLO_MODEL = "yolov5s.pt"
    CONFIDENCE_THRESHOLD = 0.35
    VEHICLE_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # Calibration
    PIXELS_PER_METER = 30
    BASELINE_OFFSET = 0.5

    # Thresholds
    QUEUE_SPEED_THRESHOLD = 10  # km/h
    CONGESTION_DENSITY_THRESHOLD = 50  # veh/km
    MIN_TRACKING_DISTANCE = 80  # pixels

    # Time Windows
    TIME_WINDOWS = [20, 40, 60, 120]  # seconds

    # Export Settings
    EXPORT_DATA = True
    EXPORT_PATH = "traffic_analysis_results/"

    # ── Traffic Signal Settings ──
    SIGNAL_DISTANCE_M = 200       # distance to next traffic signal (meters)
    SIGNAL_CYCLE_TIME = 120       # total cycle length (seconds)
    SIGNAL_GREEN_TIME = 45        # green phase duration (seconds)
    SIGNAL_RED_TIME = 65          # red phase duration (seconds)
    SIGNAL_AMBER_TIME = 10        # amber phase duration (seconds)
    SATURATION_FLOW = 1800        # saturation flow rate (veh/hr/lane at green)
    JAM_DENSITY = 150             # jam density (veh/km at standstill)
    FREE_FLOW_SPEED = 60          # free-flow speed (km/h)
    AVG_VEHICLE_LENGTH_M = 5.0    # average vehicle length including gap (meters)

# ========== ADVANCED LANE TRACKER ==========
class AdvancedLaneTracker:
    def __init__(self, lane_name, lane_index):
        self.lane_name = lane_name
        self.lane_index = lane_index

        # Basic tracking
        self.previous_centers = {}
        self.next_id = 0
        self.vehicle_count = 0
        self.last_cross_time = None

        # Advanced data structures
        self.headways = deque(maxlen=2000)
        self.speed_measurements = deque(maxlen=500)
        self.vehicle_positions = {}
        self.vehicle_speeds = {}
        self.density_history = deque(maxlen=1000)
        self.flow_history = deque(maxlen=1000)

        # Queue tracking
        self.queue_vehicles = set()
        self.queue_length_history = deque(maxlen=500)

        # Statistical tracking
        self.vehicles_in_roi = 0
        self.total_travel_time = []
        self.vehicle_entry_times = {}
        self.vehicle_classes = defaultdict(int)

        # Time series data for prediction
        self.time_series_data = []

        # Anomaly detection
        self.anomalies = []

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

    def calculate_speed(self, vehicle_id, current_y, current_time, pixels_per_meter):
        """Calculate vehicle speed in km/h"""
        if vehicle_id not in self.vehicle_positions:
            self.vehicle_positions[vehicle_id] = deque(maxlen=15)

        self.vehicle_positions[vehicle_id].append((current_time, current_y))

        positions = self.vehicle_positions[vehicle_id]
        if len(positions) >= 3:
            time_diff = positions[-1][0] - positions[0][0]
            if time_diff > 0.5:  # At least 0.5 seconds
                distance_pixels = abs(positions[-1][1] - positions[0][1])
                distance_meters = distance_pixels / pixels_per_meter
                speed_mps = distance_meters / time_diff
                speed_kmh = speed_mps * 3.6

                # Validate speed (0-200 km/h)
                if 0 <= speed_kmh <= 200:
                    self.vehicle_speeds[vehicle_id] = speed_kmh
                    return speed_kmh

        return self.vehicle_speeds.get(vehicle_id, None)

    def update_metrics(self, current_time, roi_height, pixels_per_meter):
        """Calculate comprehensive traffic metrics"""
        # Density (vehicles/km)
        roi_length_km = roi_height / pixels_per_meter / 1000
        density = self.vehicles_in_roi / roi_length_km if roi_length_km > 0 else 0
        self.density_history.append((current_time, density))

        # Flow (vehicles/hour)
        flow = 0
        recent_crosses = sum(1 for (t, _) in self.headways if current_time - t <= 60)
        if recent_crosses > 0:
            flow = (recent_crosses / 60) * 3600
        self.flow_history.append((current_time, flow))

        # Average speed
        avg_speed = 0
        if len(self.speed_measurements) > 0:
            avg_speed = np.mean(list(self.speed_measurements))

        # Queue metrics
        queue_length = len(self.queue_vehicles)
        self.queue_length_history.append((current_time, queue_length))

        # Space mean speed (harmonic mean for better traffic flow representation)
        if len(self.speed_measurements) > 0:
            valid_speeds = [s for s in self.speed_measurements if s > 0]
            if valid_speeds:
                space_mean_speed = len(valid_speeds) / sum(1/s for s in valid_speeds)
            else:
                space_mean_speed = 0
        else:
            space_mean_speed = 0

        return {
            'density': density,
            'flow': flow,
            'avg_speed': avg_speed,
            'space_mean_speed': space_mean_speed,
            'queue_length': queue_length,
            'vehicles_in_roi': self.vehicles_in_roi
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


# ========== TRAFFIC SIGNAL PREDICTOR ==========
class TrafficSignalPredictor:
    """
    Predicts congestion and traffic behavior based on proximity to a
    downstream traffic signal.  Uses classical traffic-flow theory:
      - Webster's delay model
      - Lighthill-Whitham-Richards shockwave model
      - Robertson's platoon dispersion model
      - Queue storage / spillback analysis
    """

    def __init__(self):
        self.signal_distance_m = Config.SIGNAL_DISTANCE_M
        self.cycle_time = Config.SIGNAL_CYCLE_TIME
        self.green_time = Config.SIGNAL_GREEN_TIME
        self.red_time = Config.SIGNAL_RED_TIME
        self.amber_time = Config.SIGNAL_AMBER_TIME
        self.saturation_flow = Config.SATURATION_FLOW
        self.jam_density = Config.JAM_DENSITY
        self.free_flow_speed = Config.FREE_FLOW_SPEED
        self.vehicle_length_m = Config.AVG_VEHICLE_LENGTH_M

        # History for trend analysis
        self.prediction_history = deque(maxlen=500)

    # ── Signal Phase ─────────────────────────────────────────────
    def get_signal_phase(self, current_time):
        """Determine simulated signal phase based on cycle timing."""
        pos = current_time % self.cycle_time
        if pos < self.green_time:
            return 'GREEN', self.green_time - pos
        elif pos < self.green_time + self.amber_time:
            return 'AMBER', self.green_time + self.amber_time - pos
        else:
            return 'RED', self.cycle_time - pos

    # ── Queue Prediction ─────────────────────────────────────────
    def predict_queue_at_signal(self, flow_vph):
        """Predict queue build-up at the signal during red."""
        arrival_rate = flow_vph / 3600.0                  # veh/s
        discharge_rate = self.saturation_flow / 3600.0    # veh/s

        # Vehicles that arrive during red
        queue_at_red_end = arrival_rate * self.red_time

        # During green: queue drains + new arrivals
        vehicles_to_serve = queue_at_red_end + arrival_rate * self.green_time
        vehicles_served = min(discharge_rate * self.green_time, vehicles_to_serve)
        residual_queue = max(0.0, vehicles_to_serve - vehicles_served)

        # Physical queue length (meters)
        queue_length_m = queue_at_red_end * self.vehicle_length_m

        return {
            'queue_at_red_end_veh': round(queue_at_red_end, 1),
            'queue_length_m': round(queue_length_m, 1),
            'vehicles_served_per_cycle': round(vehicles_served, 1),
            'residual_queue_veh': round(residual_queue, 1),
            'overflow': residual_queue > 0.5
        }

    # ── Webster's Delay ──────────────────────────────────────────
    def estimate_webster_delay(self, flow_vph):
        """Webster's optimised delay for signalised intersections."""
        C = self.cycle_time
        g = self.green_time
        s = self.saturation_flow

        capacity = s * g / C if C > 0 else 1
        x = min(flow_vph / capacity, 0.99) if capacity > 0 else 0   # degree of saturation
        lam = g / C if C > 0 else 0

        # Uniform delay d1
        denom = 1 - lam * x
        d1 = (C * (1 - lam) ** 2) / (2 * max(denom, 0.01))

        # Over-saturation / random delay d2
        if flow_vph > 0 and x < 0.99:
            d2 = (x ** 2) / (2 * (flow_vph / 3600) * (1 - x))
        else:
            d2 = 0

        total_delay = d1 + d2
        return {
            'uniform_delay_s': round(d1, 1),
            'random_delay_s': round(d2, 1),
            'total_delay_s': round(total_delay, 1),
            'degree_of_saturation': round(x, 3),
            'v_c_ratio': round(x, 3)
        }

    # ── Shockwave (LWR Model) ───────────────────────────────────
    def predict_shockwave(self, density, avg_speed):
        """Lighthill-Whitham-Richards backward shockwave from red signal."""
        if density <= 0 or avg_speed <= 0:
            return {
                'shockwave_speed_kmh': 0,
                'shockwave_reach_m': 0,
                'reaches_observer': False,
                'time_to_reach_s': float('inf')
            }

        approach_flow = density * avg_speed          # veh/h
        if (self.jam_density - density) != 0:
            w = abs((0 - approach_flow) / (self.jam_density - density))   # km/h
        else:
            w = 0

        reach_m = w * (self.red_time / 3600) * 1000   # metres in one red phase
        reaches = reach_m >= self.signal_distance_m
        t2r = (self.signal_distance_m / (w * 1000 / 3600)) if w > 0 else float('inf')

        return {
            'shockwave_speed_kmh': round(w, 1),
            'shockwave_reach_m': round(reach_m, 1),
            'reaches_observer': reaches,
            'time_to_reach_s': round(t2r, 1) if t2r != float('inf') else t2r
        }

    # ── Travel-to-Signal ─────────────────────────────────────────
    def predict_travel_to_signal(self, avg_speed):
        """Estimate travel time from current observation to signal."""
        if avg_speed > 0:
            tt = (self.signal_distance_m / 1000) / (avg_speed / 3600)
        else:
            tt = float('inf')
        return {
            'travel_time_s': round(tt, 1) if tt != float('inf') else tt,
            'distance_m': self.signal_distance_m
        }

    # ── Arrival Phase Prediction ─────────────────────────────────
    def predict_arrival_phase(self, current_time, avg_speed):
        """Predict what signal phase vehicles will encounter on arrival."""
        travel = self.predict_travel_to_signal(avg_speed)
        tt = travel['travel_time_s']
        if tt == float('inf'):
            return 'UNKNOWN', 0, tt

        arrival_time = current_time + tt
        phase, remaining = self.get_signal_phase(arrival_time)
        return phase, round(remaining, 1), round(tt, 1)

    # ── Platoon Dispersion (Robertson) ───────────────────────────
    def predict_platoon_dispersion(self, flow_vph, avg_speed):
        """Robertson's platoon dispersion model."""
        alpha = 0.35
        beta = 0.80
        tt = (self.signal_distance_m / 1000) / (avg_speed / 3600) if avg_speed > 0 else 0

        F = 1 / (1 + alpha * beta * tt) if tt > 0 else 1.0
        peak_flow = flow_vph * F

        return {
            'smoothing_factor_F': round(F, 3),
            'dispersion_pct': round((1 - F) * 100, 1),
            'peak_flow_at_signal_vph': round(peak_flow, 0),
            'travel_time_s': round(tt, 1)
        }

    # ── Spillback Risk ───────────────────────────────────────────
    def estimate_spillback_risk(self, flow_vph):
        """Assess probability that queue extends back to observation point."""
        q = self.predict_queue_at_signal(flow_vph)
        max_storage = (self.signal_distance_m / 1000) * self.jam_density  # vehicles
        risk = min((q['queue_at_red_end_veh'] / max(max_storage, 0.01)) * 100, 100)

        if q['residual_queue_veh'] > 0.5:
            cycles_to_spill = max(0, (max_storage - q['queue_at_red_end_veh'])) / q['residual_queue_veh']
        else:
            cycles_to_spill = float('inf')

        return {
            'spillback_risk_pct': round(risk, 1),
            'max_storage_veh': round(max_storage, 1),
            'expected_queue_veh': q['queue_at_red_end_veh'],
            'cycles_to_spillback': round(cycles_to_spill, 1) if cycles_to_spill != float('inf') else cycles_to_spill
        }

    # ── Green Wave Recommendation ────────────────────────────────
    def green_wave_speed(self):
        """Optimal speed to arrive at the signal during green."""
        offset = self.green_time / self.cycle_time if self.cycle_time > 0 else 0
        if offset > 0 and self.cycle_time > 0:
            v_ms = self.signal_distance_m / (self.cycle_time * offset)
            v_kmh = v_ms * 3.6
        else:
            v_kmh = 0
        return {
            'recommended_speed_kmh': round(v_kmh, 1),
            'signal_offset_ratio': round(offset, 3)
        }

    # ── Stop-and-Go Wave Frequency ───────────────────────────────
    def predict_stop_go_waves(self, density, avg_speed):
        """Estimate frequency of stop-and-go oscillations near the signal."""
        if density <= 0 or avg_speed <= 0:
            return {'wave_frequency_per_min': 0, 'severity': 'None'}

        # Critical density ~ 0.5 * jam_density
        critical_density = self.jam_density * 0.5
        ratio = density / critical_density

        if ratio < 0.6:
            return {'wave_frequency_per_min': 0, 'severity': 'None'}
        elif ratio < 0.8:
            freq = (ratio - 0.6) * 5   # 0-1 waves/min
            return {'wave_frequency_per_min': round(freq, 1), 'severity': 'Low'}
        elif ratio < 1.0:
            freq = 1 + (ratio - 0.8) * 10  # 1-3 waves/min
            return {'wave_frequency_per_min': round(freq, 1), 'severity': 'Moderate'}
        else:
            freq = 3 + (ratio - 1.0) * 5
            return {'wave_frequency_per_min': round(min(freq, 8), 1), 'severity': 'High'}

    # ── Fuel / Emission Impact ───────────────────────────────────
    def estimate_delay_impact(self, total_delay_s, vehicles_in_roi):
        """Rough estimate of delay cost and extra fuel due to idling."""
        # Average idle fuel consumption ≈ 0.8 L/hr for a car
        idle_fuel_lph = 0.8
        extra_fuel_per_veh = idle_fuel_lph * (total_delay_s / 3600)
        total_extra_fuel = extra_fuel_per_veh * vehicles_in_roi
        # CO2 ≈ 2.31 kg per litre petrol
        co2_kg = total_extra_fuel * 2.31
        return {
            'extra_fuel_per_veh_L': round(extra_fuel_per_veh, 4),
            'total_extra_fuel_L': round(total_extra_fuel, 4),
            'co2_emission_kg': round(co2_kg, 4)
        }

    # ── Comprehensive Prediction ─────────────────────────────────
    def comprehensive_prediction(self, current_time, flow_vph, density, avg_speed, vehicles_in_roi):
        """Aggregate all signal-aware predictions into one dict."""
        phase, phase_remaining = self.get_signal_phase(current_time)

        queue       = self.predict_queue_at_signal(flow_vph)
        delay       = self.estimate_webster_delay(flow_vph)
        shockwave   = self.predict_shockwave(density, avg_speed)
        travel      = self.predict_travel_to_signal(avg_speed)
        arr_phase, arr_remain, arr_tt = self.predict_arrival_phase(current_time, avg_speed)
        platoon     = self.predict_platoon_dispersion(flow_vph, avg_speed)
        spillback   = self.estimate_spillback_risk(flow_vph)
        greenwave   = self.green_wave_speed()
        stop_go     = self.predict_stop_go_waves(density, avg_speed)
        env_impact  = self.estimate_delay_impact(delay['total_delay_s'], vehicles_in_roi)

        # ── Composite congestion score (0-100) ──
        score = 0
        if delay['degree_of_saturation'] > 0.85:
            score += 25
        elif delay['degree_of_saturation'] > 0.65:
            score += 12
        if queue['overflow']:
            score += 20
        if shockwave['reaches_observer']:
            score += 20
        if spillback['spillback_risk_pct'] > 60:
            score += 15
        elif spillback['spillback_risk_pct'] > 30:
            score += 8
        if density > Config.CONGESTION_DENSITY_THRESHOLD * 0.7:
            score += 10
        if stop_go['severity'] in ('Moderate', 'High'):
            score += 10
        score = min(score, 100)

        # Congestion likelihood label
        if score < 25:
            likelihood = 'LOW'
        elif score < 50:
            likelihood = 'MODERATE'
        elif score < 75:
            likelihood = 'HIGH'
        else:
            likelihood = 'CRITICAL'

        prediction = {
            'signal_phase': phase,
            'phase_remaining_s': round(phase_remaining, 1),
            'arrival_phase': arr_phase,
            'arrival_phase_remaining_s': arr_remain,
            'arrival_travel_time_s': arr_tt,
            'queue': queue,
            'delay': delay,
            'shockwave': shockwave,
            'travel': travel,
            'platoon': platoon,
            'spillback': spillback,
            'green_wave': greenwave,
            'stop_go': stop_go,
            'env_impact': env_impact,
            'congestion_score': score,
            'congestion_likelihood': likelihood
        }

        self.prediction_history.append((current_time, prediction))
        return prediction


# ========== VISUALIZATION CLASS ==========
class TrafficVisualizer:

    # ── Left-lane core metrics panel ─────────────────────────────
    @staticmethod
    def draw_metrics_panel(frame, lane, x_off, y_start, current_time,
                           roi_height, pred_density, congestion_prob):
        metrics = lane.update_metrics(current_time, roi_height, Config.PIXELS_PER_METER)

        headway_stats = {}
        for w in Config.TIME_WINDOWS:
            headway_stats[w] = (lane.calculate_avg_headway(current_time, w),
                                lane.calculate_headway_variance(current_time, w))

        last_hw = lane.headways[-1][1] if lane.headways else 0
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
        txt(f"Total Count: {lane.vehicle_count}", (0, 255, 255), 0.7)
        txt(f"Current Vehicles: {metrics['vehicles_in_roi']}", (150, 255, 150), 0.65)
        txt(f"Queue Length: {metrics['queue_length']} veh", (255, 150, 150), 0.65)
        y += 3

        txt("--- Headway Analysis ---", (200, 200, 255), 0.65)
        txt(f"Current: {last_hw:.2f}s", (255, 200, 100), 0.6)
        for w in [20, 40, 60]:
            a, v = headway_stats[w]
            txt(f"Avg({w}s): {a:.2f}s  (var:{v:.2f})", (150, 255, 200), 0.55)
        y += 3

        txt("--- Speed Metrics ---", (200, 200, 255), 0.65)
        txt(f"Time Mean: {metrics['avg_speed']:.1f} km/h", (255, 255, 150), 0.6)
        txt(f"Space Mean: {metrics['space_mean_speed']:.1f} km/h", (255, 255, 100), 0.6)
        y += 3

        txt("--- Flow Fundamentals ---", (200, 200, 255), 0.65)
        txt(f"Density: {metrics['density']:.1f} veh/km", (150, 255, 200), 0.6)
        txt(f"Flow: {metrics['flow']:.0f} veh/h", (200, 150, 255), 0.6)
        txt(f"Level of Service: {los}", los_color, 0.7, True)
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

    # ── Signal prediction panel ──────────────────────────────────
    @staticmethod
    def draw_signal_prediction_panel(frame, prediction, x_off, y_start):
        """Draw the traffic-signal prediction panel on the right side."""
        p = prediction
        panel_w = 440
        panel_h = 720
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_off, y_start),
                      (x_off + panel_w, y_start + panel_h), (25, 25, 40), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x_off, y_start),
                      (x_off + panel_w, y_start + panel_h), (80, 80, 120), 2)

        y = y_start + 30
        lh = 26

        def txt(text, color=(255, 255, 255), scale=0.55, bold=False):
            nonlocal y
            cv2.putText(frame, text, (x_off + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1)
            y += lh

        def section(title):
            nonlocal y
            y += 2
            txt(title, (180, 180, 255), 0.6, True)

        # ── Header
        txt("=== SIGNAL PREDICTION ===", (0, 220, 255), 0.75, True)
        txt(f"Signal {p['travel']['distance_m']}m ahead", (200, 200, 200), 0.6)
        y += 4

        # ── Signal Phase
        phase_colors = {'GREEN': (0, 255, 0), 'AMBER': (0, 200, 255), 'RED': (0, 0, 255)}
        pc = phase_colors.get(p['signal_phase'], (200, 200, 200))
        txt(f"Current Phase: {p['signal_phase']}  ({p['phase_remaining_s']:.0f}s left)", pc, 0.6, True)
        arr_pc = phase_colors.get(p['arrival_phase'], (200, 200, 200))
        txt(f"On Arrival: {p['arrival_phase']}  (in {p['arrival_travel_time_s']}s)", arr_pc, 0.55)

        # ── Congestion Score
        section("--- Congestion Forecast ---")
        sc = p['congestion_score']
        if sc < 25:
            sc_color = (0, 255, 0)
        elif sc < 50:
            sc_color = (0, 255, 255)
        elif sc < 75:
            sc_color = (0, 140, 255)
        else:
            sc_color = (0, 0, 255)
        txt(f"Score: {sc}/100  [{p['congestion_likelihood']}]", sc_color, 0.65, True)

        # ── Queue Prediction
        section("--- Queue at Signal ---")
        q = p['queue']
        txt(f"Queue at red end: {q['queue_at_red_end_veh']} veh ({q['queue_length_m']}m)",
            (255, 200, 150), 0.55)
        txt(f"Served / cycle: {q['vehicles_served_per_cycle']} veh", (150, 255, 200), 0.55)
        txt(f"Residual queue: {q['residual_queue_veh']} veh",
            (0, 0, 255) if q['overflow'] else (0, 255, 0), 0.55)

        # ── Webster Delay
        section("--- Delay (Webster) ---")
        d = p['delay']
        txt(f"Avg delay: {d['total_delay_s']}s  (d1={d['uniform_delay_s']}s + d2={d['random_delay_s']}s)",
            (255, 220, 150), 0.5)
        txt(f"V/C ratio (x): {d['v_c_ratio']}",
            (0, 0, 255) if d['v_c_ratio'] > 0.85 else (0, 255, 0), 0.55)

        # ── Shockwave
        section("--- Shockwave (LWR) ---")
        sw = p['shockwave']
        txt(f"Backward wave: {sw['shockwave_speed_kmh']} km/h", (255, 180, 180), 0.55)
        txt(f"Reach in red: {sw['shockwave_reach_m']}m", (255, 180, 180), 0.55)
        if sw['reaches_observer']:
            txt(f"!! REACHES YOU in {sw['time_to_reach_s']}s !!", (0, 0, 255), 0.6, True)
        else:
            txt("Does not reach observer", (0, 200, 0), 0.5)

        # ── Platoon Dispersion
        section("--- Platoon Dispersion ---")
        pl = p['platoon']
        txt(f"Dispersion: {pl['dispersion_pct']}%  (F={pl['smoothing_factor_F']})",
            (200, 200, 255), 0.55)
        txt(f"Peak flow @ signal: {pl['peak_flow_at_signal_vph']:.0f} vph",
            (200, 200, 255), 0.55)

        # ── Spillback Risk
        section("--- Spillback Risk ---")
        sp = p['spillback']
        risk_color = (0, 255, 0) if sp['spillback_risk_pct'] < 30 else \
                     (0, 200, 255) if sp['spillback_risk_pct'] < 60 else (0, 0, 255)
        txt(f"Risk: {sp['spillback_risk_pct']}%  (storage {sp['max_storage_veh']:.0f} veh)",
            risk_color, 0.55)
        if sp['cycles_to_spillback'] != float('inf'):
            txt(f"Spillback in ~{sp['cycles_to_spillback']:.0f} cycles", (0, 0, 255), 0.55, True)

        # ── Stop-and-Go Waves
        section("--- Stop-Go Waves ---")
        sg = p['stop_go']
        sev_colors = {'None': (0, 200, 0), 'Low': (0, 255, 255),
                      'Moderate': (0, 165, 255), 'High': (0, 0, 255)}
        txt(f"Frequency: {sg['wave_frequency_per_min']}/min  [{sg['severity']}]",
            sev_colors.get(sg['severity'], (200, 200, 200)), 0.55)

        # ── Green Wave
        section("--- Green Wave Advice ---")
        gw = p['green_wave']
        txt(f"Recommended speed: {gw['recommended_speed_kmh']} km/h", (100, 255, 100), 0.55)

        # ── Environmental Impact
        section("--- Delay Impact (est.) ---")
        ei = p['env_impact']
        txt(f"Extra fuel/veh: {ei['extra_fuel_per_veh_L']:.4f} L", (200, 200, 200), 0.5)
        txt(f"CO2 estimate: {ei['co2_emission_kg']:.4f} kg", (200, 200, 200), 0.5)

    # ── Density graph ────────────────────────────────────────────
    @staticmethod
    def draw_density_graph(frame, lane, x_off, y_start, current_time, width=500):
        graph_h = 90
        if len(lane.density_history) < 2:
            return

        recent = [(t, d) for (t, d) in lane.density_history if current_time - t <= 60]
        if len(recent) < 2:
            return

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_off, y_start),
                      (x_off + width, y_start + graph_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x_off, y_start),
                      (x_off + width, y_start + graph_h), (100, 100, 100), 1)

        cv2.putText(frame, "Density (60s)", (x_off + 10, y_start + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        densities = [d for _, d in recent]
        mx = max(max(densities), 1)

        pts = []
        for i, (t, d) in enumerate(recent):
            x = x_off + 10 + int((i / len(recent)) * (width - 20))
            y = y_start + graph_h - 10 - int((d / mx) * (graph_h - 30))
            pts.append((x, y))

        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 255, 255), 2)

    # ── Congestion score bar ─────────────────────────────────────
    @staticmethod
    def draw_congestion_bar(frame, score, x_off, y_start, width=500):
        bar_h = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_off, y_start),
                      (x_off + width, y_start + bar_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Filled portion
        fill_w = int(width * score / 100)
        if score < 25:
            bar_color = (0, 200, 0)
        elif score < 50:
            bar_color = (0, 220, 220)
        elif score < 75:
            bar_color = (0, 140, 255)
        else:
            bar_color = (0, 0, 255)
        cv2.rectangle(frame, (x_off, y_start),
                      (x_off + fill_w, y_start + bar_h), bar_color, -1)
        cv2.rectangle(frame, (x_off, y_start),
                      (x_off + width, y_start + bar_h), (100, 100, 100), 1)

        cv2.putText(frame, f"Congestion Risk: {score}%",
                    (x_off + 10, y_start + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── Signal phase indicator ───────────────────────────────────
    @staticmethod
    def draw_signal_indicator(frame, phase, remaining, x_center, y_center):
        colors = {'GREEN': (0, 200, 0), 'AMBER': (0, 200, 255), 'RED': (0, 0, 220)}
        c = colors.get(phase, (128, 128, 128))
        cv2.circle(frame, (x_center, y_center), 22, c, -1)
        cv2.circle(frame, (x_center, y_center), 22, (255, 255, 255), 2)
        cv2.putText(frame, f"{remaining:.0f}s", (x_center - 14, y_center + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)


# ========== MAIN PROCESSING ==========
def process_lane(frame_roi, lane_tracker, current_time, baseline_y):
    """Process left lane with advanced tracking."""
    roi_h, roi_w = frame_roi.shape[:2]

    # Draw baseline
    cv2.line(frame_roi, (0, baseline_y), (roi_w, baseline_y), (0, 0, 255), 3)
    cv2.putText(frame_roi, "DETECTION LINE", (roi_w // 2 - 70, baseline_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Run YOLO detection
    results = model(frame_roi, conf=Config.CONFIDENCE_THRESHOLD, imgsz=640, verbose=False)[0]

    current_centers = {}
    lane_tracker.vehicles_in_roi = 0
    current_queue = set()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in Config.VEHICLE_IDS:
            continue

        lane_tracker.vehicles_in_roi += 1
        lane_tracker.vehicle_classes[cls_id] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # ID assignment
        found_id = None
        min_distance = float('inf')
        for vid, (px, py) in lane_tracker.previous_centers.items():
            distance = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if distance < Config.MIN_TRACKING_DISTANCE and distance < min_distance:
                min_distance = distance
                found_id = vid

        if found_id is None:
            found_id = lane_tracker.next_id
            lane_tracker.next_id += 1
            lane_tracker.vehicle_entry_times[found_id] = current_time

        current_centers[found_id] = (cx, cy)

        # Speed
        speed = lane_tracker.calculate_speed(found_id, cy, current_time,
                                              Config.PIXELS_PER_METER)
        if speed is not None:
            lane_tracker.speed_measurements.append(speed)
            if speed < Config.QUEUE_SPEED_THRESHOLD:
                current_queue.add(found_id)

        # Crossing detection
        if found_id in lane_tracker.previous_centers:
            prev_y = lane_tracker.previous_centers[found_id][1]
            buffer = 5
            if prev_y < (baseline_y - buffer) and cy >= baseline_y:
                lane_tracker.vehicle_count += 1
                if found_id in lane_tracker.vehicle_entry_times:
                    travel_time = current_time - lane_tracker.vehicle_entry_times[found_id]
                    lane_tracker.total_travel_time.append(travel_time)
                if lane_tracker.last_cross_time is not None:
                    headway = current_time - lane_tracker.last_cross_time
                    lane_tracker.headways.append((current_time, headway))
                lane_tracker.last_cross_time = current_time

        # Draw bounding box
        color = (0, 0, 255) if found_id in current_queue else (0, 255, 0)
        cv2.rectangle(frame_roi, (x1, y1), (x2, y2), color, 2)

        vehicle_names = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}
        label = f"{vehicle_names.get(cls_id, 'Veh')} {found_id}"
        if speed is not None:
            label += f" {speed:.0f}km/h"

        (lw, lh_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_roi, (x1, y1 - lh_ - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame_roi, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(frame_roi, (cx, cy), 4, (255, 0, 0), -1)

    lane_tracker.previous_centers = current_centers
    lane_tracker.queue_vehicles = current_queue
    return frame_roi


# ========== DATA EXPORT ==========
class DataExporter:
    @staticmethod
    def export_results(lane, signal_predictor, video_duration):
        """Export comprehensive single-lane analysis with signal predictions."""
        import os

        if not Config.EXPORT_DATA:
            return

        os.makedirs(Config.EXPORT_PATH, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Summary JSON ──
        summary = {
            'timestamp': ts,
            'video_duration': video_duration,
            'signal_config': {
                'distance_m': Config.SIGNAL_DISTANCE_M,
                'cycle_time_s': Config.SIGNAL_CYCLE_TIME,
                'green_s': Config.SIGNAL_GREEN_TIME,
                'red_s': Config.SIGNAL_RED_TIME,
                'amber_s': Config.SIGNAL_AMBER_TIME,
                'saturation_flow_vph': Config.SATURATION_FLOW,
            },
            'left_lane': {
                'total_vehicles': lane.vehicle_count,
                'vehicle_classes': dict(lane.vehicle_classes),
                'avg_headways': {
                    w: lane.calculate_avg_headway(video_duration, w)
                    for w in Config.TIME_WINDOWS
                },
                'avg_speed': float(np.mean(list(lane.speed_measurements))) if lane.speed_measurements else 0,
                'avg_travel_time': float(np.mean(lane.total_travel_time)) if lane.total_travel_time else 0,
            }
        }

        with open(f"{Config.EXPORT_PATH}summary_{ts}.json", 'w') as f:
            json.dump(summary, f, indent=4)

        # ── Time-series CSV ──
        if lane.headways:
            pd.DataFrame(list(lane.headways), columns=['time', 'headway']) \
              .to_csv(f"{Config.EXPORT_PATH}headways_left_{ts}.csv", index=False)
        if lane.density_history:
            pd.DataFrame(list(lane.density_history), columns=['time', 'density']) \
              .to_csv(f"{Config.EXPORT_PATH}density_left_{ts}.csv", index=False)
        if lane.flow_history:
            pd.DataFrame(list(lane.flow_history), columns=['time', 'flow']) \
              .to_csv(f"{Config.EXPORT_PATH}flow_left_{ts}.csv", index=False)

        # ── Signal prediction history CSV ──
        if signal_predictor.prediction_history:
            rows = []
            for t, pred in signal_predictor.prediction_history:
                rows.append({
                    'time': t,
                    'signal_phase': pred['signal_phase'],
                    'congestion_score': pred['congestion_score'],
                    'congestion_likelihood': pred['congestion_likelihood'],
                    'queue_veh': pred['queue']['queue_at_red_end_veh'],
                    'queue_length_m': pred['queue']['queue_length_m'],
                    'webster_delay_s': pred['delay']['total_delay_s'],
                    'v_c_ratio': pred['delay']['v_c_ratio'],
                    'shockwave_speed_kmh': pred['shockwave']['shockwave_speed_kmh'],
                    'shockwave_reach_m': pred['shockwave']['shockwave_reach_m'],
                    'spillback_risk_pct': pred['spillback']['spillback_risk_pct'],
                    'stop_go_severity': pred['stop_go']['severity'],
                    'platoon_dispersion_pct': pred['platoon']['dispersion_pct'],
                })
            pd.DataFrame(rows).to_csv(
                f"{Config.EXPORT_PATH}signal_predictions_{ts}.csv", index=False)

        print(f"\n[+] Data exported to {Config.EXPORT_PATH}")


# ========== MAIN EXECUTION ==========
def main():
    global model

    print("Loading YOLO model...")
    model = YOLO(Config.YOLO_MODEL)

    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video {Config.VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Single lane tracker (left lane)
    left_lane = AdvancedLaneTracker("LEFT LANE", 0)
    signal_pred = TrafficSignalPredictor()

    start_time_global = time.time()
    frame_count = 0
    processed_frames = 0

    print("\n" + "=" * 70)
    print("  ADVANCED TRAFFIC FLOW ANALYSIS  --  LEFT LANE + SIGNAL PREDICTION")
    print("=" * 70)
    print(f"  Video          : {Config.VIDEO_PATH}")
    print(f"  Total Frames   : {total_frames}")
    print(f"  FPS            : {fps}")
    print(f"  Speed Mult     : {Config.SPEED_MULTIPLIER}x")
    print(f"  Signal Distance: {Config.SIGNAL_DISTANCE_M}m ahead")
    print(f"  Cycle (G/A/R)  : {Config.SIGNAL_GREEN_TIME}/"
          f"{Config.SIGNAL_AMBER_TIME}/{Config.SIGNAL_RED_TIME}s")
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
        current_time = time.time() - start_time_global

        # ── Use left half of the frame as left lane ROI ──
        mid_x = w // 2
        left_roi = frame[:, :mid_x].copy()
        roi_h = left_roi.shape[0]
        baseline_y = int(roi_h * Config.BASELINE_OFFSET)

        # Process left lane
        left_roi = process_lane(left_roi, left_lane, current_time, baseline_y)
        frame[:, :mid_x] = left_roi

        # Dim right half to indicate it's not analysed
        right_half = frame[:, mid_x:]
        dimmed = (right_half * 0.3).astype(np.uint8)
        frame[:, mid_x:] = dimmed
        cv2.putText(frame, "RIGHT LANE - NOT ANALYSED",
                    (mid_x + 30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        # Lane divider
        cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 0), 3)

        # ── Compute metrics & predictions ──
        metrics = left_lane.update_metrics(current_time, roi_h, Config.PIXELS_PER_METER)
        pred_density, congestion_prob = left_lane.predict_congestion(current_time)

        sig_prediction = signal_pred.comprehensive_prediction(
            current_time,
            metrics['flow'],
            metrics['density'],
            metrics['avg_speed'],
            metrics['vehicles_in_roi']
        )

        # ── Draw panels ──
        # Left panel: core lane metrics
        TrafficVisualizer.draw_metrics_panel(
            frame, left_lane, 10, 50, current_time,
            roi_h, pred_density, congestion_prob
        )

        # Right panel: signal prediction (on the right-hand side of frame)
        TrafficVisualizer.draw_signal_prediction_panel(
            frame, sig_prediction, w - 460, 50
        )

        # Signal phase indicator (top-center)
        TrafficVisualizer.draw_signal_indicator(
            frame,
            sig_prediction['signal_phase'],
            sig_prediction['phase_remaining_s'],
            w // 2, 30
        )

        # Congestion bar (below panels)
        TrafficVisualizer.draw_congestion_bar(
            frame, sig_prediction['congestion_score'], 10, h - 180, w - 20
        )

        # Density graph
        TrafficVisualizer.draw_density_graph(
            frame, left_lane, 10, h - 140, current_time, mid_x - 20
        )

        # Progress bar
        progress = (frame_count / total_frames) * 100
        bar_w = w - 40
        cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 40),
                      (20 + int(bar_w * progress / 100), h - 20), (0, 255, 0), -1)
        cv2.putText(frame, f"Progress: {progress:.1f}%",
                    (w // 2 - 80, h - 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Left Lane Traffic Analysis + Signal Prediction", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            sname = f"{Config.EXPORT_PATH}screenshot_{int(current_time)}.jpg"
            cv2.imwrite(sname, frame)
            print(f"Screenshot saved: {sname}")

    # ── Cleanup ──
    final_time = time.time() - start_time_global
    cap.release()
    cv2.destroyAllWindows()

    # ══════════════════════════════════════════════════════════════
    #  FINAL STATISTICS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL COMPREHENSIVE STATISTICS  --  LEFT LANE ONLY")
    print("=" * 70)
    print(f"  Processing Time  : {final_time:.2f} s")
    print(f"  Frames Processed : {processed_frames}/{total_frames}")
    print("=" * 70)

    lane = left_lane
    print(f"\n  {lane.lane_name}")
    print("  " + "-" * 66)
    print(f"    Total Vehicles Counted : {lane.vehicle_count}")
    print(f"    Vehicle Classes        : {dict(lane.vehicle_classes)}")

    print("\n    Headway Analysis:")
    for w in Config.TIME_WINDOWS:
        a = lane.calculate_avg_headway(final_time, w)
        v = lane.calculate_headway_variance(final_time, w)
        print(f"      {w:>4}s window : Avg={a:.2f}s, Var={v:.2f}")

    if lane.speed_measurements:
        speeds = list(lane.speed_measurements)
        print(f"\n    Speed Statistics:")
        print(f"      Mean   : {np.mean(speeds):.2f} km/h")
        print(f"      Std    : {np.std(speeds):.2f} km/h")
        print(f"      Min    : {np.min(speeds):.2f} km/h")
        print(f"      Max    : {np.max(speeds):.2f} km/h")

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

    # ── Signal prediction summary ──
    print("\n  " + "=" * 66)
    print("  SIGNAL PREDICTION SUMMARY")
    print("  " + "=" * 66)
    print(f"    Signal Distance       : {Config.SIGNAL_DISTANCE_M} m")
    print(f"    Cycle (G/A/R)         : {Config.SIGNAL_GREEN_TIME}/"
          f"{Config.SIGNAL_AMBER_TIME}/{Config.SIGNAL_RED_TIME} s")

    if signal_pred.prediction_history:
        scores = [p['congestion_score'] for _, p in signal_pred.prediction_history]
        delays = [p['delay']['total_delay_s'] for _, p in signal_pred.prediction_history]
        queues = [p['queue']['queue_at_red_end_veh'] for _, p in signal_pred.prediction_history]
        vcs    = [p['delay']['v_c_ratio'] for _, p in signal_pred.prediction_history]
        spills = [p['spillback']['spillback_risk_pct'] for _, p in signal_pred.prediction_history]

        print(f"\n    Congestion Score:")
        print(f"      Average : {np.mean(scores):.1f} / 100")
        print(f"      Max     : {np.max(scores):.0f} / 100")

        print(f"\n    Webster Delay:")
        print(f"      Average : {np.mean(delays):.1f} s")
        print(f"      Max     : {np.max(delays):.1f} s")

        print(f"\n    Queue at Red End:")
        print(f"      Average : {np.mean(queues):.1f} veh")
        print(f"      Max     : {np.max(queues):.1f} veh")

        print(f"\n    V/C Ratio:")
        print(f"      Average : {np.mean(vcs):.3f}")
        print(f"      Max     : {np.max(vcs):.3f}")

        print(f"\n    Spillback Risk:")
        print(f"      Average : {np.mean(spills):.1f}%")
        print(f"      Max     : {np.max(spills):.1f}%")

        # How often was congestion HIGH or CRITICAL?
        high_count = sum(1 for s in scores if s >= 50)
        print(f"\n    Frames with HIGH/CRITICAL congestion: "
              f"{high_count}/{len(scores)} ({high_count/len(scores)*100:.1f}%)")

    print("\n" + "=" * 70)

    # Export
    DataExporter.export_results(left_lane, signal_pred, final_time)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
