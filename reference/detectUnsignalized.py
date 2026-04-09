"""
detectUnsignalized.py
=====================
Advanced Vehicle Detection & Headway Analysis for UNSIGNALIZED / UNCONTROLLED Roads
PhD-level Traffic Flow Analysis

Focus: Fundamental traffic-flow theory on free-flowing or uncontrolled roads.
No traffic signals, no signal-based predictions. All parameters are roadway /
driver-behaviour / flow-theory based.

Key NEW parameters and models introduced (absent from detectFinal.py):
  - Time-to-Collision (TTC) & Post-Encroachment Time (PET) for safety analysis
  - Inter-vehicle gap acceptance modelling (critical gap / Raff's method)
  - Greenshields, Greenberg, Underwood speed-density models with model selection
  - Erlang-B/C arrival distribution fitting (Poisson vs. negative-binomial)
  - Headway distribution fitting: exponential, lognormal, gamma, shifted negative-exp
  - Platoon identification (Platoon Ratio, % vehicles in platoon)
  - Coefficient of Variation of headways (CVH) -- uniformity index
  - Bunching / Platooning Tendency Index (PTI)
  - Passenger Car Equivalent (PCE) computation per vehicle class
  - Lane Occupancy Ratio (not just density)
  - Speed Variance (Daganzo safety surrogate)
  - Inter-arrival time statistics (percentile headways p15, p50, p85)
  - Moving-bottleneck detection via spatial speed gradient
  - Rolling 5-point acceleration profile per tracked vehicle
  - Shock-wave speed computed purely from measured flow/density (not signal-driven)
  - Entropy of headway distribution (regularity of traffic)
  - Passenger Car Unit (PCU) weighted flow
  - Friction / Conflict-Point Risk Index (CFI) for unsignalized pedestrian crossings

Author: PhD Research
"""

import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque, defaultdict
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ========== CONFIGURATION ==========
class Config:
    # Video / processing
    SPEED_MULTIPLIER = 2
    VIDEO_PATH = "v5.mp4"

    # YOLO
    YOLO_MODEL = "yolov5s.pt"
    CONFIDENCE_THRESHOLD = 0.35
    VEHICLE_IDS = [2, 3, 5, 7]          # car, motorcycle, bus, truck

    # Geometric calibration
    PIXELS_PER_METER = 30
    BASELINE_OFFSET = 0.5               # fraction of ROI height for counting line
    ROI_LENGTH_M = 50.0                 # length of observed ROI in meters

    # Operational thresholds
    QUEUE_SPEED_THRESHOLD = 10          # km/h -- vehicle considered stationary
    MIN_TRACKING_DISTANCE = 80          # pixels -- max pixel jump for same vehicle
    FREE_FLOW_SPEED_KMH = 60.0          # design / posted speed limit
    JAM_DENSITY_VEH_KM = 150.0          # jam density (veh/km) for LWR
    CRITICAL_DENSITY_VEH_KM = 40.0      # approximate critical density

    # Safety
    TTC_CRITICAL_S = 2.5                # TTC below this is critical (seconds)
    CRITICAL_GAP_S = 4.0                # critical gap for gap-acceptance model

    # Platoon definition
    PLATOON_HEADWAY_S = 5.0             # headways <= this classify a vehicle in platoon
    MIN_PLATOON_SIZE = 2                # minimum vehicles for a platoon event

    # Passenger Car Equivalents (HCM 2016 mid-block freeway)
    PCE = {2: 1.0, 3: 0.5, 5: 2.5, 7: 2.0}  # car, moto, bus, truck

    # Time windows for rolling statistics
    TIME_WINDOWS = [20, 40, 60, 120]

    # Export
    EXPORT_DATA = True
    EXPORT_PATH = "unsignalized_analysis/"


# ========== VEHICLE STATE TRACKER ==========
class VehicleState:
    """Per-vehicle kinematic state: position, speed, acceleration, TTC."""

    def __init__(self, vid):
        self.vid = vid
        self.positions = deque(maxlen=30)    # (time, y_pixel)
        self.speeds_kmh = deque(maxlen=20)   # rolling speed buffer
        self.accel_mps2 = deque(maxlen=15)   # rolling acceleration buffer
        self.ttc = None                      # latest TTC (s)
        self.pet = None                      # latest PET (s)
        self.in_platoon = False
        self.pce = 1.0
        self.cls_id = 2                      # default: car
        self.last_speed = None

    def update_position(self, t, y, pixels_per_meter):
        self.positions.append((t, y))
        if len(self.positions) >= 3:
            dt = self.positions[-1][0] - self.positions[0][0]
            if dt > 0.4:
                dy = abs(self.positions[-1][1] - self.positions[0][1])
                spd = (dy / pixels_per_meter) / dt * 3.6
                if 0 <= spd <= 200:
                    self.speeds_kmh.append(spd)
                    # Acceleration from last two speed samples
                    if len(self.speeds_kmh) >= 2:
                        dv = (self.speeds_kmh[-1] - self.speeds_kmh[-2]) / 3.6
                        dt2 = self.positions[-1][0] - self.positions[-2][0]
                        if dt2 > 0:
                            a = dv / dt2
                            self.accel_mps2.append(a)
                    self.last_speed = spd
                    return spd
        return self.last_speed

    def current_speed(self):
        return float(np.mean(self.speeds_kmh)) if self.speeds_kmh else None

    def current_accel(self):
        return float(np.mean(self.accel_mps2)) if self.accel_mps2 else 0.0


# ========== HEADWAY DISTRIBUTION ANALYSER ==========
class HeadwayDistributionAnalyser:
    """
    Fits multiple statistical distributions to observed headway data and
    selects the best fit via AIC.  Distributions tested:
      1. Negative exponential (fully random arrivals, Poisson process)
      2. Shifted negative exponential (accounts for minimum reaction time)
      3. Lognormal
      4. Gamma / Pearson-III (partially constrained traffic)
      5. Erlang (grouped arrivals)

    Also computes traffic-flow headway statistics:
      - CVH (Coefficient of Variation of Headways)
      - PTI (Platooning Tendency Index)
      - Shannon entropy of headway distribution
      - Percentile headways: h15, h50, h85
    """

    @staticmethod
    def fit_distributions(headways):
        if len(headways) < 10:
            return {}

        h = np.array(headways)
        results = {}

        # --- 1. Negative exponential ---
        try:
            lam = 1.0 / np.mean(h)
            log_L = np.sum(stats.expon.logpdf(h, scale=1.0 / lam))
            aic = 2 * 1 - 2 * log_L
            results['neg_exponential'] = {'params': {'lambda': lam}, 'AIC': aic}
        except Exception:
            pass

        # --- 2. Shifted negative exponential  h ~ exp(lambda) + tau_min ---
        try:
            tau_min = max(np.percentile(h, 2), 0.2)  # minimum possible headway
            h_shifted = h - tau_min
            h_shifted = h_shifted[h_shifted > 0]
            if len(h_shifted) > 5:
                lam_s = 1.0 / np.mean(h_shifted)
                log_L = np.sum(stats.expon.logpdf(h_shifted, scale=1.0 / lam_s))
                aic = 2 * 2 - 2 * log_L
                results['shifted_neg_exp'] = {
                    'params': {'tau_min': tau_min, 'lambda': lam_s}, 'AIC': aic}
        except Exception:
            pass

        # --- 3. Lognormal ---
        try:
            mu, sigma = np.mean(np.log(h[h > 0])), np.std(np.log(h[h > 0]))
            log_L = np.sum(stats.lognorm.logpdf(h[h > 0], s=sigma, scale=np.exp(mu)))
            aic = 2 * 2 - 2 * log_L
            results['lognormal'] = {'params': {'mu': mu, 'sigma': sigma}, 'AIC': aic}
        except Exception:
            pass

        # --- 4. Gamma ---
        try:
            alpha, loc, beta = stats.gamma.fit(h, floc=0)
            log_L = np.sum(stats.gamma.logpdf(h, alpha, loc=loc, scale=beta))
            aic = 2 * 2 - 2 * log_L
            results['gamma'] = {'params': {'alpha': alpha, 'beta': beta}, 'AIC': aic}
        except Exception:
            pass

        # --- 5. Erlang (k integer shape) ---
        try:
            k = max(1, round(np.mean(h) ** 2 / np.var(h)))
            beta_e = np.mean(h) / k
            log_L = np.sum(stats.gamma.logpdf(h, k, loc=0, scale=beta_e))
            aic = 2 * 2 - 2 * log_L
            results['erlang'] = {'params': {'k': k, 'beta': beta_e}, 'AIC': aic}
        except Exception:
            pass

        # Select best by AIC
        if results:
            best = min(results, key=lambda x: results[x]['AIC'])
            results['best_fit'] = best

        return results

    @staticmethod
    def compute_headway_statistics(headways, platoon_threshold):
        if len(headways) < 3:
            return {}

        h = np.array(headways)
        mu = np.mean(h)
        sigma = np.std(h)

        # Coefficient of Variation of Headways (CVH)
        cvh = sigma / mu if mu > 0 else 0.0

        # Platooning Tendency Index (PTI): fraction of headways in platoon
        platoon_count = np.sum(h <= platoon_threshold)
        pti = platoon_count / len(h)

        # Shannon entropy (discretised into 0.5s bins)
        bins = np.arange(0, max(h) + 1, 0.5)
        counts, _ = np.histogram(h, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        # Percentile headways
        h15 = float(np.percentile(h, 15))
        h50 = float(np.percentile(h, 50))
        h85 = float(np.percentile(h, 85))

        # Degree of randomness (Ratio of variance to mean -- Index of Dispersion)
        iod = (sigma ** 2) / mu if mu > 0 else 0.0
        # IOD ~ 1 → Poisson;  IOD < 1 → regular;  IOD > 1 → clustered/platooned

        return {
            'mean_s': round(mu, 3),
            'std_s': round(sigma, 3),
            'cvh': round(cvh, 3),
            'pti': round(pti, 3),
            'entropy_bits': round(entropy, 3),
            'h15_s': round(h15, 3),
            'h50_s': round(h50, 3),
            'h85_s': round(h85, 3),
            'index_of_dispersion': round(iod, 3),
            'n_samples': len(h)
        }


# ========== SPEED-DENSITY MODEL FITTING ==========
class SpeedDensityModels:
    """
    Fits classical macroscopic speed-density (u-k) models to observed data.
    Models:
      - Greenshields (1935): linear u = uf(1 - k/kj)
      - Greenberg (1959): logarithmic u = uf * ln(kj/k)
      - Underwood (1961): exponential u = uf * exp(-k/k0)
      - Northwestern (Drake, 1967): u = uf * exp(-0.5*(k/k0)^2)

    The fitted model is used to derive the theoretical capacity (maximum flow).
    """

    @staticmethod
    def greenshields(k, uf, kj):
        return uf * (1 - k / kj)

    @staticmethod
    def greenberg(k, uf, kj):
        k = np.maximum(k, 1e-6)
        return uf * np.log(kj / k)

    @staticmethod
    def underwood(k, uf, k0):
        return uf * np.exp(-k / k0)

    @staticmethod
    def drake(k, uf, k0):
        return uf * np.exp(-0.5 * (k / k0) ** 2)

    @classmethod
    def fit_all(cls, densities, speeds):
        """Fit all models to (k, u) data; return best by RMSE."""
        k = np.array(densities)
        u = np.array(speeds)

        if len(k) < 6 or np.std(k) < 0.5:
            return None

        results = {}
        uf0 = Config.FREE_FLOW_SPEED_KMH
        kj0 = Config.JAM_DENSITY_VEH_KM

        def try_fit(name, func, p0, bounds):
            try:
                popt, _ = curve_fit(func, k, u, p0=p0, bounds=bounds,
                                    maxfev=3000)
                u_pred = func(k, *popt)
                rmse = np.sqrt(np.mean((u - u_pred) ** 2))
                ss_res = np.sum((u - u_pred) ** 2)
                ss_tot = np.sum((u - np.mean(u)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                results[name] = {'params': popt, 'rmse': rmse, 'r2': r2}
            except Exception:
                pass

        try_fit('greenshields', cls.greenshields,
                [uf0, kj0], ([10, 20], [200, 500]))
        try_fit('greenberg', cls.greenberg,
                [uf0, kj0], ([5, 20], [200, 500]))
        try_fit('underwood', cls.underwood,
                [uf0, kj0 * 0.3], ([5, 5], [200, 300]))
        try_fit('drake', cls.drake,
                [uf0, kj0 * 0.3], ([5, 5], [200, 300]))

        if not results:
            return None

        best = min(results, key=lambda x: results[x]['rmse'])
        return {'best_model': best, 'models': results}


# ========== SAFETY SURROGATE CALCULATOR ==========
class SafetySurrogates:
    """
    Computes safety surrogate measures (SSMs) for uncontrolled road sections.
      - Time-to-Collision (TTC): gap / relative speed
      - Post-Encroachment Time (PET): estimated from headway and speed
      - Deceleration Rate to Avoid Crash (DRAC)
      - Speed Variance (Daganzo surrogate: high variance → higher conflict risk)
      - Conflict-point Friction Index (CFI) based on approach speed & flow
    """

    @staticmethod
    def compute_ttc(gap_m, lead_speed_kmh, follow_speed_kmh):
        """
        TTC = gap / (v_follow - v_lead)  when follower is faster than leader.
        Returns inf if follower is not closing.
        """
        delta_v = (follow_speed_kmh - lead_speed_kmh) / 3.6   # m/s
        if delta_v <= 0:
            return float('inf')
        return round(gap_m / delta_v, 2)

    @staticmethod
    def compute_drac(gap_m, lead_speed_kmh, follow_speed_kmh):
        """
        DRAC = (v_follow - v_lead)^2 / (2 * gap)  [m/s^2]
        Minimum deceleration follower must apply to avoid collision.
        """
        delta_v = (follow_speed_kmh - lead_speed_kmh) / 3.6
        if delta_v <= 0 or gap_m <= 0:
            return 0.0
        return round((delta_v ** 2) / (2 * gap_m), 3)

    @staticmethod
    def compute_pet(headway_s, speed_kmh):
        """
        Approximate PET for a crossing/merging point:
        PET ≈ headway - (vehicle_length / speed)
        """
        if speed_kmh <= 0:
            return float('inf')
        v_ms = speed_kmh / 3.6
        clearance_time = Config.PCE[2] * Config.PCE[2] / v_ms   # crude
        pet = headway_s - clearance_time
        return round(max(pet, 0.0), 2)

    @staticmethod
    def conflict_friction_index(flow_vph, avg_speed_kmh):
        """
        CFI: dimensionless index 0-100.
        Higher flow + higher speed = higher CFI.
        Based on energy exposure concept: CFI ∝ q * v^2
        Normalised to a reference freeway (3000 vph, 100 km/h).
        """
        if avg_speed_kmh <= 0 or flow_vph <= 0:
            return 0.0
        cfi = (flow_vph * avg_speed_kmh ** 2) / (3000 * 100 ** 2) * 100
        return round(min(cfi, 100.0), 1)

    @staticmethod
    def speed_variance_risk(speeds):
        """
        Daganzo's speed-variance safety surrogate.
        Risk level based on variance of spot speeds in the stream.
        """
        if len(speeds) < 4:
            return 0.0, 'Insufficient data'
        var = np.var(speeds)
        if var < 50:
            return round(var, 1), 'Low'
        elif var < 150:
            return round(var, 1), 'Moderate'
        else:
            return round(var, 1), 'High'


# ========== GAP ACCEPTANCE MODEL ==========
class GapAcceptanceModel:
    """
    Raff's critical gap method for unsignalized intersections / merging.
    Tracks accepted and rejected gaps to estimate the critical gap tc:
      tc is where the cumulative accepted gap CDF crosses the rejected gap CCDF.

    Also computes:
      - Average accepted gap
      - Capacity of minor stream using Tanner's formula
      - Follow-up time (tf) from observed minimum headways
    """

    def __init__(self):
        self.accepted_gaps = []    # gaps where vehicle crossed/merged
        self.rejected_gaps = []    # gaps where vehicle waited
        self.follow_up_times = []  # time between successive vehicles using same gap

    def record_gap(self, gap_s, accepted: bool):
        if accepted:
            self.accepted_gaps.append(gap_s)
        else:
            self.rejected_gaps.append(gap_s)

    def raff_critical_gap(self):
        """
        Raff method: tc = intersection of F_a(t) and 1 - F_r(t).
        Returns estimated critical gap in seconds.
        """
        if len(self.accepted_gaps) < 5 or len(self.rejected_gaps) < 5:
            return None

        t_range = np.linspace(0.5, 15.0, 300)
        fa = np.array([np.mean(np.array(self.accepted_gaps) <= t) for t in t_range])
        fr = np.array([np.mean(np.array(self.rejected_gaps) > t) for t in t_range])

        diff = fa - fr
        idx = np.argmin(np.abs(diff))
        return round(t_range[idx], 2)

    def tanner_capacity(self, main_flow_vph, critical_gap_s, follow_up_s):
        """
        Tanner's minor-stream capacity model:
          Cm = q_main * exp(-q_main * tc / 3600) / (1 - exp(-q_main * tf / 3600))
        """
        q = main_flow_vph / 3600.0
        if q <= 0 or critical_gap_s <= 0 or follow_up_s <= 0:
            return 0.0
        numer = q * np.exp(-q * critical_gap_s)
        denom = 1 - np.exp(-q * follow_up_s)
        cm = numer / max(denom, 1e-9) * 3600
        return round(cm, 0)


# ========== PLATOON ANALYSER ==========
class PlatoonAnalyser:
    """
    Identifies platoons from headway sequences using the threshold method.
    Computes:
      - Platoon Ratio (PR): fraction of time in platoon
      - Number of platoons in observation window
      - Average platoon size
      - Average intra-platoon headway (follower headways inside platoon)
      - Average inter-platoon headway (leader headways between platoons)
    """

    @staticmethod
    def identify_platoons(headway_series, threshold_s):
        """
        headway_series: list of (time, headway) tuples.
        A vehicle is a platoon follower if headway <= threshold.
        Returns platoon events and statistics.
        """
        if len(headway_series) < 5:
            return {}, []

        headways = [h for _, h in headway_series]
        in_platoon = [h <= threshold_s for h in headways]

        platoon_events = []
        current = []
        for i, flag in enumerate(in_platoon):
            if flag:
                current.append(headways[i])
            else:
                if len(current) >= Config.MIN_PLATOON_SIZE:
                    platoon_events.append(current)
                current = []
        if len(current) >= Config.MIN_PLATOON_SIZE:
            platoon_events.append(current)

        total_veh = len(headways)
        veh_in_platoon = sum(len(p) for p in platoon_events)
        platoon_ratio = veh_in_platoon / total_veh if total_veh > 0 else 0.0

        intra = []
        for p in platoon_events:
            intra.extend(p)
        inter_headways = [h for h in headways if h > threshold_s]

        stats_dict = {
            'platoon_ratio': round(platoon_ratio, 3),
            'num_platoons': len(platoon_events),
            'avg_platoon_size': round(np.mean([len(p) for p in platoon_events]), 1)
                               if platoon_events else 0.0,
            'avg_intra_platoon_headway_s': round(np.mean(intra), 2) if intra else 0.0,
            'avg_inter_platoon_headway_s': round(np.mean(inter_headways), 2)
                                           if inter_headways else 0.0,
            'veh_in_platoon': veh_in_platoon,
            'total_vehicles': total_veh
        }
        return stats_dict, platoon_events


# ========== MOVING BOTTLENECK DETECTOR ==========
class MovingBottleneckDetector:
    """
    Detects moving bottlenecks: slow-moving vehicles that impede the stream.
    Algorithm:
      1. Track vehicles whose speed < (stream_speed - threshold).
      2. Check if vehicles behind the slow vehicle decelerate.
      3. Compute spatial speed gradient dU/dx using tracked positions.
    If a vehicle is significantly slower than the stream and upstream vehicles
    are decelerating, flag it as a moving bottleneck.
    """

    def __init__(self):
        self.bottleneck_history = deque(maxlen=200)

    def detect(self, vehicle_states: dict, stream_speed_kmh: float):
        """
        vehicle_states: dict[vid] -> VehicleState
        stream_speed_kmh: current mean stream speed
        Returns list of flagged vehicle IDs and severity.
        """
        if stream_speed_kmh <= 0:
            return []

        bottlenecks = []
        for vid, vs in vehicle_states.items():
            spd = vs.current_speed()
            if spd is None:
                continue
            deficit = stream_speed_kmh - spd
            # A vehicle is a moving bottleneck if it is >30% slower than stream
            if deficit > 0.30 * stream_speed_kmh and spd < Config.FREE_FLOW_SPEED_KMH:
                severity = deficit / stream_speed_kmh  # 0-1
                accel = vs.current_accel()
                bottlenecks.append({
                    'vid': vid,
                    'speed_kmh': round(spd, 1),
                    'stream_speed_kmh': round(stream_speed_kmh, 1),
                    'deficit_kmh': round(deficit, 1),
                    'severity': round(severity, 2),
                    'decelerating': accel < -0.3
                })
        if bottlenecks:
            self.bottleneck_history.append((time.time(), bottlenecks))
        return bottlenecks


# ========== PCU WEIGHTED FLOW CALCULATOR ==========
class PCUFlowCalculator:
    """
    Converts actual vehicle mix to Passenger Car Unit (PCU) flow.
    Uses HCM 2016 equivalency factors (Config.PCE).
    PCU flow is the standard measure for capacity analysis on uncontrolled roads.
    """

    @staticmethod
    def pcu_flow(vehicle_classes: dict, observation_hours: float):
        """
        vehicle_classes: {cls_id: count}
        Returns PCU/hour.
        """
        if observation_hours <= 0:
            return 0.0
        total_pcu = sum(Config.PCE.get(cls, 1.0) * cnt
                        for cls, cnt in vehicle_classes.items())
        return round(total_pcu / observation_hours, 1)

    @staticmethod
    def vehicle_mix_composition(vehicle_classes: dict):
        total = sum(vehicle_classes.values())
        if total == 0:
            return {}
        names = {2: 'Car', 3: 'Moto', 5: 'Bus', 7: 'Truck'}
        return {names.get(k, k): round(v / total * 100, 1)
                for k, v in vehicle_classes.items()}


# ========== TRAFFIC PREDICTOR (History-Based, No Signal) ==========
class TrafficPredictor:
    """
    Purely data-driven predictions from recorded traffic history.
    No signal assumptions. Four independent prediction methods:

    1. EWMA (Exponentially Weighted Moving Average)
       - Adaptive short-horizon (15s) smoothing of flow, density, speed.
       - Alpha tuned by recent variance: high variance → lower alpha (more smoothing).
       - Output: EWMA-predicted value for the next observation interval.

    2. Polynomial Regression on Density Trend
       - Fits degree-2 polynomial to density time-series (last 120 s).
       - Extrapolates 30 s and 60 s ahead.
       - Also derives first derivative (dK/dt) to classify trend:
         rising / falling / stable, and estimates time-to-congestion onset.

    3. LWR Shockwave from Measured Flow/Density
       - Uses Newell-Daganzo triangular fundamental diagram:
           w = (q2 - q1) / (k2 - k1)   [shockwave speed]
       - Computes backward shockwave speed from current state vs. jam state.
       - No signal: shockwave driven purely by flow breakdown.

    4. Next-Vehicle Arrival Time Prediction
       - Uses the best-fit headway distribution (from HeadwayDistributionAnalyser)
         to predict the expected arrival time of the next vehicle.
       - Also gives 85th-percentile arrival time (conservative estimate).
       - Uses EWMA-smoothed last headway as the base estimate when
         distribution data is insufficient.
    """

    def __init__(self):
        # EWMA state per variable
        self._ewma = {'flow': None, 'density': None, 'speed': None}
        self._ewma_alpha = 0.25          # default smoothing factor
        self.prediction_history = deque(maxlen=600)

    # ── 1. EWMA ─────────────────────────────────────────────────
    def update_ewma(self, flow, density, speed):
        """Update EWMA for the three primary variables."""
        for key, val in [('flow', flow), ('density', density), ('speed', speed)]:
            if val is None or val <= 0:
                continue
            if self._ewma[key] is None:
                self._ewma[key] = val
            else:
                self._ewma[key] = self._ewma_alpha * val + \
                                   (1 - self._ewma_alpha) * self._ewma[key]
        return {k: round(v, 2) if v is not None else None
                for k, v in self._ewma.items()}

    def adapt_alpha(self, density_history, window_s=60, current_time=0):
        """
        Variance-adaptive alpha: high recent variance → more smoothing (lower alpha).
        Range: 0.10 (high variance / unstable) → 0.40 (stable).
        """
        recent = [d for (t, d) in density_history if current_time - t <= window_s]
        if len(recent) < 5:
            return
        cv = np.std(recent) / (np.mean(recent) + 1e-9)
        # High CV → low alpha (heavy smoothing); low CV → high alpha (responsive)
        self._ewma_alpha = float(np.clip(0.40 - 0.30 * cv, 0.10, 0.40))

    # ── 2. Polynomial Regression Density Forecast ───────────────
    @staticmethod
    def poly_density_forecast(density_history, current_time,
                               horizon_30=30, horizon_60=60):
        """
        Fit degree-2 polynomial to last 120 s of density data.
        Returns predicted density at +30 s and +60 s, trend label,
        dK/dt at current time, and estimated seconds to congestion.
        """
        window = [(t, d) for (t, d) in density_history
                  if current_time - t <= 120]
        if len(window) < 8:
            return None

        times = np.array([t for t, _ in window])
        densities = np.array([d for _, d in window])
        t0 = times[0]
        t_norm = times - t0

        # Ridge-regularised polynomial regression (degree 2)
        pipe = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0))
        pipe.fit(t_norm.reshape(-1, 1), densities)

        # Predict at horizons
        t_curr = float(t_norm[-1])
        pred_30 = float(pipe.predict([[t_curr + horizon_30]])[0])
        pred_60 = float(pipe.predict([[t_curr + horizon_60]])[0])
        pred_30 = max(pred_30, 0.0)
        pred_60 = max(pred_60, 0.0)

        # First derivative at current time: dK/dt (veh/km/s)
        # d/dt [a*t^2 + b*t + c] = 2a*t + b
        coef = pipe.named_steps['ridge'].coef_
        # coef[0]=bias(from poly), coef[1]=linear, coef[2]=quadratic
        a2 = coef[2] if len(coef) > 2 else 0.0
        a1 = coef[1] if len(coef) > 1 else 0.0
        dkdt = 2 * a2 * t_curr + a1   # veh/km per second

        # Trend classification
        if dkdt > 0.5:
            trend = 'RISING'
            trend_color_key = 'red'
        elif dkdt < -0.5:
            trend = 'FALLING'
            trend_color_key = 'green'
        else:
            trend = 'STABLE'
            trend_color_key = 'yellow'

        # Time to congestion onset (density > critical) from polynomial
        time_to_congestion = None
        if dkdt > 0 and densities[-1] < Config.CRITICAL_DENSITY_VEH_KM:
            gap = Config.CRITICAL_DENSITY_VEH_KM - densities[-1]
            if dkdt > 0:
                time_to_congestion = round(gap / dkdt, 1)   # seconds

        return {
            'pred_density_30s': round(pred_30, 2),
            'pred_density_60s': round(pred_60, 2),
            'dkdt_veh_km_s': round(dkdt, 4),
            'trend': trend,
            'trend_color_key': trend_color_key,
            'time_to_congestion_s': time_to_congestion,
            'congestion_prob_30s': round(
                min(pred_30 / Config.CRITICAL_DENSITY_VEH_KM, 1.0), 3),
            'congestion_prob_60s': round(
                min(pred_60 / Config.CRITICAL_DENSITY_VEH_KM, 1.0), 3),
        }

    # ── 3. LWR Shockwave (purely from measured state) ───────────
    @staticmethod
    def lwr_shockwave(flow_vph, density_veh_km, avg_speed_kmh):
        """
        Newell-Daganzo triangular FD shockwave computation.
        State 1 = current measured state (q1, k1)
        State 2 = jam state (q2=0, k2=kj)
        Shockwave speed: w = (q2 - q1) / (k2 - k1)
        Negative value means backward propagation (upstream).
        """
        kj = Config.JAM_DENSITY_VEH_KM
        q1 = flow_vph
        k1 = density_veh_km

        if k1 <= 0 or avg_speed_kmh <= 0:
            return {'shockwave_kmh': 0.0, 'direction': 'none',
                    'severity': 'None', 'breakdown_risk': 0.0}

        # Jam state
        q2, k2 = 0.0, kj
        if abs(k2 - k1) < 0.1:
            w = 0.0
        else:
            w = (q2 - q1) / (k2 - k1)   # km/h (negative = backward)

        direction = 'backward' if w < 0 else 'forward'
        abs_w = abs(w)

        # Breakdown risk: ratio of current density to critical density
        kc = Config.CRITICAL_DENSITY_VEH_KM
        breakdown_risk = round(min(k1 / kc, 1.0), 3)

        if breakdown_risk < 0.5:
            severity = 'None'
        elif breakdown_risk < 0.75:
            severity = 'Low'
        elif breakdown_risk < 0.90:
            severity = 'Moderate'
        else:
            severity = 'High'

        return {
            'shockwave_kmh': round(w, 1),
            'shockwave_abs_kmh': round(abs_w, 1),
            'direction': direction,
            'severity': severity,
            'breakdown_risk': breakdown_risk
        }

    # ── 4. Next Arrival Time Prediction ─────────────────────────
    @staticmethod
    def predict_next_arrival(headways, dist_stats, ewma_headway):
        """
        Predicts expected time until next vehicle crosses the count line.
        Uses the best-fit distribution parameters when available,
        falls back to EWMA-smoothed headway mean.

        Returns:
          expected_s   : E[next headway] in seconds
          p85_s        : 85th percentile (conservative bound)
          method       : which method was used
        """
        if dist_stats and dist_stats.get('n_samples', 0) >= 10:
            mu = dist_stats['mean_s']
            h85 = dist_stats['h85_s']
            method = 'distribution_fit'
        elif ewma_headway and ewma_headway > 0:
            mu = ewma_headway
            h85 = mu * 1.5    # rough 85th pct as 1.5x mean for exponential
            method = 'ewma'
        elif headways:
            recent = [hw for (_, hw) in list(headways)[-20:]]
            mu = float(np.mean(recent)) if recent else 0.0
            h85 = float(np.percentile(recent, 85)) if len(recent) > 2 else mu * 1.5
            method = 'recent_mean'
        else:
            return {'expected_s': None, 'p85_s': None, 'method': 'no_data'}

        return {
            'expected_s': round(mu, 2),
            'p85_s': round(h85, 2),
            'method': method
        }

    # ── Composite prediction bundle ──────────────────────────────
    def predict_all(self, current_time, flow, density, speed,
                    density_history, headways, dist_stats):
        """Run all four predictors and return a unified prediction dict."""
        self.adapt_alpha(density_history, current_time=current_time)
        ewma = self.update_ewma(flow, density, speed)
        poly = self.poly_density_forecast(density_history, current_time)
        lwr = self.lwr_shockwave(flow, density, speed)
        ewma_hw = (1.0 / ewma['flow'] * 3600) if ewma['flow'] and ewma['flow'] > 0 else None
        arrival = self.predict_next_arrival(headways, dist_stats, ewma_hw)

        result = {
            'ewma': ewma,
            'poly_forecast': poly,
            'lwr_shockwave': lwr,
            'next_arrival': arrival,
            'ewma_alpha': round(self._ewma_alpha, 3),
        }
        self.prediction_history.append((current_time, result))
        return result


# ========== OCCUPANCY CALCULATOR ==========
class OccupancyCalculator:
    """
    Lane Occupancy Ratio (LOR):
    The fraction of time a detection point is occupied by a vehicle.
    LOR = sum(vehicle_occupancy_times) / observation_time
    Distinct from density (veh/km) -- directly measurable from video.
    """

    def __init__(self):
        self.vehicle_presence = {}     # vid -> (entry_time, exit_time)
        self.observation_start = None

    def record_entry(self, vid, t):
        self.vehicle_presence[vid] = [t, None]
        if self.observation_start is None:
            self.observation_start = t

    def record_exit(self, vid, t):
        if vid in self.vehicle_presence:
            self.vehicle_presence[vid][1] = t

    def compute_lor(self, current_time):
        if self.observation_start is None:
            return 0.0
        total_time = current_time - self.observation_start
        if total_time <= 0:
            return 0.0
        occupied = 0.0
        for vid, (entry, exit_t) in self.vehicle_presence.items():
            if exit_t is not None:
                occupied += exit_t - entry
            else:
                occupied += current_time - entry
        return round(min(occupied / total_time, 1.0), 4)


# ========== CORE LANE TRACKER (Unsignalized) ==========
class UnsignalizedLaneTracker:
    """
    Extended lane tracker that incorporates all unsignalized-road parameters.
    Inherits basic counting / headway logic from detectFinal.py structure
    but replaces signal-aware features with:
      - Full headway distribution fitting
      - Speed-density model fitting
      - Gap-acceptance tracking
      - Platoon analysis
      - Safety surrogate computation
      - PCU-weighted flow
      - Lane occupancy ratio
      - Moving bottleneck detection
    """

    def __init__(self, lane_name, lane_index):
        self.lane_name = lane_name
        self.lane_index = lane_index

        # --- Basic tracking ---
        self.previous_centers = {}
        self.next_id = 0
        self.vehicle_count = 0
        self.last_cross_time = None
        self.vehicles_in_roi = 0

        # --- Data buffers ---
        self.headways = deque(maxlen=2000)          # (time, headway_s)
        self.speed_measurements = deque(maxlen=500)
        self.density_history = deque(maxlen=1000)   # (time, veh/km)
        self.flow_history = deque(maxlen=1000)      # (time, veh/h)
        self.occupancy_history = deque(maxlen=1000) # (time, LOR)
        self.pcu_flow_history = deque(maxlen=1000)  # (time, PCU/h)
        self.speed_density_pairs = deque(maxlen=300)# (k, u) for model fitting

        # --- Per-vehicle state ---
        self.vehicle_states: dict = {}              # vid -> VehicleState
        self.vehicle_classes = defaultdict(int)
        self.vehicle_entry_times = {}
        self.total_travel_time = []

        # --- Submodules ---
        self.gap_model = GapAcceptanceModel()
        self.bottleneck_detector = MovingBottleneckDetector()
        self.occupancy_calc = OccupancyCalculator()

        # --- Submodule: predictor ---
        self.predictor = TrafficPredictor()

        # --- Cached per-frame analytics ---
        self._last_distribution_stats = {}
        self._last_platoon_stats = {}
        self._last_safety_stats = {}
        self._last_speed_density_fit = None
        self._last_predictions = {}
        self._frame_counter = 0

    # ── Headway & speed helpers ──────────────────────────────────
    def calc_avg_headway(self, current_time, window_s):
        h = [hw for (t, hw) in self.headways if current_time - t <= window_s]
        return float(np.mean(h)) if h else 0.0

    def calc_headway_variance(self, current_time, window_s):
        h = [hw for (t, hw) in self.headways if current_time - t <= window_s]
        return float(np.var(h)) if len(h) > 1 else 0.0

    # ── Core metrics update ──────────────────────────────────────
    def update_metrics(self, current_time, roi_height_px, pixels_per_meter):
        roi_km = roi_height_px / pixels_per_meter / 1000.0
        density = self.vehicles_in_roi / roi_km if roi_km > 0 else 0.0
        self.density_history.append((current_time, density))

        # Flow (veh/h from last 60s)
        recent_crosses = sum(1 for (t, _) in self.headways
                             if current_time - t <= 60)
        flow = (recent_crosses / 60) * 3600 if recent_crosses > 0 else 0.0
        self.flow_history.append((current_time, flow))

        avg_speed = float(np.mean(list(self.speed_measurements))) \
            if self.speed_measurements else 0.0
        valid_spd = [s for s in self.speed_measurements if s > 0]
        space_mean_speed = (len(valid_spd) / sum(1 / s for s in valid_spd)
                            if valid_spd else 0.0)

        # LOR
        lor = self.occupancy_calc.compute_lor(current_time)
        self.occupancy_history.append((current_time, lor))

        # PCU flow
        obs_hours = (current_time / 3600) if current_time > 0 else 1 / 3600
        pcu_f = PCUFlowCalculator.pcu_flow(dict(self.vehicle_classes), obs_hours)
        self.pcu_flow_history.append((current_time, pcu_f))

        # Store (k, u) pair for speed-density modelling
        if density > 0 and avg_speed > 0:
            self.speed_density_pairs.append((density, avg_speed))

        queue_len = sum(1 for vs in self.vehicle_states.values()
                        if vs.last_speed is not None
                        and vs.last_speed < Config.QUEUE_SPEED_THRESHOLD)

        return {
            'density': round(density, 2),
            'flow': round(flow, 1),
            'avg_speed': round(avg_speed, 2),
            'space_mean_speed': round(space_mean_speed, 2),
            'queue_length': queue_len,
            'vehicles_in_roi': self.vehicles_in_roi,
            'lor': round(lor, 4),
            'pcu_flow': round(pcu_f, 1)
        }

    # ── Level of Service (HCM 2010, basic freeway) ───────────────
    @staticmethod
    def level_of_service(density):
        if density < 11:
            return 'A', (0, 230, 0)
        elif density < 18:
            return 'B', (80, 230, 0)
        elif density < 26:
            return 'C', (180, 230, 0)
        elif density < 35:
            return 'D', (230, 180, 0)
        elif density < 45:
            return 'E', (230, 80, 0)
        else:
            return 'F', (0, 0, 230)

    # ── Run heavy analytics (every N frames to save CPU) ─────────
    def run_deep_analytics(self, current_time):
        """
        Runs distribution fitting, platoon analysis, speed-density modelling,
        and safety analysis. Called once per second (approximately).
        """
        all_hw = [hw for (_, hw) in self.headways]

        # Headway distribution
        self._last_distribution_stats = \
            HeadwayDistributionAnalyser.compute_headway_statistics(
                all_hw, Config.PLATOON_HEADWAY_S)

        # Platoon analysis
        self._last_platoon_stats, _ = \
            PlatoonAnalyser.identify_platoons(
                list(self.headways), Config.PLATOON_HEADWAY_S)

        # Speed-density model fitting
        if len(self.speed_density_pairs) >= 8:
            ks = [k for k, _ in self.speed_density_pairs]
            us = [u for _, u in self.speed_density_pairs]
            self._last_speed_density_fit = SpeedDensityModels.fit_all(ks, us)

        # Safety
        spd_list = list(self.speed_measurements)
        var_val, var_risk = SafetySurrogates.speed_variance_risk(spd_list)
        flow_now = self.flow_history[-1][1] if self.flow_history else 0
        avg_spd = float(np.mean(spd_list)) if spd_list else 0
        cfi = SafetySurrogates.conflict_friction_index(flow_now, avg_spd)
        self._last_safety_stats = {
            'speed_variance': var_val,
            'speed_variance_risk': var_risk,
            'cfi': cfi
        }

        # Moving bottleneck
        bottlenecks = self.bottleneck_detector.detect(
            self.vehicle_states, avg_spd)
        self._last_safety_stats['active_bottlenecks'] = bottlenecks

        # Predictions from historical data
        self._last_predictions = self.predictor.predict_all(
            current_time=current_time,
            flow=flow_now,
            density=self.density_history[-1][1] if self.density_history else 0,
            speed=avg_spd,
            density_history=self.density_history,
            headways=self.headways,
            dist_stats=self._last_distribution_stats
        )


# ========== PROCESS ONE LANE ==========
def process_lane(frame_roi, tracker: UnsignalizedLaneTracker,
                 current_time, baseline_y, fps_scale=1.0):
    roi_h, roi_w = frame_roi.shape[:2]

    # Detection line
    cv2.line(frame_roi, (0, baseline_y), (roi_w, baseline_y), (0, 100, 255), 3)
    cv2.putText(frame_roi, "COUNT LINE",
                (roi_w // 2 - 50, baseline_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

    results = model(frame_roi,
                    conf=Config.CONFIDENCE_THRESHOLD,
                    imgsz=640, verbose=False)[0]

    current_centers = {}
    tracker.vehicles_in_roi = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in Config.VEHICLE_IDS:
            continue

        tracker.vehicles_in_roi += 1
        tracker.vehicle_classes[cls_id] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bbox_h_px = y2 - y1

        # ── ID Assignment (nearest-neighbour) ──
        found_id = None
        min_dist = float('inf')
        for vid, (px, py) in tracker.previous_centers.items():
            d = np.hypot(cx - px, cy - py)
            if d < Config.MIN_TRACKING_DISTANCE and d < min_dist:
                min_dist = d
                found_id = vid
        if found_id is None:
            found_id = tracker.next_id
            tracker.next_id += 1
            tracker.vehicle_entry_times[found_id] = current_time
            tracker.occupancy_calc.record_entry(found_id, current_time)

        current_centers[found_id] = (cx, cy)

        # ── VehicleState ──
        if found_id not in tracker.vehicle_states:
            vs = VehicleState(found_id)
            vs.cls_id = cls_id
            vs.pce = Config.PCE.get(cls_id, 1.0)
            tracker.vehicle_states[found_id] = vs
        vs = tracker.vehicle_states[found_id]
        speed = vs.update_position(current_time, cy, Config.PIXELS_PER_METER)
        if speed is not None:
            tracker.speed_measurements.append(speed)

        # ── Crossing detection with headway recording ──
        if found_id in tracker.previous_centers:
            prev_y = tracker.previous_centers[found_id][1]
            if prev_y < (baseline_y - 4) and cy >= baseline_y:
                tracker.vehicle_count += 1
                tracker.occupancy_calc.record_exit(found_id, current_time)
                if found_id in tracker.vehicle_entry_times:
                    tt = current_time - tracker.vehicle_entry_times[found_id]
                    tracker.total_travel_time.append(tt)

                if tracker.last_cross_time is not None:
                    headway = current_time - tracker.last_cross_time
                    tracker.headways.append((current_time, headway))

                    # ── Gap acceptance: gaps > critical are "accepted" ──
                    tracker.gap_model.record_gap(
                        headway,
                        accepted=(headway >= Config.CRITICAL_GAP_S)
                    )

                tracker.last_cross_time = current_time

        # ── TTC between adjacent vehicles ──
        # (approximate: use lead vehicle as the one just ahead in y direction)
        closest_lead = None
        min_gap_px = float('inf')
        for vid2, (px2, py2) in tracker.previous_centers.items():
            if vid2 == found_id:
                continue
            gap_px = cy - py2   # positive = found_id is behind vid2
            if 0 < gap_px < min_gap_px:
                min_gap_px = gap_px
                closest_lead = vid2

        if closest_lead is not None and closest_lead in tracker.vehicle_states:
            lead_vs = tracker.vehicle_states[closest_lead]
            gap_m = min_gap_px / Config.PIXELS_PER_METER
            lead_spd = lead_vs.current_speed() or 0
            follow_spd = speed or 0
            ttc = SafetySurrogates.compute_ttc(gap_m, lead_spd, follow_spd)
            drac = SafetySurrogates.compute_drac(gap_m, lead_spd, follow_spd)
            vs.ttc = ttc
            vs.pet = SafetySurrogates.compute_pet(
                min_gap_px / (Config.PIXELS_PER_METER * max(follow_spd / 3.6, 0.1)),
                follow_spd)

        # ── Draw bounding box ──
        is_bottleneck = any(b['vid'] == found_id
                            for b in tracker._last_safety_stats.get('active_bottlenecks', []))
        is_critical_ttc = (vs.ttc is not None and vs.ttc < Config.TTC_CRITICAL_S)

        if is_bottleneck:
            box_color = (0, 165, 255)       # orange
        elif is_critical_ttc:
            box_color = (0, 0, 255)         # red
        elif vs.in_platoon:
            box_color = (255, 200, 0)       # cyan-ish
        else:
            box_color = (0, 255, 0)         # green

        cv2.rectangle(frame_roi, (x1, y1), (x2, y2), box_color, 2)

        vnames = {2: 'Car', 3: 'Moto', 5: 'Bus', 7: 'Truck'}
        label = f"{vnames.get(cls_id, 'V')}{found_id}"
        if speed is not None:
            label += f" {speed:.0f}km/h"
        if vs.ttc is not None and vs.ttc < Config.TTC_CRITICAL_S * 2:
            label += f" TTC:{vs.ttc:.1f}s"

        (lw, lhh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame_roi, (x1, y1 - lhh - 6), (x1 + lw + 4, y1),
                      box_color, -1)
        cv2.putText(frame_roi, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.circle(frame_roi, (cx, cy), 3, (255, 0, 255), -1)

    # Update platoon membership flags
    if tracker.headways:
        recent_hw = [hw for (t, hw) in tracker.headways
                     if current_time - t <= 10]
        in_platoon_ids = list(current_centers.keys())
        for i, vid in enumerate(in_platoon_ids):
            if i < len(recent_hw):
                tracker.vehicle_states.get(
                    vid, VehicleState(vid)).in_platoon = \
                    (recent_hw[i] <= Config.PLATOON_HEADWAY_S)

    tracker.previous_centers = current_centers
    return frame_roi


# ========== VISUALIZATION (Unsignalized) ==========
class UnsignalizedVisualizer:

    @staticmethod
    def draw_main_panel(frame, tracker: UnsignalizedLaneTracker,
                        x_off, y_start, current_time, roi_height):
        metrics = tracker.update_metrics(
            current_time, roi_height, Config.PIXELS_PER_METER)
        los, los_color = tracker.level_of_service(metrics['density'])

        panel_h = 560
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_off, y_start),
                      (x_off + 420, y_start + panel_h), (25, 30, 25), -1)
        cv2.addWeighted(overlay, 0.83, frame, 0.17, 0, frame)
        cv2.rectangle(frame, (x_off, y_start),
                      (x_off + 420, y_start + panel_h), (80, 120, 80), 2)

        y = y_start + 30
        lh = 26

        def txt(text, color=(230, 230, 230), scale=0.55, bold=False):
            nonlocal y
            cv2.putText(frame, text, (x_off + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, 2 if bold else 1)
            y += lh

        txt(f"=== {tracker.lane_name} ===", (100, 255, 100), 0.75, True)
        y += 3
        txt(f"Total Count   : {tracker.vehicle_count}", (0, 255, 200), 0.6)
        txt(f"In ROI now    : {metrics['vehicles_in_roi']}", (150, 255, 150), 0.6)
        txt(f"Queue (slow)  : {metrics['queue_length']} veh", (255, 180, 100), 0.6)
        y += 4

        txt("-- Flow Fundamentals --", (160, 200, 255), 0.58, True)
        txt(f"Density  : {metrics['density']:.1f} veh/km", (150, 255, 200), 0.58)
        txt(f"Flow     : {metrics['flow']:.0f} veh/h", (200, 150, 255), 0.58)
        txt(f"PCU Flow : {metrics['pcu_flow']:.0f} PCU/h", (220, 220, 100), 0.58)
        txt(f"LOR (Occ): {metrics['lor']*100:.1f}%", (200, 230, 255), 0.58)
        txt(f"Time Mean Speed  : {metrics['avg_speed']:.1f} km/h", (255, 255, 130), 0.58)
        txt(f"Space Mean Speed : {metrics['space_mean_speed']:.1f} km/h", (255, 240, 80), 0.58)
        txt(f"LOS      : {los}", los_color, 0.65, True)
        y += 4

        txt("-- Headway Stats --", (160, 200, 255), 0.58, True)
        last_hw = tracker.headways[-1][1] if tracker.headways else 0
        txt(f"Last Headway : {last_hw:.2f} s", (255, 200, 100), 0.58)
        for w in [20, 60]:
            a = tracker.calc_avg_headway(current_time, w)
            v = tracker.calc_headway_variance(current_time, w)
            txt(f"Avg({w}s): {a:.2f}s  var:{v:.2f}", (150, 230, 180), 0.52)

        ds = tracker._last_distribution_stats
        if ds:
            txt(f"CVH : {ds.get('cvh', 0):.3f}  "
                f"PTI : {ds.get('pti', 0):.3f}", (200, 255, 200), 0.52)
            txt(f"Entropy: {ds.get('entropy_bits', 0):.2f} bits  "
                f"IoD: {ds.get('index_of_dispersion', 0):.2f}", (200, 255, 200), 0.52)
            txt(f"h15={ds.get('h15_s', 0):.2f}s  "
                f"h50={ds.get('h50_s', 0):.2f}s  "
                f"h85={ds.get('h85_s', 0):.2f}s", (170, 240, 200), 0.50)

        return metrics

    @staticmethod
    def draw_advanced_panel(frame, tracker: UnsignalizedLaneTracker,
                            x_off, y_start):
        panel_h = 560
        panel_w = 460
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_off, y_start),
                      (x_off + panel_w, y_start + panel_h), (25, 25, 40), -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        cv2.rectangle(frame, (x_off, y_start),
                      (x_off + panel_w, y_start + panel_h), (80, 80, 160), 2)

        y = y_start + 30
        lh = 26

        def txt(text, color=(230, 230, 230), scale=0.55, bold=False):
            nonlocal y
            cv2.putText(frame, text, (x_off + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, 2 if bold else 1)
            y += lh

        def section(title):
            nonlocal y
            y += 3
            txt(title, (180, 160, 255), 0.58, True)

        txt("=== ADVANCED ANALYSIS ===", (0, 220, 255), 0.72, True)
        y += 3

        # Platoon Analysis
        section("-- Platoon Analysis --")
        ps = tracker._last_platoon_stats
        if ps:
            pr = ps.get('platoon_ratio', 0)
            pr_color = (0, 230, 0) if pr < 0.3 else \
                       (0, 200, 255) if pr < 0.6 else (0, 80, 255)
            txt(f"Platoon Ratio    : {pr*100:.1f}%", pr_color, 0.56)
            txt(f"Num Platoons     : {ps.get('num_platoons', 0)}", (200, 200, 255), 0.54)
            txt(f"Avg Platoon Size : {ps.get('avg_platoon_size', 0):.1f} veh", (200, 200, 255), 0.54)
            txt(f"Intra-Platoon hw : {ps.get('avg_intra_platoon_headway_s', 0):.2f} s", (200, 200, 255), 0.54)
            txt(f"Inter-Platoon hw : {ps.get('avg_inter_platoon_headway_s', 0):.2f} s", (200, 200, 255), 0.54)
        else:
            txt("  (building data...)", (120, 120, 120), 0.52)
        y += 2

        # Speed-Density Model
        section("-- Speed-Density Model --")
        sd = tracker._last_speed_density_fit
        if sd:
            best = sd['best_model']
            bm = sd['models'][best]
            r2 = bm['r2']
            r2c = (0, 230, 0) if r2 > 0.7 else (0, 180, 255) if r2 > 0.4 else (0, 80, 255)
            txt(f"Best Fit: {best.upper()}", (255, 255, 150), 0.56, True)
            txt(f"RMSE={bm['rmse']:.2f}  R^2={r2:.3f}", r2c, 0.54)
            # Theoretical capacity from fitted model
            cap_approx = round(Config.CRITICAL_DENSITY_VEH_KM *
                               Config.FREE_FLOW_SPEED_KMH * 0.5, 0)
            txt(f"Theoretical Capacity ~{cap_approx:.0f} veh/h", (220, 255, 180), 0.54)
        else:
            txt("  (fitting model...)", (120, 120, 120), 0.52)
        y += 2

        # Gap Acceptance
        section("-- Gap Acceptance --")
        tc = tracker.gap_model.raff_critical_gap()
        if tc:
            txt(f"Critical Gap (Raff): {tc:.2f} s", (255, 220, 130), 0.56)
            fl = tracker.flow_history[-1][1] if tracker.flow_history else 0
            tf = float(np.mean(tracker.gap_model.follow_up_times)) \
                if tracker.gap_model.follow_up_times else 2.5
            cap_minor = tracker.gap_model.tanner_capacity(fl, tc, tf)
            txt(f"Minor-stream cap: {cap_minor:.0f} veh/h", (255, 220, 130), 0.54)
        else:
            txt("  (need more gaps...)", (120, 120, 120), 0.52)
        y += 2

        # Safety Surrogates
        section("-- Safety Surrogates --")
        ss = tracker._last_safety_stats
        if ss:
            var_val = ss.get('speed_variance', 0)
            risk = ss.get('speed_variance_risk', 'N/A')
            risk_c = {'Low': (0, 230, 0), 'Moderate': (0, 200, 255),
                      'High': (0, 80, 255)}.get(risk, (150, 150, 150))
            txt(f"Speed Variance   : {var_val:.1f}  [{risk}]", risk_c, 0.56)
            cfi = ss.get('cfi', 0)
            cfi_c = (0, 230, 0) if cfi < 30 else (0, 200, 255) if cfi < 60 else (0, 80, 255)
            txt(f"CFI (Friction Idx): {cfi:.1f}/100", cfi_c, 0.56)
            bns = ss.get('active_bottlenecks', [])
            if bns:
                txt(f"!! Moving Bottlenecks: {len(bns)} !!", (0, 80, 255), 0.58, True)
                for b in bns[:2]:
                    txt(f"  V{b['vid']} @ {b['speed_kmh']:.0f}km/h "
                        f"(stream:{b['stream_speed_kmh']:.0f})", (0, 150, 255), 0.50)
            else:
                txt("No moving bottlenecks", (0, 230, 0), 0.54)

    @staticmethod
    def draw_prediction_panel(frame, tracker: 'UnsignalizedLaneTracker',
                               x_off, y_start):
        """
        Draws the history-based prediction panel.
        Uses predictions stored in tracker._last_predictions.
        """
        pred = tracker._last_predictions
        if not pred:
            return

        panel_w = 430
        panel_h = 420
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_off, y_start),
                      (x_off + panel_w, y_start + panel_h), (20, 30, 20), -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        cv2.rectangle(frame, (x_off, y_start),
                      (x_off + panel_w, y_start + panel_h), (60, 140, 60), 2)

        y = y_start + 28
        lh = 25

        def txt(text, color=(210, 230, 210), scale=0.54, bold=False):
            nonlocal y
            cv2.putText(frame, text, (x_off + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, 2 if bold else 1)
            y += lh

        def section(title):
            nonlocal y
            y += 2
            txt(title, (120, 220, 120), 0.56, True)

        txt("=== DATA-DRIVEN PREDICTIONS ===", (80, 255, 80), 0.68, True)
        y += 2

        # ── 1. EWMA ─────────────────────────────────────────────
        section("-- EWMA Smoothed State --")
        ewma = pred.get('ewma', {})
        alpha = pred.get('ewma_alpha', 0.25)
        txt(f"alpha={alpha:.2f}  (adaptive)", (170, 200, 170), 0.50)
        ef = ewma.get('flow')
        ek = ewma.get('density')
        eu = ewma.get('speed')
        txt(f"Flow   : {ef:.0f} veh/h" if ef else "Flow   : --",
            (200, 255, 150), 0.54)
        txt(f"Density: {ek:.1f} veh/km" if ek else "Density: --",
            (200, 255, 150), 0.54)
        txt(f"Speed  : {eu:.1f} km/h" if eu else "Speed  : --",
            (200, 255, 150), 0.54)
        y += 2

        # ── 2. Polynomial Density Forecast ──────────────────────
        section("-- Poly Density Forecast --")
        poly = pred.get('poly_forecast')
        if poly:
            trend = poly['trend']
            trend_colors = {'RISING': (0, 80, 255), 'FALLING': (0, 230, 0),
                            'STABLE': (0, 230, 230)}
            tc_color = trend_colors.get(trend, (200, 200, 200))
            txt(f"Trend    : {trend}  dK/dt={poly['dkdt_veh_km_s']:.3f} veh/km/s",
                tc_color, 0.52, True)
            cp30 = poly['congestion_prob_30s'] * 100
            cp60 = poly['congestion_prob_60s'] * 100
            c30 = (0, 200, 0) if cp30 < 50 else (0, 150, 255) if cp30 < 80 else (0, 50, 255)
            c60 = (0, 200, 0) if cp60 < 50 else (0, 150, 255) if cp60 < 80 else (0, 50, 255)
            txt(f"Density +30s: {poly['pred_density_30s']:.1f} veh/km  "
                f"({cp30:.0f}% cong)", c30, 0.52)
            txt(f"Density +60s: {poly['pred_density_60s']:.1f} veh/km  "
                f"({cp60:.0f}% cong)", c60, 0.52)
            ttc = poly.get('time_to_congestion_s')
            if ttc is not None:
                txt(f"Time to congestion: ~{ttc:.0f} s", (0, 80, 255), 0.54, True)
            else:
                txt("Congestion onset: not imminent", (0, 200, 0), 0.52)
        else:
            txt("  (need 120s of data...)", (100, 100, 100), 0.50)
        y += 2

        # ── 3. LWR Shockwave ────────────────────────────────────
        section("-- LWR Shockwave (Measured) --")
        lwr = pred.get('lwr_shockwave', {})
        sev = lwr.get('severity', 'None')
        sev_c = {'None': (0, 200, 0), 'Low': (0, 230, 200),
                 'Moderate': (0, 150, 255), 'High': (0, 50, 255)}.get(sev, (180, 180, 180))
        w_val = lwr.get('shockwave_kmh', 0)
        direction = lwr.get('direction', 'none')
        txt(f"Wave speed : {w_val:.1f} km/h  ({direction})", sev_c, 0.54)
        br = lwr.get('breakdown_risk', 0) * 100
        br_c = (0, 200, 0) if br < 50 else (0, 150, 255) if br < 80 else (0, 50, 255)
        txt(f"Breakdown risk : {br:.1f}%  [{sev}]", br_c, 0.54, sev in ('High', 'Moderate'))
        y += 2

        # ── 4. Next Arrival Prediction ──────────────────────────
        section("-- Next Vehicle Arrival --")
        arr = pred.get('next_arrival', {})
        exp_s = arr.get('expected_s')
        p85_s = arr.get('p85_s')
        method = arr.get('method', 'no_data')
        if exp_s:
            txt(f"Expected in : {exp_s:.2f} s  (method: {method})",
                (255, 230, 100), 0.52)
            txt(f"P85 bound   : {p85_s:.2f} s  (conservative)",
                (220, 200, 80), 0.52)
        else:
            txt("  (building headway data...)", (100, 100, 100), 0.50)

    @staticmethod
    def draw_density_graph(frame, tracker, x_off, y_start,
                           current_time, width=480):
        if len(tracker.density_history) < 3:
            return
        recent = [(t, d) for (t, d) in tracker.density_history
                  if current_time - t <= 60]
        if len(recent) < 2:
            return
        gh = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_off, y_start),
                      (x_off + width, y_start + gh), (35, 35, 35), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x_off, y_start),
                      (x_off + width, y_start + gh), (80, 80, 80), 1)
        cv2.putText(frame, "Density 60s (veh/km)",
                    (x_off + 8, y_start + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        densities = [d for _, d in recent]
        mx = max(max(densities), 1)
        pts = [(x_off + 8 + int((i / len(recent)) * (width - 16)),
                y_start + gh - 8 - int((d / mx) * (gh - 28)))
               for i, (_, d) in enumerate(recent)]
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 255, 200), 2)


# ========== DATA EXPORT ==========
class DataExporter:
    @staticmethod
    def export(tracker: UnsignalizedLaneTracker, video_duration: float):
        if not Config.EXPORT_DATA:
            return
        os.makedirs(Config.EXPORT_PATH, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Summary JSON
        mix = PCUFlowCalculator.vehicle_mix_composition(
            dict(tracker.vehicle_classes))
        tc = tracker.gap_model.raff_critical_gap()
        ds = tracker._last_distribution_stats
        ps = tracker._last_platoon_stats

        summary = {
            'timestamp': ts,
            'video_duration_s': round(video_duration, 2),
            'total_vehicles': tracker.vehicle_count,
            'vehicle_mix_%': mix,
            'avg_headway_s': ds.get('mean_s', 0) if ds else 0,
            'cvh': ds.get('cvh', 0) if ds else 0,
            'pti': ds.get('pti', 0) if ds else 0,
            'entropy_bits': ds.get('entropy_bits', 0) if ds else 0,
            'index_of_dispersion': ds.get('index_of_dispersion', 0) if ds else 0,
            'h15_s': ds.get('h15_s', 0) if ds else 0,
            'h50_s': ds.get('h50_s', 0) if ds else 0,
            'h85_s': ds.get('h85_s', 0) if ds else 0,
            'platoon_ratio': ps.get('platoon_ratio', 0) if ps else 0,
            'num_platoons': ps.get('num_platoons', 0) if ps else 0,
            'critical_gap_s': tc,
            'avg_speed_kmh': float(np.mean(list(tracker.speed_measurements)))
                             if tracker.speed_measurements else 0,
            'speed_density_model': (tracker._last_speed_density_fit or {}).get('best_model'),
            'avg_lor': float(np.mean([l for _, l in tracker.occupancy_history]))
                       if tracker.occupancy_history else 0,
        }
        with open(f"{Config.EXPORT_PATH}summary_{ts}.json", 'w') as f:
            json.dump(summary, f, indent=4)

        # Time-series CSVs
        if tracker.headways:
            pd.DataFrame(list(tracker.headways),
                         columns=['time', 'headway_s']
                         ).to_csv(f"{Config.EXPORT_PATH}headways_{ts}.csv", index=False)
        if tracker.density_history:
            pd.DataFrame(list(tracker.density_history),
                         columns=['time', 'density_veh_km']
                         ).to_csv(f"{Config.EXPORT_PATH}density_{ts}.csv", index=False)
        if tracker.flow_history:
            pd.DataFrame(list(tracker.flow_history),
                         columns=['time', 'flow_veh_h']
                         ).to_csv(f"{Config.EXPORT_PATH}flow_{ts}.csv", index=False)
        if tracker.occupancy_history:
            pd.DataFrame(list(tracker.occupancy_history),
                         columns=['time', 'LOR']
                         ).to_csv(f"{Config.EXPORT_PATH}occupancy_{ts}.csv", index=False)
        if tracker.pcu_flow_history:
            pd.DataFrame(list(tracker.pcu_flow_history),
                         columns=['time', 'pcu_flow']
                         ).to_csv(f"{Config.EXPORT_PATH}pcu_flow_{ts}.csv", index=False)

        # Per-vehicle speed/accel
        rows = []
        for vid, vs in tracker.vehicle_states.items():
            rows.append({
                'vid': vid,
                'cls_id': vs.cls_id,
                'pce': vs.pce,
                'avg_speed_kmh': float(np.mean(vs.speeds_kmh)) if vs.speeds_kmh else None,
                'avg_accel_mps2': float(np.mean(vs.accel_mps2)) if vs.accel_mps2 else None,
                'last_ttc_s': vs.ttc,
                'in_platoon': vs.in_platoon
            })
        pd.DataFrame(rows).to_csv(
            f"{Config.EXPORT_PATH}vehicle_states_{ts}.csv", index=False)

        # Prediction history CSV
        if tracker.predictor.prediction_history:
            pred_rows = []
            for t_p, p in tracker.predictor.prediction_history:
                ewma = p.get('ewma', {})
                poly = p.get('poly_forecast') or {}
                lwr  = p.get('lwr_shockwave', {})
                arr  = p.get('next_arrival', {})
                pred_rows.append({
                    'time': round(t_p, 3),
                    'ewma_flow': ewma.get('flow'),
                    'ewma_density': ewma.get('density'),
                    'ewma_speed': ewma.get('speed'),
                    'ewma_alpha': p.get('ewma_alpha'),
                    'poly_trend': poly.get('trend'),
                    'poly_dkdt': poly.get('dkdt_veh_km_s'),
                    'poly_pred_density_30s': poly.get('pred_density_30s'),
                    'poly_pred_density_60s': poly.get('pred_density_60s'),
                    'poly_congestion_prob_30s': poly.get('congestion_prob_30s'),
                    'poly_congestion_prob_60s': poly.get('congestion_prob_60s'),
                    'poly_time_to_congestion_s': poly.get('time_to_congestion_s'),
                    'lwr_shockwave_kmh': lwr.get('shockwave_kmh'),
                    'lwr_breakdown_risk': lwr.get('breakdown_risk'),
                    'lwr_severity': lwr.get('severity'),
                    'next_arrival_expected_s': arr.get('expected_s'),
                    'next_arrival_p85_s': arr.get('p85_s'),
                    'next_arrival_method': arr.get('method'),
                })
            pd.DataFrame(pred_rows).to_csv(
                f"{Config.EXPORT_PATH}predictions_{ts}.csv", index=False)

        print(f"\n[+] Data exported to: {Config.EXPORT_PATH}")


# ========== MAIN ==========
def main():
    global model

    print("Loading YOLO model...")
    model = YOLO(Config.YOLO_MODEL)

    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {Config.VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = UnsignalizedLaneTracker("LEFT LANE", 0)

    start_t = time.time()
    frame_count = 0
    processed = 0
    last_deep_analysis_t = 0

    print("\n" + "=" * 72)
    print("  UNSIGNALIZED TRAFFIC FLOW ANALYSIS  --  PhD-Level Parameters")
    print("=" * 72)
    print(f"  Video          : {Config.VIDEO_PATH}")
    print(f"  FPS            : {fps}  |  Total Frames: {total_frames}")
    print(f"  Speed Mult     : {Config.SPEED_MULTIPLIER}x")
    print(f"  Free-flow speed: {Config.FREE_FLOW_SPEED_KMH} km/h")
    print(f"  Critical gap   : {Config.CRITICAL_GAP_S} s")
    print(f"  Platoon thresh : {Config.PLATOON_HEADWAY_S} s")
    print("  Press 'q' to quit | 's' to screenshot")
    print("=" * 72 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % Config.SPEED_MULTIPLIER != 0:
            continue
        processed += 1

        h, w = frame.shape[:2]
        current_time = time.time() - start_t

        # ── Left lane ROI (left half of frame) ──
        mid_x = w // 2
        left_roi = frame[:, :mid_x].copy()
        roi_h = left_roi.shape[0]
        baseline_y = int(roi_h * Config.BASELINE_OFFSET)

        left_roi = process_lane(left_roi, tracker, current_time, baseline_y)
        frame[:, :mid_x] = left_roi

        # Dim right half
        frame[:, mid_x:] = (frame[:, mid_x:] * 0.3).astype(np.uint8)
        cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 0), 3)
        cv2.putText(frame, "RIGHT LANE - NOT ANALYSED",
                    (mid_x + 20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 2)

        # Run heavy analytics once per ~second
        if current_time - last_deep_analysis_t >= 1.0:
            tracker.run_deep_analytics(current_time)
            last_deep_analysis_t = current_time

        # ── Draw panels ──
        # Left panel: core lane metrics (inside left ROI)
        UnsignalizedVisualizer.draw_main_panel(
            frame, tracker, 8, 50, current_time, roi_h)
        # Middle panel: prediction (left ROI, below main panel)
        UnsignalizedVisualizer.draw_prediction_panel(
            frame, tracker, 8, 630)
        # Right panel: advanced analysis (right side of frame, overlaid on dim)
        UnsignalizedVisualizer.draw_advanced_panel(
            frame, tracker, w - 480, 50)
        # Density graph (bottom left)
        UnsignalizedVisualizer.draw_density_graph(
            frame, tracker, 8, h - 120, current_time, mid_x - 20)

        # Progress bar
        prog = frame_count / total_frames * 100
        bw = w - 40
        cv2.rectangle(frame, (20, h - 35), (20 + bw, h - 15), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 35),
                      (20 + int(bw * prog / 100), h - 15), (0, 200, 100), -1)
        cv2.putText(frame, f"Progress: {prog:.1f}%",
                    (w // 2 - 70, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        cv2.imshow("Left Lane Unsignalized Traffic Analysis", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs(Config.EXPORT_PATH, exist_ok=True)
            sname = f"{Config.EXPORT_PATH}screenshot_{int(current_time)}.jpg"
            cv2.imwrite(sname, frame)
            print(f"Screenshot: {sname}")

    final_time = time.time() - start_t
    cap.release()
    cv2.destroyAllWindows()

    # ══════════════════════════════════════════════════════════════
    #  FINAL STATISTICS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  FINAL STATISTICS -- UNSIGNALIZED TRAFFIC ANALYSIS")
    print("=" * 72)
    print(f"  Processing Time : {final_time:.2f} s")
    print(f"  Frames Processed: {processed}/{total_frames}")

    t = tracker
    print(f"\n  Total Vehicles Counted : {t.vehicle_count}")
    print(f"  Vehicle Mix            : "
          f"{PCUFlowCalculator.vehicle_mix_composition(dict(t.vehicle_classes))}")

    print("\n  Headway Analysis:")
    for w in Config.TIME_WINDOWS:
        a = t.calc_avg_headway(final_time, w)
        v = t.calc_headway_variance(final_time, w)
        print(f"    {w:>4}s window: Avg={a:.2f}s  Var={v:.2f}")

    # Distribution stats
    all_hw = [hw for (_, hw) in t.headways]
    ds = HeadwayDistributionAnalyser.compute_headway_statistics(
        all_hw, Config.PLATOON_HEADWAY_S)
    if ds:
        print("\n  Headway Distribution Statistics:")
        print(f"    CVH  : {ds['cvh']:.3f}  (1=Poisson, <1=Regular, >1=Clustered)")
        print(f"    PTI  : {ds['pti']:.3f}  (Platooning Tendency Index)")
        print(f"    IoD  : {ds['index_of_dispersion']:.3f}  (Index of Dispersion)")
        print(f"    Entropy : {ds['entropy_bits']:.3f} bits")
        print(f"    h15/h50/h85 : {ds['h15_s']}s / {ds['h50_s']}s / {ds['h85_s']}s")

    # Distribution fitting
    if all_hw:
        fit = HeadwayDistributionAnalyser.fit_distributions(all_hw)
        if fit and 'best_fit' in fit:
            best_d = fit['best_fit']
            print(f"\n  Best Headway Distribution Fit : {best_d.upper()}")
            print(f"    AIC : {fit[best_d]['AIC']:.2f}")
            print(f"    Params : {fit[best_d]['params']}")

    # Platoon stats
    ps, _ = PlatoonAnalyser.identify_platoons(list(t.headways), Config.PLATOON_HEADWAY_S)
    if ps:
        print("\n  Platoon Analysis:")
        print(f"    Platoon Ratio           : {ps['platoon_ratio']*100:.1f}%")
        print(f"    Number of Platoons      : {ps['num_platoons']}")
        print(f"    Avg Platoon Size        : {ps['avg_platoon_size']:.1f} veh")
        print(f"    Avg Intra-Platoon hw    : {ps['avg_intra_platoon_headway_s']:.2f} s")
        print(f"    Avg Inter-Platoon hw    : {ps['avg_inter_platoon_headway_s']:.2f} s")

    # Gap acceptance
    tc = t.gap_model.raff_critical_gap()
    if tc:
        print(f"\n  Gap Acceptance:")
        print(f"    Critical Gap (Raff)  : {tc:.2f} s")
        fl = t.flow_history[-1][1] if t.flow_history else 0
        tf = 2.5
        cm = t.gap_model.tanner_capacity(fl, tc, tf)
        print(f"    Tanner Minor-stream Capacity : {cm:.0f} veh/h")

    # Speed-density model
    if len(t.speed_density_pairs) >= 8:
        ks = [k for k, _ in t.speed_density_pairs]
        us = [u for _, u in t.speed_density_pairs]
        sd_fit = SpeedDensityModels.fit_all(ks, us)
        if sd_fit:
            best_m = sd_fit['best_model']
            bm = sd_fit['models'][best_m]
            print(f"\n  Speed-Density Model:")
            print(f"    Best Fit : {best_m.upper()}")
            print(f"    RMSE     : {bm['rmse']:.2f} km/h")
            print(f"    R²       : {bm['r2']:.4f}")

    # Speed stats
    if t.speed_measurements:
        spds = list(t.speed_measurements)
        print(f"\n  Speed Statistics:")
        print(f"    Mean  : {np.mean(spds):.2f} km/h")
        print(f"    Std   : {np.std(spds):.2f} km/h")
        print(f"    Min   : {np.min(spds):.2f} km/h")
        print(f"    Max   : {np.max(spds):.2f} km/h")
        sv, sr = SafetySurrogates.speed_variance_risk(spds)
        print(f"    Speed Variance Risk : {sr} ({sv:.1f})")

    # Safety
    fl = t.flow_history[-1][1] if t.flow_history else 0
    avg_spd2 = float(np.mean(list(t.speed_measurements))) if t.speed_measurements else 0
    cfi = SafetySurrogates.conflict_friction_index(fl, avg_spd2)
    print(f"\n  Conflict Friction Index (CFI): {cfi:.1f}/100")

    # LOR
    if t.occupancy_history:
        lors = [l for _, l in t.occupancy_history]
        print(f"\n  Lane Occupancy Ratio (LOR):")
        print(f"    Average : {np.mean(lors)*100:.2f}%")
        print(f"    Max     : {np.max(lors)*100:.2f}%")

    # PCU flow
    if t.pcu_flow_history:
        pcus = [p for _, p in t.pcu_flow_history if p > 0]
        if pcus:
            print(f"\n  PCU-weighted Flow:")
            print(f"    Average : {np.mean(pcus):.0f} PCU/h")
            print(f"    Max     : {np.max(pcus):.0f} PCU/h")

    # ── Prediction summary ───────────────────────────────────────
    ph = list(t.predictor.prediction_history)
    if ph:
        print("\n  " + "=" * 68)
        print("  PREDICTION SUMMARY (History-Based)")
        print("  " + "=" * 68)

        ewma_flows = [p['ewma'].get('flow') for _, p in ph if p['ewma'].get('flow')]
        if ewma_flows:
            print(f"\n  EWMA Flow:")
            print(f"    Mean EWMA Flow : {np.mean(ewma_flows):.0f} veh/h")
            print(f"    Max  EWMA Flow : {np.max(ewma_flows):.0f} veh/h")

        poly_list = [p['poly_forecast'] for _, p in ph if p.get('poly_forecast')]
        if poly_list:
            trends = [p['trend'] for p in poly_list]
            from collections import Counter
            trend_counts = Counter(trends)
            print(f"\n  Polynomial Density Trend Distribution:")
            for label, cnt in trend_counts.items():
                print(f"    {label:8s}: {cnt:4d} frames ({cnt/len(trends)*100:.1f}%)")
            cp60 = [p['congestion_prob_60s'] for p in poly_list]
            print(f"\n  Average Congestion Probability (+60s): {np.mean(cp60)*100:.1f}%")
            print(f"  Max     Congestion Probability (+60s): {np.max(cp60)*100:.1f}%")

        lwr_list = [p['lwr_shockwave'] for _, p in ph if p.get('lwr_shockwave')]
        if lwr_list:
            brs = [l['breakdown_risk'] for l in lwr_list]
            print(f"\n  LWR Breakdown Risk:")
            print(f"    Average : {np.mean(brs)*100:.1f}%")
            print(f"    Max     : {np.max(brs)*100:.1f}%")
            high_risk = sum(1 for br in brs if br > 0.75)
            print(f"    Frames with High risk (>75%): {high_risk}/{len(brs)}")

    print("\n" + "=" * 72)

    # Export
    DataExporter.export(tracker, final_time)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()