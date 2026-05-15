#!/usr/bin/env python3
"""
Dual‑aware cascaded MPC for ACC with front and rear ultrasonic sensors.
Hardware pins:
  Front HC‑SR04  TRIG=GPIO23  ECHO=GPIO24
  Rear  HC‑SR04  TRIG=GPIO20  ECHO=GPIO21
  Motor L: IN1=GPIO5  IN2=GPIO6   ENA=GPIO27 (PWM)
  Motor R: IN3=GPIO19 IN4=GPIO26  ENB=GPIO22 (PWM)
  Encoder: GPIO17 (single‑channel, pull‑up)
  Wheel: 348 counts/rev, circumference 0.2042 m
"""

import math
import time
import signal
import csv
from pathlib import Path
import numpy as np

# GPIO and sensor libraries
try:
    from gpiozero import DistanceSensor, Motor, PWMOutputDevice
    from gpiozero.pins.pigpio import PiGPIOFactory
    import RPi.GPIO as _GPIO
    ON_PI = True
    factory = PiGPIOFactory()
except ImportError:
    ON_PI = False
    factory = None
    _GPIO = None
    print("[WARN] gpiozero/pigpio not available – mock mode")

# ============================================================================
# Hardware Pins
# ============================================================================
ENABLE_LOCAL_LOGGING = True
LOG_DIR = Path.home() / "revanthdualaware"
FRONT_TRIG = 23
FRONT_ECHO = 24
REAR_TRIG  = 20
REAR_ECHO  = 21

IN1, IN2 = 5, 6
IN3, IN4 = 19, 26
ENA, ENB = 27, 22
ENCODER_PIN = 16


COUNTS_PER_REV = 12 * 29          # 348 pulses per revolution
WHEEL_CIRCUM   = math.pi * 0.065  # 0.2042 m

# ============================================================================
# MPC Parameters (same as front‑only, plus dual‑aware weights)
# ============================================================================
class P:
    # Low‑level MPC (MPC2) – unchanged from working front‑only version
    Np      = 10
    Nc      = 3
    Q       = 300.0      # speed tracking weight
    R       = 50.0       # control effort weight
    DUmax   = 0.05
    DUmin   = -0.15
    kT      = 1.0745

    # Speed / headway policy
    vset    = 0.38       # desired cruise speed (m/s)
    hw      = 0.70       # time headway (s)
    sd      = 0.25       # standstill distance (m)
    hs      = 0.30       # hold speed distance (m)
    ah      = 0.30       # acceleration gain (m/s² per m error)

    # Vehicle physics
    mass    = 0.6        # kg
    Cr      = 0.035      # rolling resistance coefficient
    Cd      = 0.8        # drag coefficient
    Af      = 0.001      # frontal area (m²)
    rho     = 1.2        # air density (kg/m³)
    Ts      = 0.20       # control period (s)

    # Motor deadband (from calibration)
    MIN_PWM = 95
    MAX_PWM = 255

    # Dual‑aware high‑level MPC (MPC1) weights
# Dual‑aware high‑level MPC (MPC1) weights
    Q_FRONT = 1000.0     # penalty on front distance error
    Q_REAR  = 200.0      # penalty on rear distance error (lower = less aggressive)
    Q_VEL   = 10.0       # NEW: penalty for deviating from the set speed
    R_ACCEL = 50.0       # penalty on acceleration changes      # penalty on acceleration changes

# ----------------------------------------------------------------------------
def motor_k() -> float:
    """Normalised velocity per PWM count."""
    return P.vset / P.MAX_PWM

# ============================================================================
# Hardware Objects
# ============================================================================
if ON_PI:
    front_sensor = DistanceSensor(echo=FRONT_ECHO, trigger=FRONT_TRIG,
                                  pin_factory=factory, max_distance=2.0)
    rear_sensor  = DistanceSensor(echo=REAR_ECHO, trigger=REAR_TRIG,
                                  pin_factory=factory, max_distance=2.0)
    motor_left   = Motor(backward=IN1, forward=IN2, pwm=False, pin_factory=factory)
    motor_right  = Motor(backward=IN3, forward=IN4, pwm=False, pin_factory=factory)
    pwm_ena = PWMOutputDevice(ENA, pin_factory=factory)
    pwm_enb = PWMOutputDevice(ENB, pin_factory=factory)
else:
    front_sensor = rear_sensor = motor_left = motor_right = None
    pwm_ena = pwm_enb = None

# Encoder interrupt
pulse_count = 0

def _encoder_cb(channel):
    global pulse_count
    pulse_count += 1

if ON_PI:
    _GPIO.setmode(_GPIO.BCM)
    _GPIO.setup(ENCODER_PIN, _GPIO.IN, pull_up_down=_GPIO.PUD_UP)
    _GPIO.add_event_detect(ENCODER_PIN, _GPIO.BOTH,
                           callback=_encoder_cb, bouncetime=1)

# ============================================================================
# Motor Control
# ============================================================================
def u_to_pwm(u: float) -> int:
    """Map control [0,1] to PWM value (0 or MIN_PWM..MAX_PWM)."""
    if u < 0.02:
        return 0
    return int(P.MIN_PWM + (P.MAX_PWM - P.MIN_PWM) * min(u, 1.0))

def drive(pwm_out: int):
    """Set both motors forward with given PWM."""
    if not ON_PI:
        return
    mag = pwm_out / 255.0
    pwm_ena.value = mag
    pwm_enb.value = mag
    motor_left.forward()
    motor_right.forward()

def stop_motors():
    if ON_PI:
        motor_left.stop()
        motor_right.stop()
        pwm_ena.off()
        pwm_enb.off()

# ============================================================================
# Velocity Estimator for Front and Rear Vehicles
# ============================================================================
class VehicleVelocityEstimator:
    """Estimate absolute speed of other vehicle from distance derivative."""
    def __init__(self, is_rear=False):
        self._prev_dist = None
        self._est = 0.0
        self.is_rear = is_rear

    def update(self, dist: float, v_ego: float) -> float:
        if self._prev_dist is None:
            self._prev_dist = dist
            return 0.0
        # Relative speed = (dist_change) / dt
        rel_speed = (dist - self._prev_dist) / P.Ts
        if self.is_rear:
            # For rear vehicle: v_rear = v_ego - rel_speed
            raw = max(0.0, v_ego - rel_speed)
        else:
            # For front vehicle: v_front = v_ego + rel_speed
            raw = max(0.0, v_ego + rel_speed)
        self._est = 0.30 * raw + 0.70 * self._est
        self._prev_dist = dist
        return self._est

    def reset(self):
        self._prev_dist = None
        self._est = 0.0
# ============================================================================
# High‑Level MPC (MPC1) – True 3x3 Kinematic State-Space Model
# ============================================================================
class HighLevelMPC:
    """
    Implements the true kinematic MPC1:
    State: x = [df, dr, v_ego]ᵀ
    Control: u = a_e (acceleration)
    Outputs: y = [Δd_f, Δd_r, v_ego]ᵀ
    """
    def __init__(self):
        self.dt = P.Ts
        self.hw = P.hw
        self.Np = P.Np
        self.Nc = P.Nc
        
        self._init = False
        self.x_prev = np.zeros((3, 1))
        self.u_prev = 0.0
        self.g_prev = np.zeros((2, 1))  # Disturbances: [vf, vr]ᵀ
        
        self._build_matrices()

    def _build_matrices(self):
        """Precomputes the LTI Augmented Prediction Matrices for Real-Time Execution."""
        dt = self.dt
        
        # 1. Base State Space: x(k+1) = A x(k) + B u(k) + G g(k)
        # df(k+1) = df(k) - ve*dt + vf*dt
        # dr(k+1) = dr(k) + ve*dt - vr*dt
        # ve(k+1) = ve(k) + a_e*dt
        A = np.array([
            [1.0, 0.0, -dt],
            [0.0, 1.0,  dt],
            [0.0, 0.0, 1.0]
        ])
        B = np.array([[0.0], [0.0], [dt]])
        
        # Output mapping: y = C x. 
        # We output tracking targets: [df - hw*ve, dr - hw*ve, ve]
        C = np.array([
            [1.0, 0.0, -self.hw],
            [0.0, 1.0, -self.hw],
            [0.0, 0.0, 1.0]
        ])
        
        # Disturbance matrix (G): maps vf and vr changes to the gaps
        G_mat = np.array([
            [dt, 0.0],
            [0.0, -dt],
            [0.0, 0.0]
        ])

        nx, nu, ny = 3, 1, 3
        
        # 2. Augmentation for offset-free tracking (integrator)
        A_aug = np.block([
            [A, np.zeros((nx, ny))],
            [C @ A, np.eye(ny)]
        ])
        B_aug = np.block([[B], [C @ B]])
        G_aug = np.block([[G_mat], [C @ G_mat]])
        C_aug = np.block([[np.zeros((ny, nx)), np.eye(ny)]])
        
        # 3. Build Prediction Matrices over horizon (Φ1, Φ2, Φ3)
        self.Phi1 = np.vstack([C_aug @ np.linalg.matrix_power(A_aug, i) for i in range(1, self.Np + 1)])
        
        self.Phi2 = np.zeros((self.Np * ny, self.Nc * nu))
        for i in range(self.Np):
            for j in range(min(i + 1, self.Nc)):
                self.Phi2[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = C_aug @ np.linalg.matrix_power(A_aug, i - j) @ B_aug
                
        self.Phi3 = np.vstack([C_aug @ np.linalg.matrix_power(A_aug, i) @ G_aug for i in range(self.Np)])
        
        # 4. Penalty Weights
        Q_arr = np.zeros((self.Np * ny, self.Np * ny))
        for i in range(self.Np):
            Q_arr[i*3, i*3]     = P.Q_FRONT     # Heavy penalty for front collision
            Q_arr[i*3+1, i*3+1] = P.Q_REAR      # Medium penalty for rear collision
            Q_arr[i*3+2, i*3+2] = P.Q_VEL       # Light penalty for not meeting set speed
        self.Q = Q_arr
        self.R = np.eye(self.Nc) * P.R_ACCEL

    def step(self, df, dr, ve, vf, vr, af=0.0, ar=0.0):
        """Calculates optimal reference speed considering both front and rear."""
        if not self._init:
            self.x_prev = np.array([[df], [dr], [ve]])
            self.g_prev = np.array([[vf], [vr]])
            self.u_prev = 0.0
            self._init = True
            return ve
            
        # 1. Capture Current State & Disturbances
        x_k = np.array([[df], [dr], [ve]])
        g_k = np.array([[vf], [vr]])
        
        # y_k inherently includes the Standstill Distance (sd) error
        y_k = np.array([
            [df - (self.hw * ve + P.sd)],
            [dr - (self.hw * ve + P.sd)],
            [ve]
        ])
        
        dx = x_k - self.x_prev
        dg = g_k - self.g_prev
        
        self.x_prev = x_k
        self.g_prev = g_k
        
        x_aug = np.vstack([dx, y_k])
        
        # 2. Build Reference Trajectory [0 error, 0 error, Target Speed]
        Y_ref = np.zeros((self.Np * 3, 1))
        for i in range(self.Np):
            Y_ref[i*3+2, 0] = P.vset
            
        # 3. Calculate Free Response and Error
        free = self.Phi1 @ x_aug + self.Phi3 @ dg
        E = Y_ref - free
        
        # 4. Quadratic Programming Matrices
        H = 2 * (self.Phi2.T @ self.Q @ self.Phi2 + self.R)
        f = -2 * (self.Phi2.T @ self.Q @ E)
        
        # 5. Physical constraints
        a_max, a_min = 2.0, -3.0     # Max Accel / Decel (m/s^2)
        du_max, du_min = 1.0, -1.0   # Max Jerk per step
        
        I = np.eye(self.Nc)
        M = np.tril(np.ones((self.Nc, self.Nc)))
        
        A_ineq = np.vstack([I, -I, M, -M])
        b_ineq = np.vstack([
            np.full((self.Nc, 1), du_max),
            np.full((self.Nc, 1), -du_min),
            np.full((self.Nc, 1), a_max - self.u_prev),
            np.full((self.Nc, 1), self.u_prev - a_min)
        ])
        
        # 6. Fast Hildreth QP Solver
        H_inv = np.linalg.inv(H)
        M_mat = A_ineq @ H_inv @ A_ineq.T
        K_mat = A_ineq @ H_inv @ f + b_ineq
        
        lam = np.zeros((len(b_ineq), 1))
        for _ in range(100):
            lam_old = lam.copy()
            for i in range(len(b_ineq)):
                w = K_mat[i, 0]
                for j in range(len(b_ineq)):
                    if i != j:
                        w += M_mat[i, j] * lam[j, 0]
                lam[i, 0] = max(0, -w / (M_mat[i, i] + 1e-8))
            if np.linalg.norm(lam - lam_old) < 1e-6:
                break
                
        dU = -H_inv @ (f + A_ineq.T @ lam)
        
        # 7. Apply optimal acceleration to the target velocity
        a_e_opt = self.u_prev + dU[0, 0]
        self.u_prev = np.clip(a_e_opt, a_min, a_max)
        
        v_ref_new = ve + self.u_prev * self.dt
        
        # Return safely clamped velocity command
        return float(np.clip(v_ref_new, 0.0, P.vset))

    def reset(self):
        self._init = False
        self.x_prev = np.zeros((3, 1))
        self.g_prev = np.zeros((2, 1))
        self.u_prev = 0.0
# ============================================================================
# Low‑Level MPC (MPC2) – unchanged from working front‑only version
# ============================================================================
class LowLevelMPC:
    """Tracks reference speed using throttle/brake (0..1)."""
    def __init__(self):
        self.prev_u = 0.0
        self.y_prev = 0.0
        self._init = False
        self._last_pwm = 0

    def _friction_ff(self, ve: float, vref: float) -> float:
        min_ctrl = P.MIN_PWM * motor_k()
        if vref < min_ctrl:
            return 0.0
        F = P.Cr * 9.81 + P.rho * P.Cd * P.Af * ve * ve / (2.0 * P.mass)
        return min(max(F / P.kT, 0.0), 0.5)

    def _u_ss(self, vref: float) -> float:
        val = (vref / motor_k() - P.MIN_PWM) / (P.MAX_PWM - P.MIN_PWM)
        return max(0.0, min(1.0, val))

    def step(self, ve: float, vref: float) -> dict:
        if not self._init:
            self.y_prev = ve
            self._init = True

        if vref < 0.02:
            self.prev_u = max(0.0, self.prev_u + P.DUmin * 0.3)
            self.y_prev = ve
            return {"uTotal": 0.0, "pwm": 0, "u_mpc": self.prev_u, "uff": 0.0}

        Ts = P.Ts
        ve0 = max(ve, 0.05)
        dam = -(P.rho * P.Cd * P.Af * ve0 / P.mass) - P.Cr * 9.81
        a = 1.0 + Ts * dam
        b = Ts * P.kT

        F = [a ** (i+1) for i in range(P.Np)]
        G = [[0.0]*P.Nc for _ in range(P.Np)]
        for i in range(P.Np):
            for j in range(min(i+1, P.Nc)):
                G[i][j] = a ** (i-j) * b

        eps = [vref - F[i] * self.y_prev for i in range(P.Np)]

        H2 = [[0.0]*P.Nc for _ in range(P.Nc)]
        f2 = [0.0]*P.Nc
        for i in range(P.Nc):
            for j in range(P.Nc):
                s = sum(P.Q * G[k][i] * G[k][j] for k in range(P.Np))
                H2[i][j] = 2.0 * (s + (P.R if i==j else 0.0))
            f2[i] = -2.0 * sum(P.Q * G[k][i] * eps[k] for k in range(P.Np))

        if P.Nc == 1:
            du0 = -f2[0] / (H2[0][0] or 1.0)
        else:
            det = H2[0][0]*H2[1][1] - H2[0][1]*H2[1][0]
            du0 = -(H2[1][1]*f2[0] - H2[0][1]*f2[1]) / (det + 1e-8)

        # Adaptive rate limits
        v_err = vref - ve
        under_frac = min(max(0.0, v_err) / max(vref, 0.01), 1.0)
        over_frac  = min(max(0.0, -v_err) / max(ve, 0.01), 1.0)
        adapt_dumax = P.DUmax * max(under_frac / 0.15, 0.10) if under_frac < 0.15 else P.DUmax
        adapt_dumin = P.DUmin * (1.0 + 2.0 * over_frac)
        du0 = np.clip(du0, adapt_dumin, adapt_dumax)
        u_inc = np.clip(self.prev_u + du0, 0.0, 1.0)

        u_ss = self._u_ss(vref)
        uff = self._friction_ff(ve, vref)
        uff_applied = max(0.0, u_ss - u_inc)
        u_total = np.clip(uff_applied + u_inc, 0.0, 1.0)

# Anti‑windup
        err_frac = abs(v_err) / max(abs(vref), 0.01)  # <--- FIX: Moved this line up here
        
        if v_err < -0.01:
            snap = 0.40 + 0.45 * min(err_frac, 1.0)
            u_inc = u_inc * (1-snap) + u_ss * snap
            u_inc = max(-0.05-0.15*min(err_frac,1.0), min(1.0, u_inc))
        elif abs(v_err) < 0.008:
            u_inc = u_inc * 0.92 + u_ss * 0.08
        elif err_frac < 0.12:
            blend = 0.05 + 0.15 * (1.0 - err_frac/0.12)
            u_inc = u_inc * (1-blend) + u_ss * blend

        self.prev_u = np.clip(u_inc, 0.0, 1.0)

        mk = motor_k()
        pwm_raw = u_to_pwm(u_total)
        v_motor = pwm_raw * mk
        v_pred = 0.30 * v_motor + 0.70 * ve
        self.y_prev = 0.55 * v_pred + 0.45 * self.y_prev

        return {"uTotal": u_total, "u_mpc": self.prev_u, "uff": uff, "pwm": pwm_raw}

    def reset(self):
        self.prev_u = 0.0
        self.y_prev = 0.0
        self._init = False
        self._last_pwm = 0

# ============================================================================
# Main Loop
# ============================================================================
def main():
    # ── Setup logging to a local file backup ──
    csv_file = csv_writer = None
    
    # These match the exact 8 inputs and 1 output your ML model needs
    csv_fields = [
        "time", "ve", "vf", "vr", 
        "df", "dr", "d_safe_f", "d_safe_r", "delta_df", 
        "u_total", "pwm"
    ]
    
    if ENABLE_LOCAL_LOGGING:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOG_DIR / "dual_mpc_ml_training_data.csv"
        
        # Check if file exists and has data (so we don't write headers in the middle of the file)
        file_exists = file_path.exists() and file_path.stat().st_size > 0

        # OPEN IN APPEND MODE ("a") INSTEAD OF WRITE ("w")
        csv_file = open(file_path, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)

        # Only write the header row if this is a brand new file
        if not file_exists:
            csv_writer.writeheader()
    global pulse_count

    # Controllers and estimators
    high_mpc = HighLevelMPC()
    low_mpc = LowLevelMPC()
    front_est = VehicleVelocityEstimator(is_rear=False)
    rear_est  = VehicleVelocityEstimator(is_rear=True)

    # State variables
    measured_v = 0.0
    v_smooth = 0.0
    filt_df = 0.0
    filt_dr = 0.0
    vf_prev, vr_prev = 0.0, 0.0
    prev_time = time.time()
    loop_count = 0

    # Shutdown handling
    running = True
    def shutdown(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, shutdown)

    print("Dual‑aware cascaded MPC started")
    print(f"Ts={P.Ts}s, vset={P.vset}m/s, hw={P.hw}s, sd={P.sd}m")
    print("Front sensor: GPIO{FRONT_TRIG}/{FRONT_ECHO}  Rear: GPIO{REAR_TRIG}/{REAR_ECHO}")

    try:
        while running:
            now = time.time()
            dt = max(now - prev_time, 1e-3)

            # --- Read distances -------------------------------------------------
            if ON_PI:
                raw_df = front_sensor.distance
                raw_dr = rear_sensor.distance
            else:
                raw_df, raw_dr = 0.65, 0.80   # mock values

            # Sanitise and filter
            raw_df = 2.0 if (raw_df <= 0.02 or math.isnan(raw_df)) else raw_df
            raw_dr = 2.0 if (raw_dr <= 0.02 or math.isnan(raw_dr)) else raw_dr
            # Spike rejection
            if filt_df > 0 and abs(raw_df - filt_df) > 0.30:
                raw_df = filt_df
            if filt_dr > 0 and abs(raw_dr - filt_dr) > 0.30:
                raw_dr = filt_dr
            # IIR filter (α=0.15)
            filt_df = raw_df if filt_df == 0 else 0.15 * raw_df + 0.85 * filt_df
            filt_dr = raw_dr if filt_dr == 0 else 0.15 * raw_dr + 0.85 * filt_dr

            # --- Encoder velocity ----------------------------------------------
            count = pulse_count
            pulse_count = 0
            if count > 0:
                raw_v = min((count / COUNTS_PER_REV) * WHEEL_CIRCUM / dt, 1.0)
                gap = abs(raw_v - measured_v)
                alpha = 0.70 if gap > 0.08 else (0.55 if gap > 0.03 else 0.25)
                measured_v = alpha * raw_v + (1 - alpha) * measured_v
            else:
                measured_v = measured_v * 0.95 if measured_v > 0.01 else 0.0

            v_smooth = 0.05 * measured_v + 0.95 * v_smooth

            # --- Estimate front and rear speeds --------------------------------
            vf = front_est.update(filt_df, v_smooth)
            vr = rear_est.update(filt_dr, v_smooth)
            # Estimate accelerations (simple difference)
            af = (vf - vf_prev) / dt if loop_count > 0 else 0.0
            ar = (vr - vr_prev) / dt if loop_count > 0 else 0.0
            vf_prev, vr_prev = vf, vr

            # --- Dual‑aware cascaded MPC ---------------------------------------
            vref = high_mpc.step(filt_df, filt_dr, measured_v, vf, vr, af, ar)
            ctrl = low_mpc.step(measured_v, vref)
            pwm_out = ctrl["pwm"]

            # --- Apply motor command -------------------------------------------
            # --- Apply motor command -------------------------------------------
            drive(pwm_out)

            # --- CALCULATE ML FEATURES -----------------------------------------
            # The ML model needs to know the exact safe boundaries the MPC was looking at
            d_safe_f = P.hw * measured_v + P.sd
            d_safe_r = P.hw * vr + P.sd
            delta_df = filt_df - d_safe_f

            # --- CSV LOGGING ---------------------------------------------------
            if csv_writer:
                csv_writer.writerow({
                    "time": f"{now:.3f}", 
                    "ve": f"{measured_v:.4f}", 
                    "vf": f"{vf:.4f}", 
                    "vr": f"{vr:.4f}", 
                    "df": f"{filt_df:.4f}", 
                    "dr": f"{filt_dr:.4f}", 
                    "d_safe_f": f"{d_safe_f:.4f}", 
                    "d_safe_r": f"{d_safe_r:.4f}", 
                    "delta_df": f"{delta_df:.4f}", 
                    "u_total": f"{ctrl['uTotal']:.4f}", # The target for the ML model
                    "pwm": pwm_out
                })

            # --- Console output (every 2 seconds) -----------------------------
            if loop_count % 10 == 0:
                d_safe_f = P.hw * measured_v + P.sd
                d_safe_r = P.hw * vr + P.sd
                print(f"Dist F/R: {filt_df:.2f}/{filt_dr:.2f}  Safe F/R: {d_safe_f:.2f}/{d_safe_r:.2f}  "
                      f"v_ego: {measured_v:.2f}  v_ref: {vref:.2f}  v_f: {vf:.2f}  v_r: {vr:.2f}  "
                      f"PWM: {pwm_out}")

            # --- Timing --------------------------------------------------------
            prev_time = now
            elapsed = time.time() - now
            if elapsed < P.Ts:
                time.sleep(P.Ts - elapsed)
            else:
                print(f"[WARN] Loop overrun {1000*(elapsed-P.Ts):.1f}ms")
            loop_count += 1

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_motors()
        if ON_PI:
            _GPIO.remove_event_detect(ENCODER_PIN)
            _GPIO.cleanup(ENCODER_PIN)
        print("Clean exit")

if __name__ == "__main__":
    main()
