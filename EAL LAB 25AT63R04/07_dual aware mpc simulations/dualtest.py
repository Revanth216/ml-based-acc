#!/usr/bin/env python3
"""
Enhanced Dual-Aware Cascaded MPC for ACC
Incorporates theoretical kinematics from Sharma et al. with real-world embedded hardware constraints.
"""

import math
import time
import signal
import csv
from pathlib import Path
import numpy as np

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
    print("[WARN] gpiozero/pigpio not available – running in simulation/mock mode")

# ============================================================================
# Hardware Pins & Logging Setup
# ============================================================================
ENABLE_LOCAL_LOGGING = True
LOG_DIR = Path.home() / "revanthdualaware"
FRONT_TRIG, FRONT_ECHO = 23, 24
REAR_TRIG, REAR_ECHO   = 20, 21
IN1, IN2, IN3, IN4     = 5, 6, 19, 26
ENA, ENB               = 27, 22
ENCODER_PIN            = 16

COUNTS_PER_REV = 12 * 29          
WHEEL_CIRCUM   = math.pi * 0.065  

# ============================================================================
# System & MPC Parameters
# ============================================================================
class P:
    v_phys_max = 0.68
    Np      = 10
    Nc      = 3
    Ts      = 0.20       # Control period (s)
    
    # Velocity and Headway Policies
    vset    = 0.38       # Desired cruise speed (m/s)
    hw      = 0.70       # Time headway (s)
    sd      = 0.25       # Standstill distance (m)
    
    # Low-Level MPC (MPC2)
    Q_LOW   = 600.0      
    R_LOW   = 20.0       
    DUmax   = 0.05
    DUmin   = -0.15
    kT      = 1.0745
    MIN_PWM, MAX_PWM = 95, 255

    # High-Level MPC (MPC1) - Strict Priority Weighting
    Q_FRONT_BASE = 1000.0     
    Q_REAR_BASE  = 200.0      
    Q_VEL   = 10.0       
    R_ACCEL = 50.0       

    # Vehicle Physics (Bicycle Model Constants)
    mass, Cr, Cd, Af, rho = 0.6, 0.035, 0.8, 0.001, 1.2

def motor_k() -> float:
    return P.v_phys_max / P.MAX_PWM

# ============================================================================
# Hardware Initialization
# ============================================================================
pulse_count = 0
def _encoder_cb(channel):
    global pulse_count
    pulse_count += 1

if ON_PI:
    front_sensor = DistanceSensor(echo=FRONT_ECHO, trigger=FRONT_TRIG, pin_factory=factory, max_distance=2.0)
    rear_sensor  = DistanceSensor(echo=REAR_ECHO, trigger=REAR_TRIG, pin_factory=factory, max_distance=2.0)
    motor_left   = Motor(backward=IN1, forward=IN2, pwm=False, pin_factory=factory)
    motor_right  = Motor(backward=IN3, forward=IN4, pwm=False, pin_factory=factory)
    pwm_ena, pwm_enb = PWMOutputDevice(ENA, pin_factory=factory), PWMOutputDevice(ENB, pin_factory=factory)
    
    _GPIO.setmode(_GPIO.BCM)
    _GPIO.setup(ENCODER_PIN, _GPIO.IN, pull_up_down=_GPIO.PUD_UP)
    _GPIO.add_event_detect(ENCODER_PIN, _GPIO.BOTH, callback=_encoder_cb, bouncetime=1)
else:
    front_sensor = rear_sensor = motor_left = motor_right = pwm_ena = pwm_enb = None

def u_to_pwm(u: float) -> int:
    if u < 0.02: return 0
    return int(P.MIN_PWM + (P.MAX_PWM - P.MIN_PWM) * min(u, 1.0))

def drive(pwm_out: int):
    if not ON_PI: return
    mag = pwm_out / 255.0
    pwm_ena.value = pwm_enb.value = mag
    motor_left.forward()
    motor_right.forward()

def stop_motors():
    if ON_PI:
        motor_left.stop()
        motor_right.stop()
        pwm_ena.off()
        pwm_enb.off()

# ============================================================================
# Estimators
# ============================================================================
class VehicleVelocityEstimator:
    def __init__(self, is_rear=False):
        self._prev_dist = None
        self._est = 0.0
        self.is_rear = is_rear

    def update(self, dist: float, v_ego: float) -> float:
        if self._prev_dist is None:
            self._prev_dist = dist
            return 0.0
        rel_speed = (dist - self._prev_dist) / P.Ts
        raw = max(0.0, v_ego - rel_speed) if self.is_rear else max(0.0, v_ego + rel_speed)
        self._est = 0.30 * raw + 0.70 * self._est
        self._prev_dist = dist
        return self._est

# ============================================================================
# High-Level MPC (MPC1) - Refined with Slack Variable Logic
# ============================================================================
class HighLevelMPC:
    def __init__(self):
        self.dt = P.Ts
        self.hw = P.hw
        self.Np, self.Nc = P.Np, P.Nc
        self._init = False
        self._build_matrices()

    def _build_matrices(self):
        dt = self.dt
        # LTI State Space Model based on relative kinematics
        A = np.array([[1.0, 0.0, -dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        B = np.array([[0.0], [0.0], [dt]])
        C = np.array([[1.0, 0.0, -self.hw], [0.0, 1.0, -self.hw], [0.0, 0.0, 1.0]])
        G_mat = np.array([[dt, 0.0], [0.0, -dt], [0.0, 0.0]])

        nx, nu, ny = 3, 1, 3
        A_aug = np.block([[A, np.zeros((nx, ny))], [C @ A, np.eye(ny)]])
        B_aug = np.block([[B], [C @ B]])
        G_aug = np.block([[G_mat], [C @ G_mat]])
        C_aug = np.block([[np.zeros((ny, nx)), np.eye(ny)]])
        
        self.Phi1 = np.vstack([C_aug @ np.linalg.matrix_power(A_aug, i) for i in range(1, self.Np + 1)])
        self.Phi2 = np.zeros((self.Np * ny, self.Nc * nu))
        for i in range(self.Np):
            for j in range(min(i + 1, self.Nc)):
                self.Phi2[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = C_aug @ np.linalg.matrix_power(A_aug, i - j) @ B_aug
        self.Phi3 = np.vstack([C_aug @ np.linalg.matrix_power(A_aug, i) @ G_aug for i in range(self.Np)])

    def step(self, df, dr, ve, vf, vr):
        if not self._init:
            self.x_prev = np.array([[df], [dr], [ve]])
            self.g_prev = np.array([[vf], [vr]])
            self.u_prev = 0.0
            self._init = True
            return ve
            
        x_k = np.array([[df], [dr], [ve]])
        g_k = np.array([[vf], [vr]])
        
        # FIX 1: Use vr for rear standstill distance requirement
        y_k = np.array([
            [df - (self.hw * ve + P.sd)],
            [dr - (self.hw * vr + P.sd)], # <-- Changed ve to vr
            [ve]
        ])
        
        dx = x_k - self.x_prev
        dg = g_k - self.g_prev
        self.x_prev, self.g_prev = x_k, g_k
        x_aug = np.vstack([dx, y_k])
        
        Y_ref = np.zeros((self.Np * 3, 1))
        for i in range(self.Np): Y_ref[i*3+2, 0] = P.vset
            
        free = self.Phi1 @ x_aug + self.Phi3 @ dg
        E = Y_ref - free

        # Dynamic Prioritization (Slack proxy mechanism)
# Dynamic Prioritization (Slack proxy mechanism)
        Q_arr = np.zeros((self.Np * 3, self.Np * 3))
        for i in range(self.Np):
            
            # FRONT BUMPER: 
            # If we have > 15cm of safe buffer, ignore the front car.
            # If buffer is shrinking (0 to 15cm), apply normal penalty to brake smoothly.
            # If unsafe (< 0cm), double the penalty for emergency braking.
            if y_k[0,0] > 0.15:
                front_penalty = 0.0
            elif y_k[0,0] > 0.0:
                front_penalty = P.Q_FRONT_BASE
            else:
                front_penalty = P.Q_FRONT_BASE * 2.0
                
            # REAR BUMPER: 
            # NEVER penalize pulling away from the rear car. 
            # Only apply penalty if the rear car is an active threat (< 5cm buffer).
            if y_k[1,0] > 0.05:
                rear_penalty = 0.0
            elif y_k[1,0] > 0.0:
                rear_penalty = P.Q_REAR_BASE
            else:
                rear_penalty = P.Q_REAR_BASE * 2.0
            
            # Apply to Q Matrix
            Q_arr[i*3, i*3]     = front_penalty     
            Q_arr[i*3+1, i*3+1] = rear_penalty      
            Q_arr[i*3+2, i*3+2] = P.Q_VEL      
        
        H = 2 * (self.Phi2.T @ Q_arr @ self.Phi2 + np.eye(self.Nc) * P.R_ACCEL)
        f = -2 * (self.Phi2.T @ Q_arr @ E)
        
        # Acceleration Constraints
        a_max, a_min = 2.0, -3.0     
        du_max, du_min = 1.0, -1.0   
        I = np.eye(self.Nc)
        M = np.tril(np.ones((self.Nc, self.Nc)))
        
        A_ineq = np.vstack([I, -I, M, -M])
        b_ineq = np.vstack([
            np.full((self.Nc, 1), du_max),
            np.full((self.Nc, 1), -du_min),
            np.full((self.Nc, 1), a_max - self.u_prev),
            np.full((self.Nc, 1), self.u_prev - a_min)
        ])
        
        # Hildreth QP Solver
        H_inv = np.linalg.inv(H)
        M_mat = A_ineq @ H_inv @ A_ineq.T
        K_mat = A_ineq @ H_inv @ f + b_ineq
        
        lam = np.zeros((len(b_ineq), 1))
        for _ in range(100):
            lam_old = lam.copy()
            for i in range(len(b_ineq)):
                w = K_mat[i, 0] + sum(M_mat[i, j] * lam[j, 0] for j in range(len(b_ineq)) if i != j)
                lam[i, 0] = max(0, -w / (M_mat[i, i] + 1e-8))
            if np.linalg.norm(lam - lam_old) < 1e-6: break
                
        dU = -H_inv @ (f + A_ineq.T @ lam)
        self.u_prev = np.clip(self.u_prev + dU[0, 0], a_min, a_max)
        
        # --- NEW FIX: Anti-Windup & Dynamic Speed Ceiling ---
        # Check if the rear buffer has dropped into the danger zone (< 5cm)
        emergency_escape = True if y_k[1,0] < 0.05 else False
        
        # Anti-Windup: Bleed off lingering acceleration if we are safely cruising near vset
        if not emergency_escape and ve >= P.vset - 0.02:
            self.u_prev *= 0.50 
            
        # Dynamic Ceiling: Lock to vset normally, unlock to v_phys_max if fleeing
        current_vmax = P.v_phys_max if emergency_escape else P.vset
        
        return float(np.clip(ve + self.u_prev * self.dt, 0.0, current_vmax))
# ============================================================================
# Low-Level MPC (MPC2)
# ============================================================================
class LowLevelMPC:
    def __init__(self):
        self.prev_u = 0.0
        self.y_prev = 0.0
        self._init = False

    def _friction_ff(self, ve: float, vref: float) -> float:
        if vref < P.MIN_PWM * motor_k(): return 0.0
        F = P.Cr * 9.81 + P.rho * P.Cd * P.Af * ve * ve / (2.0 * P.mass)
        return min(max(F / P.kT, 0.0), 0.5)

    def _u_ss(self, vref: float) -> float:
        return max(0.0, min(1.0, (vref / motor_k() - P.MIN_PWM) / (P.MAX_PWM - P.MIN_PWM)))

    def step(self, ve: float, vref: float) -> dict:
        if not self._init:
            self.y_prev = ve
            self._init = True

        if vref < 0.02:
            self.prev_u = max(0.0, self.prev_u + P.DUmin * 0.3)
            self.y_prev = ve
            return {"uTotal": 0.0, "pwm": 0}

        Ts, ve0 = P.Ts, max(ve, 0.05)
        a = 1.0 + Ts * (-(P.rho * P.Cd * P.Af * ve0 / P.mass) - P.Cr * 9.81)
        b = Ts * P.kT

        F_vec = [a ** (i+1) for i in range(P.Np)]
        G_mat = [[(a ** (i-j) * b if j <= i else 0.0) for j in range(P.Nc)] for i in range(P.Np)]
        eps = [vref - F_vec[i] * self.y_prev for i in range(P.Np)]

        H2 = [[2.0 * (sum(P.Q_LOW * G_mat[k][i] * G_mat[k][j] for k in range(P.Np)) + (P.R_LOW if i==j else 0.0)) 
               for j in range(P.Nc)] for i in range(P.Nc)]
        f2 = [-2.0 * sum(P.Q_LOW * G_mat[k][i] * eps[k] for k in range(P.Np)) for i in range(P.Nc)]

        # Simplified unconstrained step + clipping for Low-Level
        if P.Nc == 1: du0 = -f2[0] / (H2[0][0] or 1.0)
        else:
            det = H2[0][0]*H2[1][1] - H2[0][1]*H2[1][0]
            du0 = -(H2[1][1]*f2[0] - H2[0][1]*f2[1]) / (det + 1e-8)

        v_err = vref - ve
        under_frac = min(max(0.0, v_err) / max(vref, 0.01), 1.0)
        over_frac  = min(max(0.0, -v_err) / max(ve, 0.01), 1.0)
        
        du0 = np.clip(du0, P.DUmin * (1.0 + 2.0 * over_frac), 
                      P.DUmax * max(under_frac / 0.15, 0.10) if under_frac < 0.15 else P.DUmax)
        u_inc = np.clip(self.prev_u + du0, 0.0, 1.0)

        u_ss, uff = self._u_ss(vref), self._friction_ff(ve, vref)
        u_total = np.clip(max(0.0, u_ss - u_inc) + u_inc, 0.0, 1.0)

        # Anti-windup
        err_frac = abs(v_err) / max(abs(vref), 0.01)
        if v_err < -0.01:
            u_inc = np.clip(u_inc * (1 - (0.40 + 0.45 * min(err_frac, 1.0))) + u_ss * (0.40 + 0.45 * min(err_frac, 1.0)), 
                            -0.05-0.15*min(err_frac,1.0), 1.0)
        elif abs(v_err) < 0.008:
            u_inc = u_inc * 0.92 + u_ss * 0.08

        self.prev_u = np.clip(u_inc, 0.0, 1.0)
        pwm_raw = u_to_pwm(u_total)
        self.y_prev = 0.55 * (0.30 * (pwm_raw * motor_k()) + 0.70 * ve) + 0.45 * self.y_prev

        return {"uTotal": u_total, "pwm": pwm_raw}

# ============================================================================
# Main Loop execution
# ============================================================================
def main():
    global pulse_count
    
    csv_file, csv_writer = None, None
    if ENABLE_LOCAL_LOGGING:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOG_DIR / "dual_mpc_ml_training_data.csv"
        file_exists = file_path.exists() and file_path.stat().st_size > 0
        csv_file = open(file_path, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["time", "ve", "vf", "vr", "df", "dr", "d_safe_f", "d_safe_r", "delta_df", "u_total", "pwm"])
        if not file_exists: csv_writer.writeheader()

    high_mpc, low_mpc = HighLevelMPC(), LowLevelMPC()
    front_est, rear_est = VehicleVelocityEstimator(is_rear=False), VehicleVelocityEstimator(is_rear=True)

    measured_v = v_smooth = filt_df = filt_dr = 0.0
    prev_time = time.time()
    loop_count = 0
    running = True

    def shutdown(sig, frame): nonlocal running; running = False
    signal.signal(signal.SIGINT, shutdown)

    print("Executing Hybrid Dual-Aware MPC...")
    try:
        while running:
            now = time.time()
            dt = max(now - prev_time, 1e-3)

            # Sensor reading & IIR filtering
            raw_df = front_sensor.distance if ON_PI else 0.65
            raw_dr = rear_sensor.distance if ON_PI else 0.80
            raw_df = 2.0 if (raw_df <= 0.02 or math.isnan(raw_df)) else raw_df
            raw_dr = 2.0 if (raw_dr <= 0.02 or math.isnan(raw_dr)) else raw_dr
            
            filt_df = raw_df if filt_df == 0 else (filt_df if abs(raw_df - filt_df) > 0.30 else 0.15 * raw_df + 0.85 * filt_df)
            filt_dr = raw_dr if filt_dr == 0 else (filt_dr if abs(raw_dr - filt_dr) > 0.30 else 0.15 * raw_dr + 0.85 * filt_dr)

            # Velocity Estimation
            if pulse_count > 0:
                raw_v = min((pulse_count / COUNTS_PER_REV) * WHEEL_CIRCUM / dt, 1.0)
                pulse_count = 0
                alpha = 0.70 if abs(raw_v - measured_v) > 0.08 else 0.25
                measured_v = alpha * raw_v + (1 - alpha) * measured_v
            else:
                measured_v = measured_v * 0.95 if measured_v > 0.01 else 0.0

            v_smooth = 0.05 * measured_v + 0.95 * v_smooth
            vf = front_est.update(filt_df, v_smooth)
            vr = rear_est.update(filt_dr, v_smooth)

            # Cascaded MPC execution
            vref = high_mpc.step(filt_df, filt_dr, measured_v, vf, vr)
            ctrl = low_mpc.step(measured_v, vref)
            drive(ctrl["pwm"])

            d_safe_f = P.hw * measured_v + P.sd
            d_safe_r = P.hw * vr + P.sd

            if csv_writer:
                csv_writer.writerow({"time": f"{now:.3f}", "ve": f"{measured_v:.4f}", "vf": f"{vf:.4f}", "vr": f"{vr:.4f}", 
                                     "df": f"{filt_df:.4f}", "dr": f"{filt_dr:.4f}", "d_safe_f": f"{d_safe_f:.4f}", 
                                     "d_safe_r": f"{d_safe_r:.4f}", "delta_df": f"{filt_df - d_safe_f:.4f}", 
                                     "u_total": f"{ctrl['uTotal']:.4f}", "pwm": ctrl["pwm"]})

            if loop_count % 10 == 0:
                print(f"Dist F/R: {filt_df:.2f}/{filt_dr:.2f} | Safe F/R: {d_safe_f:.2f}/{d_safe_r:.2f} | v_ego: {measured_v:.2f} | v_ref: {vref:.2f} | PWM: {ctrl['pwm']}")

            prev_time = now
            elapsed = time.time() - now
            if elapsed < P.Ts: time.sleep(P.Ts - elapsed)
            loop_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        stop_motors()
        if ON_PI:
            _GPIO.remove_event_detect(ENCODER_PIN)
            _GPIO.cleanup()

if __name__ == "__main__":
    main()
