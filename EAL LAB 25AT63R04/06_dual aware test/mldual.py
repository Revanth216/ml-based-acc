#!/usr/bin/env python3
import math
import signal
import time
import numpy as np

# ── Hardware Libraries ─────────────────────────────────────────────────────────
try:
    from gpiozero import DistanceSensor, Motor, PWMOutputDevice
    from gpiozero.pins.pigpio import PiGPIOFactory
    import RPi.GPIO as _GPIO
    ON_PI = True
    factory = PiGPIOFactory()
except Exception:
    ON_PI = False
    factory = None
    _GPIO = None
    print("[WARN] Hardware libraries not found. Running in mock mode.")

# ── Pins ───────────────────────────────────────────────────────────────────────
FRONT_TRIG = 23;  FRONT_ECHO = 24
REAR_TRIG  = 20;  REAR_ECHO  = 21
IN1_PIN = 5;      IN2_PIN = 6;     ENA_PIN = 27
IN3_PIN = 19;     IN4_PIN = 26;    ENB_PIN = 22
ENCODER_PIN = 16

COUNTS_PER_REV = 12 * 29       
WHEEL_CIRCUM   = math.pi * 0.065   

class P:
    vset    = 0.38
    hw      = 0.70
    sd      = 0.25
    Ts      = 0.20
    MIN_PWM = 95
    MAX_PWM = 255

# ── Hardware Init ──────────────────────────────────────────────────────────────
if ON_PI:
    front_sensor = DistanceSensor(echo=FRONT_ECHO, trigger=FRONT_TRIG, pin_factory=factory, max_distance=2.0)
    rear_sensor  = DistanceSensor(echo=REAR_ECHO, trigger=REAR_TRIG, pin_factory=factory, max_distance=2.0)
    motor_left  = Motor(backward=IN1_PIN, forward=IN2_PIN, pwm=False, pin_factory=factory)
    motor_right = Motor(backward=IN3_PIN, forward=IN4_PIN, pwm=False, pin_factory=factory)
    pwm_ena = PWMOutputDevice(ENA_PIN, pin_factory=factory)
    pwm_enb = PWMOutputDevice(ENB_PIN, pin_factory=factory)

pulse_count = 0
def _encoder_cb(channel):
    global pulse_count
    pulse_count += 1

if ON_PI:
    _GPIO.setmode(_GPIO.BCM)
    _GPIO.setup(ENCODER_PIN, _GPIO.IN, pull_up_down=_GPIO.PUD_UP)
    _GPIO.add_event_detect(ENCODER_PIN, _GPIO.BOTH, callback=_encoder_cb, bouncetime=1)

def drive(pwm_out: int):
    if not ON_PI: return
    mag = pwm_out / 255.0
    pwm_ena.value = mag; pwm_enb.value = mag
    motor_left.forward(); motor_right.forward()

def stop_motors():
    if not ON_PI: return
    motor_left.stop(); motor_right.stop()
    pwm_ena.off(); pwm_enb.off()

# ── Velocity Estimator (Handles both Front & Rear) ─────────────────────────────
class VehicleVelocityEstimator:
    def __init__(self, is_rear=False):
        self._prev = None
        self._est  = 0.0
        self.is_rear = is_rear

    def update(self, filt_dist: float, v_ego: float) -> float:
        if self._prev is None:
            self._prev = filt_dist
            return 0.0
            
        if not self.is_rear:
            raw = max(0.0, (filt_dist - self._prev) / P.Ts + v_ego)
        else:
            raw = max(0.0, v_ego - (filt_dist - self._prev) / P.Ts)
            
        self._est  = 0.30 * raw + 0.70 * self._est
        self._prev = filt_dist
        return self._est

# ── Dual-Aware Neural Network Controller ───────────────────────────────────────
class DualAwareMLPController:
    def __init__(self):
        try:
            # Load the unified .npz file
            model_data = np.load("dual_aware_weights.npz")
            self.W1, self.W2, self.W3 = model_data['W1'], model_data['W2'], model_data['W3']
            self.b1, self.b2, self.b3 = model_data['b1'], model_data['b2'], model_data['b3']
            self.mean, self.scale = model_data['mean'], model_data['scale']
        except Exception as e:
            print(f"[FATAL] Missing 'dual_aware_weights.npz'! Error: {e}")
            exit(1)
        
        self._last_pwm = 0

    def relu(self, x):
        return np.maximum(0, x)
        
    def u_to_pwm(self, u: float) -> int:
        if u < 0.02: return 0
        return int(P.MIN_PWM + (P.MAX_PWM - P.MIN_PWM) * min(u, 1.0))

    def step(self, df: float, dr: float, ve: float, vf: float, vr: float) -> dict:
        # 1. Calculate derived features
        d_safe_f = P.hw * ve + P.sd
        d_safe_r = P.hw * vr + P.sd
        delta_df = df - d_safe_f
        
        # 2. Package 8 features
        raw_state = np.array([ve, vf, vr, df, dr, d_safe_f, d_safe_r, delta_df])
        
        # 3. Standard Scaler
        scaled_state = (raw_state - self.mean) / self.scale
        
        # 4. Neural Network Forward Pass (8 -> 64 -> 32 -> 1)
        a1 = self.relu(np.dot(scaled_state, self.W1) + self.b1)
        a2 = self.relu(np.dot(a1, self.W2) + self.b2)
        out = np.dot(a2, self.W3) + self.b3
        
        # Clamp u_total to physical limits [0.0, 1.0]
        u_pred = max(0.0, min(1.0, float(out[0])))
        
        # 5. Convert predicted u_total to PWM
        target_pwm = self.u_to_pwm(u_pred)
        
        # 6. Smooth the PWM to prevent motor voltage spikes
        max_dpwm = 30 if (ve > vf + 0.05) else 8
        pwm_lim = int(min(max(target_pwm, self._last_pwm - max_dpwm), self._last_pwm + max_dpwm))
        
        # Clamp PWM safely
        if pwm_lim > P.MAX_PWM: pwm_lim = P.MAX_PWM
        if 0 < pwm_lim < P.MIN_PWM: pwm_lim = P.MIN_PWM if self._last_pwm == 0 else 0
        if pwm_lim < 0: pwm_lim = 0
            
        self._last_pwm = pwm_lim
        
        return {
            'pwm': pwm_lim,
            'u_pred': u_pred
        }

# ── Main Loop ──────────────────────────────────────────────────────────────────
def main():
    global pulse_count
    running = [True]
    signal.signal(signal.SIGINT, lambda sig, frame: running.__setitem__(0, False))

    controller = DualAwareMLPController()
    front_est = VehicleVelocityEstimator(is_rear=False)
    rear_est  = VehicleVelocityEstimator(is_rear=True)
    
    measured_v = 0.0; filt_df = 0.0; filt_dr = 0.0; v_smooth = 0.0   
    prev_time = time.time()
    
    print("Dual-Aware AI Control Started. Press Ctrl+C to stop.\n")
    print(f"{'Dist F/R(m)':>12} | {'Vel Ego(m/s)':>12} | {'Vel F/R(m/s)':>14} | {'AI u_out':>8} | {'PWM':>5}")
    print("-" * 65)

    try:
        while running[0]:
            now = time.time()
            dt  = max(now - prev_time, 1e-3)
            
            # 1. Read & Filter Distances
            raw_df = front_sensor.distance if ON_PI else 0.65
            raw_dr = rear_sensor.distance if ON_PI else 0.80
            
            if raw_df <= 0.02 or raw_df >= 2.0 or math.isnan(raw_df): raw_df = 2.0
            if raw_dr <= 0.02 or raw_dr >= 2.0 or math.isnan(raw_dr): raw_dr = 2.0
            
            if filt_df > 0.0 and abs(raw_df - filt_df) > 0.30: raw_df = filt_df
            if filt_dr > 0.0 and abs(raw_dr - filt_dr) > 0.30: raw_dr = filt_dr
            
            filt_df = (raw_df if filt_df == 0.0 else 0.15 * raw_df + 0.85 * filt_df)
            filt_dr = (raw_dr if filt_dr == 0.0 else 0.15 * raw_dr + 0.85 * filt_dr)

            # 2. Read Ego Velocity
            count = pulse_count
            pulse_count = 0
            if count > 0:
                raw_v = min((count / COUNTS_PER_REV) * WHEEL_CIRCUM / dt, 1.0)
                gap = abs(raw_v - measured_v)
                alpha = 0.70 if gap > 0.08 else (0.55 if gap > 0.03 else 0.25)
                measured_v = alpha * raw_v + (1 - alpha) * measured_v
            else:
                measured_v = (measured_v * 0.95 if measured_v > 0.01 else 0.0)
            
            v_smooth = 0.05 * measured_v + 0.95 * v_smooth
            
            # 3. Estimate Leader and Rear Velocities
            vf = front_est.update(filt_df, v_smooth)
            vr = rear_est.update(filt_dr, v_smooth)
            
            # 4. Neural Network Inference
            ctrl = controller.step(filt_df, filt_dr, measured_v, vf, vr)
            
            # 5. Apply Motor Output
            drive(ctrl['pwm'])
            
            # 6. Terminal Output
            print(f"{filt_df:5.2f}/{filt_dr:<5.2f} | {measured_v:12.2f} | {vf:5.2f}/{vr:<6.2f} | {ctrl['u_pred']:8.3f} | {ctrl['pwm']:5}")
            
            # Timing Governance
            prev_time = now
            elapsed = time.time() - now
            if elapsed < P.Ts:
                time.sleep(P.Ts - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        stop_motors()
        if ON_PI:
            _GPIO.remove_event_detect(ENCODER_PIN)
            _GPIO.cleanup(ENCODER_PIN)
        print("\nShutdown complete.")

if __name__ == "__main__":
    main()