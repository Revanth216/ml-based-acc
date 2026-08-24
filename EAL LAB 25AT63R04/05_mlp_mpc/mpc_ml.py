import math
import signal
import time
import numpy as np
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
TRIG_PIN = 23; ECHO_PIN = 24
IN1_PIN = 5;   IN2_PIN = 6;   ENA_PIN = 27
IN3_PIN = 19;  IN4_PIN = 26;  ENB_PIN = 22
ENCODER_PIN = 16

COUNTS_PER_REV = 12 * 29       
WHEEL_CIRCUM   = math.pi * 0.065   

class P:
    vset    = 0.38
    hw      = 0.70
    sd      = 0.25
    hs      = 0.30
    ah      = 0.30
    Ts      = 0.20
    MIN_PWM = 95
    MAX_PWM = 255

if ON_PI:
    distance_sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN, pin_factory=factory, max_distance=2.0)
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

class MLPController:
    def __init__(self):
        try:
            weights = np.load("mpc_mlp_weights.npz")
            self.W1, self.W2, self.W3 = weights['W1'], weights['W2'], weights['W3']
            self.b1, self.b2, self.b3 = weights['b1'], weights['b2'], weights['b3']
            scaler = np.load("mpc_mlp_scaler.npz")
            self.mean, self.scale = scaler['mean'], scaler['scale']
        except Exception as e:
            print(f"[FATAL] Missing .npz files! Error: {e}")
            exit(1)
        
        self.vref_filt = 0.0
        self._last_pwm = 0
        self._init = False

    def relu(self, x):
        return np.maximum(0, x)
        
    def step(self, filt_dist: float, ve: float, v_leader: float) -> dict:
        vL = max(v_leader, 0.0)
        eqDist = P.hw * vL + P.sd
        effHs = min(P.hs, eqDist - 0.04)
        span = eqDist - effHs
        df = filt_dist - eqDist

        if filt_dist >= eqDist:
            vref_raw = min(P.vset, vL + P.ah * (filt_dist - eqDist))
        elif filt_dist > effHs:
            ratio = max(0.0, min(1.0, (filt_dist - effHs) / span))
            vref_raw = vL * math.sqrt(ratio)
        else:
            vref_raw = 0.0

        is_step_down = vref_raw < self.vref_filt - 0.01
        in_r2 = effHs < filt_dist < eqDist
        alpha_vref = 1.0 if in_r2 else (0.60 if is_step_down else 0.35)
        
        if not self._init:
            self.vref_filt = ve
            self._init = True
            
        self.vref_filt = alpha_vref * vref_raw + (1 - alpha_vref) * self.vref_filt
        vref_smooth = self.vref_filt
        
        distance_error_cm    = df * 100.0
        velocity_error_cmps  = (vref_smooth - ve) * 100.0
        actual_velocity_cmps = ve * 100.0
        leader_velocity_cmps = v_leader * 100.0
        prev_pwm             = float(self._last_pwm)
        
        features = np.array([
            distance_error_cm, 
            velocity_error_cmps, 
            actual_velocity_cmps, 
            leader_velocity_cmps, 
            prev_pwm
        ])
        
        x_scaled = (features - self.mean) / self.scale
        a1 = self.relu(np.dot(x_scaled, self.W1) + self.b1)
        a2 = self.relu(np.dot(a1, self.W2) + self.b2)
        out = np.dot(a2, self.W3) + self.b3
        raw_pwm = out[0]
        
        max_dpwm = 30 if (ve > vL + 0.05) else 8
        pwm_lim = int(min(max(raw_pwm, self._last_pwm - max_dpwm), self._last_pwm + max_dpwm))
        
        if pwm_lim > P.MAX_PWM: pwm_lim = P.MAX_PWM
        if 0 < pwm_lim < P.MIN_PWM: pwm_lim = P.MIN_PWM if self._last_pwm == 0 else 0
        if pwm_lim < 0: pwm_lim = 0
            
        self._last_pwm = pwm_lim
        
        return {
            'pwm': pwm_lim,
            'dist_err': distance_error_cm,
            'vel_err': velocity_error_cmps
        }

#Leader 
class LeaderVelocityEstimator:
    def __init__(self):
        self._prev = None
        self._est  = 0.0

    def update(self, filt_dist: float, v_ego: float) -> float:
        if self._prev is None:
            self._prev = filt_dist
            return 0.0
        raw = max(0.0, (filt_dist - self._prev) / P.Ts + v_ego)
        self._est  = 0.30 * raw + 0.70 * self._est
        self._prev = filt_dist
        return self._est

#Main Loop
def main():
    global pulse_count
    running = [True]
    signal.signal(signal.SIGINT, lambda sig, frame: running.__setitem__(0, False))

    controller = MLPController()
    leader_est = LeaderVelocityEstimator()
    measured_v = 0.0; filtered_d = 0.0; v_smooth = 0.0   
    prev_time = time.time()
    print("Press Ctrl+C to stop.\n")
    print(f"{'Dist(m)':>8} | {'DistErr(cm)':>11} | {'EgoVel(m/s)':>11} | {'VelErr(cm/s)':>12} | {'LeadVel(m/s)':>12} | {'MLP_PWM':>7}")
    print("-" * 75)

    try:
        while running[0]:
            now = time.time()
            dt  = max(now - prev_time, 1e-3)
            raw_d = distance_sensor.distance if ON_PI else 0.65   
            if raw_d <= 0.02 or raw_d >= 2.0 or math.isnan(raw_d): raw_d = 2.0
            if filtered_d > 0.0 and abs(raw_d - filtered_d) > 0.30: raw_d = filtered_d
            filtered_d = (raw_d if filtered_d == 0.0 else 0.15 * raw_d + 0.85 * filtered_d)

            # 2. Read Velocity
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
            v_leader = leader_est.update(filtered_d, v_smooth)
            ctrl = controller.step(filtered_d, measured_v, v_leader)
            drive(ctrl['pwm'])
            print(f"{filtered_d:8.2f} | {ctrl['dist_err']:11.1f} | {measured_v:11.2f} | {ctrl['vel_err']:12.1f} | {v_leader:12.2f} | {ctrl['pwm']:7}")
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