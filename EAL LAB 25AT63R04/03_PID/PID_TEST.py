import math
import csv
import os
from time import sleep, time

from gpiozero import Button, DistanceSensor, Motor, PWMOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory

# ================= GPIO SETUP =================
factory = PiGPIOFactory()

# Pin numbers 
TRIG_PIN    = 23
ECHO_PIN    = 24
IN1_PIN     = 5      # motor_left  backward
IN2_PIN     = 6      # motor_left  forward
IN3_PIN     = 19     # motor_right backward
IN4_PIN     = 26     # motor_right forward
ENA_PIN     = 27
ENB_PIN     = 22
ENCODER_PIN = 16

# ================= VELOCITY CONSTANTS =================
COUNTS_PER_REV = 12 * 29          # 348 counts per revolution
WHEEL_CIRCUM   = math.pi * 0.065  # metres (65 mm diameter wheel)
SAMPLE_INTERVAL = 0.15            # seconds 

# ================= PID PARAMETERS =================
Kp = 210   
Ki = 390    
Kd = 5.5     

# Velocities and distances in SI units (m/s and metres)
setVelocity          = 0.35        # m/s  (≈ 25 cm/s)
headwayTime          = 0.5         # s
standstillDistance   = 0.15        # m    (15 cm)
pwmStepLimit         = 255  
startPWM             = 90   
runPWM               = 60   
deadbandError        = 0.0         # m/s  
maxTargetVelocityDecrease = 0.03   # m/s per step  
maxIntegral          = 2.0         # m/s·s  
MAX_ACCEL            = 2.0         # m/s²  

WMA_ALPHA_VELOCITY   = 0.25        # WMA weight for velocity  
WMA_ALPHA_DISTANCE   = 0.15        # WMA weight for distance

# ================= CSV CONFIGURATION =================
CSV_FILENAME = "telemetry_data.csv"
CSV_HEADERS = [
    "Time(s)", "FilteredDistance(m)", "SafeDistance(m)", "DistanceError(m)",
    "TargetVelocity(m/s)", "MeasuredVelocity(m/s)", "VelocityError(m/s)",
    "PIDOutput", "PWMOutput"
]

# ================= GPIO RESET (prevents PinSetInput on dirty restart) =================
import pigpio as _pigpio
_pi = _pigpio.pi()
if _pi.connected:
    for _pin in [IN1_PIN, IN2_PIN, IN3_PIN, IN4_PIN, ENA_PIN, ENB_PIN]:
        _pi.set_mode(_pin, _pigpio.OUTPUT)
        _pi.write(_pin, 0)
    _pi.stop()

# ================= HARDWARE OBJECTS =================
distance_sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN,
                                 pin_factory=factory, max_distance=4.0)

motor_left  = Motor(backward=IN1_PIN, forward=IN2_PIN,
                    pwm=False, pin_factory=factory)
motor_right = Motor(backward=IN3_PIN, forward=IN4_PIN,
                    pwm=False, pin_factory=factory)

pwm_ena = PWMOutputDevice(ENA_PIN, pin_factory=factory)
pwm_enb = PWMOutputDevice(ENB_PIN, pin_factory=factory)

# Encoder
encoder = Button(ENCODER_PIN, pull_up=False, pin_factory=factory)

# ================= STATE VARIABLES =================
pulse_count          = 0
measuredVelocity     = 0.0   # m/s
filteredDistance     = None  # m
prevError            = 0.0
integral             = 0.0
lastPwmOutput        = 0
lastTargetVelocity   = setVelocity
filteredSafeDistance = None            # WMA-filtered safe distance


def encoder_callback():
    global pulse_count
    pulse_count += 1


encoder.when_pressed = encoder_callback


# ================= HELPERS =================
def wma_filter(prev, new_sample, alpha):
    """Weighted moving average"""
    if prev == 0:
        return new_sample
    return alpha * new_sample + (1 - alpha) * prev


# ================= MAIN LOOP =================
def main():
    global pulse_count, measuredVelocity, prevError, integral, \
        lastPwmOutput, lastTargetVelocity, filteredDistance, setVelocity, \
        filteredSafeDistance

    telemetry_log = []
    
    # We use time.time() to keep track of absolute time across multiple runs
    # Alternatively, you can reset this to 0.0 on every run depending on your preference.
    run_start_time = time()

    print("Robot started. Press Ctrl+C to stop and save data.")

    try:
        while True:
            loop_start = time()

            # ── Distance reading (WMA) ────────────
            raw_d = distance_sensor.distance  # already in metres
            if raw_d <= 0.02 or raw_d >= 4.0 or math.isnan(raw_d):
                raw_d = 4.0
                sensor_valid = False
            else:
                sensor_valid = True

            if filteredDistance is None:
                filteredDistance = raw_d
            else:
                # Spike rejection 
                if sensor_valid and abs(raw_d - filteredDistance) > 0.3:
                    raw_d = filteredDistance
                filteredDistance = wma_filter(filteredDistance, raw_d, WMA_ALPHA_DISTANCE)

            # ── Encoder → velocity ──────────
            count = pulse_count
            pulse_count = 0
            if count > 0:
                raw_v = (count / COUNTS_PER_REV) * WHEEL_CIRCUM / SAMPLE_INTERVAL
                if raw_v > 2.0:          # clamp spurious readings
                    raw_v = measuredVelocity
                measuredVelocity = wma_filter(measuredVelocity, raw_v, WMA_ALPHA_VELOCITY)
            else:
                measuredVelocity = measuredVelocity * 0.95 if measuredVelocity > 0.001 else 0.0

            # ── Safe distance & target velocity ────────────────────────────
            rawSafeDistance = standstillDistance + headwayTime * measuredVelocity
            
            if filteredSafeDistance is None:
                filteredSafeDistance = rawSafeDistance
            else:
                filteredSafeDistance = wma_filter(filteredSafeDistance, rawSafeDistance, 0.25)
            
            safeDistance  = filteredSafeDistance
            distanceError = filteredDistance - safeDistance   

            v_sq = measuredVelocity ** 2 + 2 * MAX_ACCEL * distanceError
            targetVelocity = math.sqrt(max(v_sq, 0.0))
            targetVelocity = min(targetVelocity, setVelocity)

            if filteredDistance <= standstillDistance:
                targetVelocity = 0.0
                integral       = 0.0

            # Hard safety: if too close AND moving, force stop immediately
            if filteredDistance < standstillDistance * 0.8:
                targetVelocity = 0.0
                integral       = 0.0
                prevError      = 0.0

            # Rate-limit target velocity decrease
            if targetVelocity < lastTargetVelocity:
                diff = lastTargetVelocity - targetVelocity
                if diff > maxTargetVelocityDecrease:
                    targetVelocity = lastTargetVelocity - maxTargetVelocityDecrease
            lastTargetVelocity = targetVelocity

            # ── PID Controller ─────────────────────────────────────────────
            velocityError = targetVelocity - measuredVelocity
            pwmOutput     = lastPwmOutput
            pidOutput     = 0.0

            # Anti-windup
            if measuredVelocity < 0.01 and lastPwmOutput >= 200 and targetVelocity > 0:
                integral = 0.0

            if abs(velocityError) >= deadbandError:
                derivative = (velocityError - prevError) / SAMPLE_INTERVAL
                prevError  = velocityError
                pidOutput  = Kp * velocityError + Ki * integral + Kd * derivative

                pwmOutput  = int(min(max(pidOutput, 0), 255))

                not_saturated = (pwmOutput < 255 or velocityError < 0) and \
                                (pwmOutput > 0  or velocityError > 0)
                if not_saturated:
                    integral += velocityError * SAMPLE_INTERVAL
                    integral  = max(-maxIntegral, min(integral, maxIntegral))

                floor = startPWM if measuredVelocity < 0.01 else runPWM
                if 0 < pwmOutput < floor:
                    pwmOutput = floor

            # Rate-limit PWM step
            if pwmOutput > lastPwmOutput + pwmStepLimit:
                pwmOutput = lastPwmOutput + pwmStepLimit
            elif pwmOutput < lastPwmOutput - pwmStepLimit:
                pwmOutput = lastPwmOutput - pwmStepLimit
            lastPwmOutput = pwmOutput

            # ── Motor output ───────────────────────────────────────────────
            if filteredDistance <= standstillDistance:
                pwmOutput = 0

            pwm_val = max(0.0, min(pwmOutput / 255.0, 1.0))
            pwm_ena.value = pwm_val
            pwm_enb.value = pwm_val
            if pwm_val > 0:
                motor_left.forward()
                motor_right.forward()
            else:
                motor_left.stop()
                motor_right.stop()

            # ── Telemetry & Logging ────────────────────────────────────────
            # Using current epoch time so multiple runs are easily distinguishable chronologically
            # If you prefer elapsed time per run, use (time() - run_start_time)
            current_timestamp = time() 
            
            # Save data row to memory
            telemetry_log.append([
                round(current_timestamp, 3),
                round(filteredDistance, 3),
                round(safeDistance, 3),
                round(distanceError, 3),
                round(targetVelocity, 3),
                round(measuredVelocity, 3),
                round(velocityError, 3),
                round(pidOutput, 1),
                pwmOutput
            ])

            # Print to console so you still see what's happening
            print(f"[{current_timestamp:.1f}] Dist: {filteredDistance:.2f}m | TgtVel: {targetVelocity:.2f}m/s | Vel: {measuredVelocity:.2f}m/s | PWM: {pwmOutput}")

            # ── Maintain sample interval ───────────────────────────────────
            elapsed = time() - loop_start
            if elapsed < SAMPLE_INTERVAL:
                sleep(SAMPLE_INTERVAL - elapsed)

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        # Stop Motors
        pwm_ena.off()
        pwm_enb.off()
        motor_left.stop()
        motor_right.stop()
        
        # Save accumulated data to CSV in APPEND mode
        print(f"Saving collected data to '{CSV_FILENAME}'...")
        try:
            # Check if file exists and is not empty to decide if we need headers
            file_exists = os.path.isfile(CSV_FILENAME)
            is_empty = not file_exists or os.path.getsize(CSV_FILENAME) == 0

            # mode='a' appends to the file instead of overwriting
            with open(CSV_FILENAME, mode='a', newline='') as file:
                writer = csv.writer(file)
                
                # Only write headers if this is a brand new file
                if is_empty:
                    writer.writerow(CSV_HEADERS)
                
                writer.writerows(telemetry_log)
                
            print(f"Successfully appended {len(telemetry_log)} records.")
        except Exception as e:
            print(f"Failed to save CSV file: {e}")
            
        print("All systems safely shut down.")

if __name__ == "__main__":
    main()