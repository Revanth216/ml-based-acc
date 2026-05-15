from gpiozero import Device, Motor, PWMOutputDevice, DistanceSensor, Button
from gpiozero.pins.pigpio import PiGPIOFactory
import math
import time
import numpy as np
import tflite_runtime.interpreter as tflite

# --- Configuration ---
Device.pin_factory = PiGPIOFactory()

# Scalers (Keep these exactly as your model expects them)
SCALER_MEAN = np.array([115.64379085, 0.98366013, 29.0245098, 30.0], dtype=np.float32)
SCALER_STD = np.array([80.22929096, 2.47927244, 2.47821632, 1.0], dtype=np.float32)

# Pin Definitions
ENA = 17
IN1 = 27
IN2 = 22
IN3 = 23
IN4 = 24
ENB = 25
TRIG = 13
ECHO = 6
ENCODER_DT = 5

# Mechanical Constants
COUNTS_PER_REV = 400
WHEEL_DIAMETER = 6.5
WHEEL_CIRCUM = math.pi * WHEEL_DIAMETER

# --- TUNING PARAMETERS ---
SAMPLE_INTERVAL = 0.05  # Reduced slightly for faster reaction
standstillDistance = 15.0
headwayTime = 1.0

# 1. UPDATED MAX VELOCITY to 30 cm/s as requested
maxVelocity = 30.0 
maxAccel = 10

# 2. PWM LIMITS
pwmStepLimit_Accel = 10  # Slow acceleration (smooth)
pwmStepLimit_Brake = 255 # Infinite braking (instant stop)

# Filters
alpha = 0.8
distance_alpha = 0.8    # Increased trust in new sensor data for faster reaction

# TFLite Setup
interpreter = tflite.Interpreter(model_path="/home/pi/tensor.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

# Hardware Setup
motor_left = Motor(forward=IN1, backward=IN2)
motor_right = Motor(backward=IN3, forward=IN4)
pwm_ena = PWMOutputDevice(ENA)
pwm_enb = PWMOutputDevice(ENB)
distance_sensor = DistanceSensor(echo=ECHO, trigger=TRIG, max_distance=4)

pulse_count = 0

def encoder_callback():
    global pulse_count
    pulse_count += 1

encoder = Button(ENCODER_DT, pull_up=True)
encoder.when_pressed = encoder_callback

def ewma_filter(prev, new_val, alpha):
    if prev is None:
        return new_val
    return alpha * prev + (1 - alpha) * new_val

def main():
    global pulse_count 
    filtered_dist = None
    filtered_vel = 0
    last_pwm = 0
    
    print("System Started. Waiting for sensor stabilization...")
    time.sleep(1)

    try:
        while True:
            start_time = time.time()
            
            # --- 1. SENSOR READING ---
            # Multiply by 100 to get cm
            raw_dist = distance_sensor.distance * 100
            
            # Filter distance (EWMA)
            filtered_dist = ewma_filter(filtered_dist, raw_dist, distance_alpha)
            
            # Calculate Velocity
            current_pulses = pulse_count
            pulse_count = 0
            revs = current_pulses / COUNTS_PER_REV
            dist_traveled = revs * WHEEL_CIRCUM
            raw_vel = dist_traveled / SAMPLE_INTERVAL
            filtered_vel = ewma_filter(filtered_vel, raw_vel, alpha)

            # --- 2. SAFETY CALCULATIONS ---
            safe_distance = standstillDistance + (headwayTime * filtered_vel)
            distance_error = filtered_dist - safe_distance

            # --- 3. HARD SAFETY OVERRIDE (PRIORITY) ---
            # If we are physically closer than 15cm (plus a tiny buffer), STOP regardless of the AI.
            if raw_dist < standstillDistance:
                target_vel = 0
                pwm_out = 0
                print(f"!!! EMERGENCY BRAKE !!! Dist: {raw_dist:.1f}")
                
                # Apply brake immediately
                pwm_ena.value = 0
                pwm_enb.value = 0
                motor_left.stop()
                motor_right.stop()
                last_pwm = 0
                
                # Sleep and restart loop to avoid running the AI
                time.sleep(SAMPLE_INTERVAL)
                continue

            # --- 4. TARGET VELOCITY LOGIC ---
            if distance_error > 0:
                # Calculate required velocity using kinematics v^2 = u^2 + 2as
                # We cap this to maxVelocity immediately
                needed_v_sq = (filtered_vel ** 2) + (2 * maxAccel * distance_error)
                target_vel = math.sqrt(needed_v_sq)
                target_vel = min(target_vel, maxVelocity)
            else:
                # If error is negative (too close relative to speed), target is 0
                target_vel = 0

            velocity_error = target_vel - filtered_vel

            # --- 5. AI PREDICTION ---
            input_features = np.array([distance_error, velocity_error, filtered_vel, target_vel], dtype=np.float32)
            scaled_input = (input_features - SCALER_MEAN) / SCALER_STD
            input_tensor = np.expand_dims(scaled_input, axis=0)
            
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_index)[0][0]
            
            # Clip raw prediction to valid PWM range
            pwm_out = int(np.clip(prediction, 0, 255))

            # --- 6. ASYMMETRIC PWM RAMPING ---
            # This is the critical fix. 
            # If we are speeding up, we go slow (limit step).
            # If we are slowing down, we allow BIG steps (brake fast).
            
            if pwm_out > last_pwm:
                # Accelerating
                if pwm_out > last_pwm + pwmStepLimit_Accel:
                    pwm_out = last_pwm + pwmStepLimit_Accel
            else:
                # Decelerating / Braking
                # We do NOT limit the step here significantly. 
                # This allows PWM to drop from 100 to 0 instantly if needed.
                pass 
           
            last_pwm = pwm_out
            pwm_val = pwm_out / 255.0
            
            # --- 7. ACTUATION ---
            pwm_ena.value = pwm_val
            pwm_enb.value = pwm_val
            
            if pwm_val > 0:
                motor_left.forward()
                motor_right.forward()
            else:
                motor_left.stop()
                motor_right.stop()
          
            # Debug output
            print(f"D:{filtered_dist:.1f} Safe:{safe_distance:.1f} Err:{distance_error:.1f} TV:{target_vel:.1f} FV:{filtered_vel:.1f} PWM:{pwm_out}")
            
            # Loop Timing
            elapsed = time.time() - start_time
            if elapsed < SAMPLE_INTERVAL:
                time.sleep(SAMPLE_INTERVAL - elapsed)
                
    except KeyboardInterrupt:
        pass
    finally:
        pwm_ena.off()
        pwm_enb.off()
        motor_left.close()
        motor_right.close()

if __name__ == "__main__":
    main()