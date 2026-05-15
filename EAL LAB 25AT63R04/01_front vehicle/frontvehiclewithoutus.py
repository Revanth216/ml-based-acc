from flask import Flask, request, render_template_string
from gpiozero import Motor, PWMOutputDevice

app = Flask(__name__)

ENA = 17
IN1 = 27
IN2 = 22
IN3 = 23
IN4 = 24
ENB = 25

motor_left = Motor(forward=IN1, backward=IN2, pwm=True)
motor_right = Motor(backward=IN3, forward=IN4, pwm=True)
pwm_ena = PWMOutputDevice(ENA)
pwm_enb = PWMOutputDevice(ENB)
current_direction = "stop"

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Pi Car Test</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            display: flex; flex-direction: column; align-items: center; 
            background-color: #222; color: white; height: 100vh; margin: 0;
            padding-top: 20px;
        }
        .controls {
            display: flex; flex-direction: column; align-items: center; gap: 20px;
            width: 90%; max-width: 400px;
        }
        .btn {
            padding: 20px; font-size: 1.5em; font-weight: bold; border-radius: 10px;
            border: none; width: 100%; cursor: pointer; color: white;
        }
        .btn-fwd { background-color: #4CAF50; }
        .btn-rev { background-color: #2196F3; }
        .btn-stop { background-color: #f44336; height: 100px; }
        
        .slider-container { width: 100%; margin-top: 20px; text-align: center; }
        input[type=range] { width: 100%; height: 40px; }
        #speed-display { font-size: 1.5em; margin-bottom: 10px; }
    </style>
</head>
<body>
    <h2>Manual Test</h2>
    
    <div class="controls">
        <button class="btn btn-fwd" onclick="sendCommand('fwd')">FORWARD</button>
        <button class="btn btn-stop" onclick="sendCommand('stop')">STOP</button>
        <button class="btn btn-rev" onclick="sendCommand('rev')">REVERSE</button>
        
        <div class="slider-container">
            <div id="speed-display">Set PWM: 0</div>
            <input type="range" min="0" max="255" value="0" id="speed-slider" oninput="updateSpeed(this.value)" onchange="sendCommand('current')">
        </div>
    </div>

    <script>
        let currentPwm = 0;
        let currentDir = 'stop';

        function updateSpeed(value) {
            currentPwm = value;
            document.getElementById('speed-display').innerText = 'Set PWM: ' + value;
        }

        function sendCommand(dir) {
            if (dir !== 'current') {
                currentDir = dir;
            }
            fetch(`/drive?dir=${currentDir}&pwm=${currentPwm}`);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/drive')
def drive():
    global current_direction
    direction = request.args.get('dir', 'stop')
    pwm_str = request.args.get('pwm', '0')
    try:
        target_pwm = int(pwm_str)
    except ValueError:
        target_pwm = 0
    target_pwm = max(0, min(target_pwm, 255))
    pwm_val = target_pwm / 255.0
    pwm_ena.value = pwm_val
    pwm_enb.value = pwm_val
    if direction == 'fwd':
        motor_left.forward()
        motor_right.forward()
    elif direction == 'rev':
        motor_left.backward()
        motor_right.backward()
    else:
        motor_left.stop()
        motor_right.stop()
    current_direction = direction
    return "OK"

if __name__ == '__main__':
    print("Starting server... Open your phone browser and go to:")
    print("http://<YOUR_PI_IP_ADDRESS>:5000")
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        motor_left.stop()
        motor_right.stop()
        pwm_ena.off()
        pwm_enb.off()
        print("Motors shut down.")