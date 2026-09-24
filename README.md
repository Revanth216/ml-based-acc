# ML Equivalent Control Strategy For ACC Considering Front And Rear Vehicles

## Overview
This repository contains the implementation of a Miniature Adaptive Cruise Control (ACC) testbed built on a Raspberry Pi 4[cite: 4]. The system investigates Machine Learning-enhanced predictive control by combining cascaded Model Predictive Control (MPC) with trained Multilayer Perceptron (MLP) networks[cite: 4]. 

The core innovation replaces computationally heavy online optimization with a fast, offline-trained NumPy MLP that replicates optimal MPC behavior[cite: 4]. By eliminating deep-learning framework overhead via a TensorFlow-to-NumPy migration, the surrogate achieves sub-millisecond inference speeds suitable for resource-constrained edge deployment[cite: 4].

## Mathematical Foundation & Architecture
The control strategy extends environmental awareness to both front and rear traffic to mitigate rear-end collisions[cite: 4].
* **Dynamic Safe Distance Model:** The safe distance is formulated as $d_{safe} = HWT_{dem} \times v_{e} + d_{standstill}$, where demanded headway time ($HWT_{dem}$) is 2.5 s and standstill distance is 4 m[cite: 4].
* **Cascaded Dual-Aware MPC:** 
  * **High-Level MPC (MPC1):** Uses relative kinematic equations for front, ego, and rear vehicles to compute a reference velocity, solved via Hildreth's Quadratic Programming[cite: 4].
  * **Low-Level MPC (MPC2):** A 3-state dynamic model that receives the reference velocity and outputs throttle/brake commands balancing speed tracking and smooth transitions[cite: 4].
* **Velocity Estimation:** An adaptive Weighted Moving Average (WMA) filter with spike rejection calculates velocity from encoder pulses using $v = (\frac{\Delta counts}{348}) \times \pi \times 0.065$ m/s[cite: 4].

## Hardware Specifications
| Component | Specification | Role |
| :--- | :--- | :--- |
| **Compute Unit** | Raspberry Pi 4 (4 GB RAM) | Runs Python controller, Flask GUI, and NumPy MLP inference[cite: 4] |
| **Motor Driver** | L298N Dual H-Bridge | PWM motor control via GPIO; utilizes calibrated deadband logic at $MIN_{PWM} \approx 95/255$[cite: 4] |
| **Front Sensor** | HC-SR04 Ultrasonic | Measures distance to the vehicle ahead[cite: 4] |
| **Rear Sensor** | HC-SR04 Ultrasonic | Measures distance to the following vehicle[cite: 4] |
| **Wheel Encoder** | 348 counts/rev (65 mm wheel) | Provides real-time velocity feedback[cite: 4] |

## Software Pipeline
The control pipeline progresses through five sequential phases[cite: 4]:
1. **`frontvehiclewithoutus.py`**: Flask GUI for empirical PWM deadband calibration[cite: 4].
2. **`mlpid.py`**: 4-feature NumPy MLP replacing the standard TensorFlow PID controller[cite: 4].
3. **`frontmpc.py` & `mlfront.py`**: 8,000-point cascaded front-aware MPC simulation and MLP surrogate training (RMSE-optimized)[cite: 4].
4. **`dualaware.py`**: Implementation of High-Level (kinematic) and Low-Level (dynamic) MPC using Hildreth's QP[cite: 4].
5. **`mldual.py`**: Hardware deployment of the dual-aware MLP surrogate demonstrating rear-approach speed-increase logic[cite: 4].

## Key Performance Benchmarks
The NumPy MLP surrogate was validated against the mathematical MPC and standard PID controllers on held-out test sets[cite: 4]:

* **Inference Latency:** Achieved full on-device deployment with sub-0.4 ms inference latency[cite: 4]. Specifically, inference required only 0.398 ms for the real-world front-aware model and 0.145 ms for the dual-aware simulation model[cite: 4].
* **Tracking Accuracy:** Demonstrated a 50% reduction in velocity-tracking RMSE over the TensorFlow PID baseline[cite: 4].
* **Model Fidelity:** The front-aware surrogate achieved an $R^{2}$ correlation of 0.991 (Sim) and 0.988 (Real)[cite: 4]. The dual-aware surrogate achieved an $R^{2}$ of 0.947 with a Mean Absolute Error (MAE) of 0.022[cite: 4].

## Artifacts & Visualizations
*(Instructions for Revanth: Upload your hardware images and validation plots to an `assets` folder in your repository, then uncomment and update the placeholder links below)*

* `![Hardware Architecture](./assets/system_block_diagram.png)` - System Block Diagram[cite: 4]
* `![Hardware Setup Image](./assets/car_setup.jpg)` - Robotic car setup with dual HC-SR04 sensors and Raspberry Pi[cite: 4]
* `![PWM Validation Plot](./assets/mpc_vs_mlp_validation.png)` - Near-perfect overlap between actual MPC PWM and MLP Predicted PWM[cite: 4]
* `[Watch the Hardware Demonstration Video](link_to_youtube_or_drive)`

## Future Work
* Hardware-tuning of MPC1/MPC2 Q/R weights[cite: 4].
* Integration of a Reinforcement Learning policy replacing supervised MLP mimicry[cite: 4].
* Implementation of EKF-based sensor fusion for ultrasonic noise rejection[cite: 4].
