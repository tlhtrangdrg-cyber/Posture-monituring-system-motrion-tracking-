# Posture-Monitoring-System-Motion-Tracking

**Real-Time Vision-Based Posture Monitoring System using vision-based motion tracking to detect and visualize posture deviations through webcam input.**


## Abstract

This project implements a real-time posture monitoring system using vision-based motion tracking from a standard webcam. The system analyzes upper-body posture by tracking 2D body landmarks and detecting deviations from a calibrated neutral posture. Both front-view and side-view analysis modes are supported. The system provides real-time visual feedback and post-session analytical graphs to evaluate posture behavior over time.


## 1. Introduction

Maintaining proper sitting posture is important for reducing musculoskeletal strain, particularly during prolonged computer use. Traditional posture monitoring solutions often rely on wearable sensors or specialized hardware. This project explores a non-invasive alternative using computer vision and motion tracking.

The goals of this system are to:

* Track upper-body posture in real time
* Detect posture deviations relative to an individualized baseline
* Visualize posture behavior during a monitoring session


## 2. System Overview

The system processes live webcam input and follows this pipeline:

1. Video capture using OpenCV
2. 2D pose landmark detection
3. User-specific posture calibration
4. Motion-based posture metric computation
5. Signal stabilization and threshold evaluation
6. Real-time visualization and data logging


## 3. Motion Tracking and Pose Estimation

The system uses **MediaPipe Pose Landmarker** to extract 2D body landmarks from each video frame. These landmarks represent key anatomical points such as shoulders, hips, neck, and ears.

All posture measurements are derived from the relative geometry of these tracked points.


## 4. Calibration Phase

To account for individual differences in body proportions and camera setup, the system performs a calibration step:

* **Duration:** 5 seconds
* The user is instructed to sit upright
* Posture metrics are collected and averaged
* The averaged values define the neutral posture baseline

Subsequent posture evaluations are measured as deviations from this baseline.


## 5. Posture Metrics

### Front View

**Shoulder Tilt**
Measures vertical imbalance between the left and right shoulders.

**Neck Offset**
Measures horizontal displacement of the neck relative to the shoulder midpoint.

**Body Axis Angle**
Measures angular deviation between shoulder and hip midpoints.


### Side View

**Neck Angle**
Measures forward head posture relative to the shoulder.

**Torso Angle**
Measures upper-body inclination to detect slouching.


These metrics are computed from landmark coordinates and expressed as normalized offsets or angular values.


## 6. Signal Stabilization

To reduce noise and false detections, the system applies:

* Temporal smoothing using a sliding window average
* Deadzone filtering to ignore insignificant motion
* Threshold-based detection to classify posture deviations

These techniques improve stability during real-time tracking.


## 7. Visualization and Feedback

The system provides:

* A real-time skeletal overlay drawn on the user
* Color-coded posture indication (normal vs. deviated)
* On-screen calibration progress indicator
* Time-series plots after program termination

Post-session graphs display posture metrics over time with threshold reference lines and highlighted deviation periods.


## 8. User Controls

| Key | Action                                |
| --- | ------------------------------------- |
| `F` | Switch to front-view posture analysis |
| `S` | Switch to side-view posture analysis  |
| `Q` | Exit program and display graphs       |


## 9. Mathematical Formulation and Pseudocode

### Coordinate Definition

Each pose landmark is represented as a 2D point:

```
p = (x, y)
```

Coordinates are scaled to the image resolution.


### Shoulder Tilt

```
(y_left - y_right) / |x_right - x_left|
```


### Neck Offset

```
(x_neck - x_midpoint) / |x_right - x_left|
```


### Body Axis Angle

```
atan2(y_hip_mid - y_shoulder_mid, x_hip_mid - x_shoulder_mid)
```


### Angle Computation (Side View)

```
|atan2(x2 - x1, y1 - y2)|
```

Used for:

* Forward head posture
* Torso inclination


### Temporal Smoothing

Moving average filter:

```
SmoothedValue = (1 / N) * Σ(value_i)
```


### Deadzone Filtering

```
if |value| < deadzone:
    value = 0
```


### Calibration Baseline

```
Baseline = (1 / T) * Σ(metric_t)
Deviation = metric - Baseline
```


### Threshold-Based Detection

```
if |Deviation| > Threshold:
    PostureDeviation = True
```


### Overall Detection Logic (Pseudocode)

```
Initialize camera and pose detector
Set mode = FRONT
calibrated = False

WHILE application is running:
    Capture frame
    Detect pose landmarks

    IF landmarks detected:
        Compute posture metrics
        Apply smoothing

        IF not calibrated:
            Collect calibration data
            IF calibration time reached:
                Compute baseline
                calibrated = True
        ELSE:
            Subtract baseline
            Apply deadzone
            Compare with thresholds
            Log deviations

        Render skeleton and posture state

    IF key pressed:
        Switch mode or exit

Display posture graphs
```


## 10. Technologies Used

* Python
* OpenCV – video capture and rendering
* MediaPipe Pose – vision-based motion tracking
* NumPy – numerical computations
* Matplotlib – data visualization


## 11. Installation

### Requirements

* Python 3.8+
* Webcam

### Dependencies

```bash
pip install opencv-python mediapipe numpy matplotlib
```

Ensure the MediaPipe model file is present:

```
pose_landmarker_lite.task
```


## 12. Usage

```bash
python main.py
```

1. Sit upright during the calibration phase
2. Maintain normal posture while monitoring
3. Switch between front and side views as needed
4. Press `Q` to exit and view posture graphs


## 13. Limitations

* Supports single-person tracking only
* Uses 2D pose estimation (no depth information)
* Accuracy depends on lighting and camera placement
* Not intended for medical diagnosis


## 14. Future Improvements

* Multi-person posture tracking
* Real-time alerts or notifications
* Data export for long-term analysis


## 15. License

This project is intended for **educational and research purposes**.


