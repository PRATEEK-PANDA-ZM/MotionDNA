# MotionDNA

MotionDNA is a markerless-motion-capture dashboard starter for biomechanics projects. Upload a movement video to extract lower-limb landmarks locally with MediaPipe, or upload calibrated 3D landmarks directly. It calculates knee flexion and angular velocity, then runs transparent movement-screening rules for squat and drop-jump trials.

## Run it

```powershell
py -3.14 -m pip install -r requirements.txt
py -3.14 -m streamlit run app.py
```

If your Python version differs, replace `3.14` with the version shown by `py --list`.

The app opens with synthetic data. Upload a CSV or a movement video (MP4, MOV, AVI, M4V) when you are ready to analyse a trial. The first video run downloads MediaPipe's pose-landmarker model to the operating system temporary folder.

## Input format

One row represents one video frame. Coordinates must be **calibrated 3D metres**. Include `time_s`, or set an FPS in the dashboard. Required landmark names are `left_hip`, `left_knee`, `left_ankle`, `right_hip`, `right_knee`, and `right_ankle`, each with `_x`, `_y`, `_z` suffixes.

## Important interpretation note

The risk score is a configurable, explainable educational screen—not a validated clinical prediction or medical diagnosis. Single-camera video analysis is 2D plus depth-relative estimation; it is not calibrated 3D. Validate pose-estimation error, camera calibration, and thresholds against your own study protocol and an appropriate reference system before using it for research conclusions.
