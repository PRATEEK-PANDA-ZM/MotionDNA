"""Video-to-landmark extraction using MediaPipe Pose Landmarker.

The source video is processed locally.  Image-normalised pose coordinates are
scaled by the entered body-height estimate solely to make dashboard values
readable; they are not camera-calibrated 3D coordinates.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = Path(tempfile.gettempdir()) / "motiondna_pose_landmarker_full.task"
LANDMARKS = {
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def _model_path() -> str:
    if not MODEL_PATH.exists():
        urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


def _draw_lower_limb(frame: np.ndarray, landmarks: list) -> np.ndarray:
    """Draw only the lower-limb landmarks needed by MotionDNA."""
    image = frame.copy()
    height, width = image.shape[:2]
    for first, second in ((23, 25), (25, 27), (24, 26), (26, 28), (23, 24)):
        a, b = landmarks[first], landmarks[second]
        p1, p2 = (int(a.x * width), int(a.y * height)), (int(b.x * width), int(b.y * height))
        cv2.line(image, p1, p2, (40, 220, 80), 3)
    for index in LANDMARKS.values():
        point = landmarks[index]
        cv2.circle(image, (int(point.x * width), int(point.y * height)), 5, (30, 60, 255), -1)
    return image


def extract_video_landmarks(
    video_path: str,
    body_height_m: float = 1.70,
    sample_every: int = 1,
    progress_callback=None,
) -> tuple[pd.DataFrame, np.ndarray | None, float, int]:
    """Extract one-person pose landmarks from a video.

Returns a landmark table, annotated preview frame, source FPS, and the number
of frames where a pose was found.  `sample_every` allows faster dashboard
prototyping on long videos.
    """
    if sample_every < 1 or body_height_m <= 0:
        raise ValueError("Body height and frame sampling values must be positive.")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("The video could not be opened. Try MP4, MOV, or AVI.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_model_path()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    rows: list[dict[str, float]] = []
    preview = None
    index = 0
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                if index % sample_every == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect_for_video(image, round(index / fps * 1000))
                    if result.pose_landmarks:
                        pose = result.pose_landmarks[0]
                        row = {"time_s": index / fps}
                        for name, landmark_index in LANDMARKS.items():
                            point = pose[landmark_index]
                            # Image coordinates scaled by an anthropometric estimate, not calibration.
                            row[f"{name}_x"] = point.x * body_height_m
                            row[f"{name}_y"] = (1 - point.y) * body_height_m
                            row[f"{name}_z"] = point.z * body_height_m
                        rows.append(row)
                        if preview is None:
                            preview = cv2.cvtColor(_draw_lower_limb(bgr, pose), cv2.COLOR_BGR2RGB)
                index += 1
                if progress_callback and frame_count:
                    progress_callback(min(index / frame_count, 1.0))
    finally:
        capture.release()
    if len(rows) < 3:
        raise ValueError("Too few poses were detected. Ensure one full body is visible and well lit.")
    return pd.DataFrame(rows), preview, float(fps) / sample_every, len(rows)
