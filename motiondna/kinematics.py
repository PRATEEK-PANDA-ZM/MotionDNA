"""Landmark-based lower-limb kinematic calculations.

Coordinates are expected in metres, with one row per frame and columns named
``<landmark>_<axis>`` (for example ``left_hip_x``).  The algorithms work with
MediaPipe/YOLO landmark exports after their landmark names are mapped to this
format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_LANDMARKS = (
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
)


def _points(frame: pd.DataFrame, landmark: str) -> np.ndarray:
    columns = [f"{landmark}_{axis}" for axis in "xyz"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing coordinate columns: {', '.join(missing)}")
    return frame[columns].to_numpy(dtype=float)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the included angle ABC in degrees for each frame."""
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cosine = np.divide(np.sum(ba * bc, axis=1), denom, out=np.full(len(b), np.nan), where=denom > 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def _velocity(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values))
    return np.gradient(values, time)


def analyze_landmarks(landmarks: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    """Compute sagittal-plane proxy angles and angular velocities.

The knee measure is 180° at full extension; the displayed knee flexion equals
180° minus this included angle.  Frontal-plane knee-to-ankle offset is a simple
screening proxy, not a clinical valgus measurement.
    """
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    frame = landmarks.copy().reset_index(drop=True)
    if "time_s" not in frame:
        frame["time_s"] = np.arange(len(frame)) / fps
    time = frame["time_s"].to_numpy(float)
    if len(time) > 1 and np.any(np.diff(time) <= 0):
        raise ValueError("time_s must increase for every row.")

    for side in ("left", "right"):
        hip, knee, ankle = (_points(frame, f"{side}_{part}") for part in ("hip", "knee", "ankle"))
        included = joint_angle(hip, knee, ankle)
        frame[f"{side}_knee_flexion_deg"] = 180 - included
        frame[f"{side}_knee_angular_velocity_dps"] = _velocity(frame[f"{side}_knee_flexion_deg"].to_numpy(), time)
        # Positive values mean the knee is medial to the ankle in the camera's x axis.
        frame[f"{side}_knee_ankle_offset_cm"] = (knee[:, 0] - ankle[:, 0]) * 100
        frame[f"{side}_hip_height_m"] = hip[:, 1]

    frame["mean_knee_flexion_deg"] = frame[["left_knee_flexion_deg", "right_knee_flexion_deg"]].mean(axis=1)
    frame["mean_knee_velocity_dps"] = frame[["left_knee_angular_velocity_dps", "right_knee_angular_velocity_dps"]].mean(axis=1)
    return frame


def make_demo_trial(movement: str = "Squat", frames: int = 150, fps: float = 30.0) -> pd.DataFrame:
    """Create plausible, labelled synthetic coordinates for testing the dashboard."""
    t = np.arange(frames) / fps
    phase = np.linspace(0, np.pi, frames)
    depth = 0.34 * np.sin(phase) if movement == "Squat" else 0.18 * np.exp(-((t - t.mean()) / 0.18) ** 2)
    data: dict[str, np.ndarray] = {"time_s": t}
    for sign, side in ((-1, "left"), (1, "right")):
        hip_x = sign * 0.15 + 0.025 * np.sin(phase)
        hip_y = 1.0 - depth
        knee_x = sign * (0.12 - 0.018 * np.sin(phase))
        knee_y = 0.56 - depth * 0.35
        ankle_x = sign * 0.12
        ankle_y = np.full(frames, 0.10)
        for part, x, y in (("hip", hip_x, hip_y), ("knee", knee_x, knee_y), ("ankle", ankle_x, ankle_y)):
            data[f"{side}_{part}_x"] = x
            data[f"{side}_{part}_y"] = y
            data[f"{side}_{part}_z"] = np.zeros(frames)
    return pd.DataFrame(data)
