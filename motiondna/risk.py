"""Transparent movement-screening heuristics for MotionDNA.

This module is intentionally a decision-support screen rather than a medical
diagnosis or a trained clinical prediction model.
"""

from __future__ import annotations

import pandas as pd


def score_movement_risk(kinematics: pd.DataFrame, movement: str) -> tuple[int, str, list[dict[str, str]]]:
    """Return a 0–100 heuristic score, band, and plain-language findings."""
    findings: list[dict[str, str]] = []
    score = 0
    peak_flexion = float(kinematics["mean_knee_flexion_deg"].max())
    max_velocity = float(kinematics["mean_knee_velocity_dps"].abs().max())
    asymmetry = float((kinematics["left_knee_flexion_deg"] - kinematics["right_knee_flexion_deg"]).abs().max())
    medial_offset = float(kinematics[["left_knee_ankle_offset_cm", "right_knee_ankle_offset_cm"]].abs().max().max())

    if movement == "Squat" and peak_flexion < 70:
        score += 20; findings.append({"factor": "Squat depth", "detail": f"Peak flexion was {peak_flexion:.0f}° (screen target: ≥70°).", "level": "Watch"})
    if movement == "Drop jump" and peak_flexion < 35:
        score += 25; findings.append({"factor": "Landing strategy", "detail": f"Peak landing flexion was {peak_flexion:.0f}° (screen target: ≥35°).", "level": "Watch"})
    if asymmetry > 12:
        score += 25; findings.append({"factor": "Side-to-side symmetry", "detail": f"Maximum knee-flexion difference was {asymmetry:.0f}° (screen threshold: 12°).", "level": "Elevated"})
    if medial_offset > 4:
        score += 20; findings.append({"factor": "Knee–ankle alignment proxy", "detail": f"Maximum horizontal offset was {medial_offset:.1f} cm (screen threshold: 4 cm).", "level": "Watch"})
    if max_velocity > 500:
        score += 10; findings.append({"factor": "Angular velocity", "detail": f"Peak knee angular velocity was {max_velocity:.0f}°/s.", "level": "Watch"})
    score = min(score, 100)
    band = "Lower" if score < 20 else "Moderate" if score < 50 else "Elevated"
    if not findings:
        findings.append({"factor": "Screen result", "detail": "No configured screening thresholds were exceeded in this trial.", "level": "Clear"})
    return score, band, findings
