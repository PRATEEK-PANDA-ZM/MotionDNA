from pathlib import Path
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st

from motiondna.kinematics import analyze_landmarks, make_demo_trial
from motiondna.pose import extract_video_landmarks
from motiondna.risk import score_movement_risk

st.set_page_config(page_title="MotionDNA", page_icon="🧬", layout="wide")
st.title("MotionDNA")
st.caption("Markerless movement analysis • kinematics • injury-risk screening")

with st.sidebar:
    st.header("Trial settings")
    movement = st.selectbox("Movement", ["Squat", "Drop jump"])
    fps = st.number_input("Video frame rate (FPS)", min_value=1.0, value=30.0, step=1.0)
    uploaded = st.file_uploader("Landmark CSV", type="csv")
    video = st.file_uploader("Or upload a movement video", type=["mp4", "mov", "avi", "m4v"])
    body_height = st.number_input("Participant height estimate (m)", min_value=0.8, max_value=2.5, value=1.70, step=0.01)
    sample_every = st.selectbox("Video sampling", [1, 2, 3, 5], format_func=lambda x: f"Every {x} frame(s)")
    st.caption("Expected: time_s (optional) plus left/right hip, knee, ankle x/y/z columns in metres.")

try:
    preview = None
    source_note = "Showing synthetic demo data."
    if video:
        suffix = Path(video.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(video.getbuffer())
            video_path = temporary.name
        progress = st.progress(0, text="Detecting pose landmarks in video…")
        try:
            raw, preview, derived_fps, detected = extract_video_landmarks(
                video_path, body_height, sample_every, lambda value: progress.progress(value, text="Detecting pose landmarks in video…")
            )
        finally:
            Path(video_path).unlink(missing_ok=True)
            progress.empty()
        result = analyze_landmarks(raw, derived_fps)
        source_note = f"Video analysed: pose found in {detected} frames. Values are 2D/depth-relative estimates."
    else:
        raw = pd.read_csv(uploaded) if uploaded else make_demo_trial(movement, fps=fps)
        result = analyze_landmarks(raw, fps)
        source_note = "Showing uploaded landmark trial." if uploaded else source_note
except (ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not analyse the trial: {error}")
    st.stop()

score, band, findings = score_movement_risk(result, movement)
peak = result["mean_knee_flexion_deg"].max()
asymmetry = (result["left_knee_flexion_deg"] - result["right_knee_flexion_deg"]).abs().max()
velocity = result["mean_knee_velocity_dps"].abs().max()

st.info(source_note)
if preview is not None:
    st.image(preview, caption="MediaPipe lower-limb pose preview", use_container_width=True)
a, b, c, d = st.columns(4)
a.metric("Screening score", f"{score}/100", band)
b.metric("Peak knee flexion", f"{peak:.1f}°")
c.metric("Max asymmetry", f"{asymmetry:.1f}°")
d.metric("Peak angular velocity", f"{velocity:.0f}°/s")

st.subheader("Knee kinematics")
chart = result.melt(id_vars="time_s", value_vars=["left_knee_flexion_deg", "right_knee_flexion_deg"], var_name="Side", value_name="Knee flexion (°)")
st.plotly_chart(px.line(chart, x="time_s", y="Knee flexion (°)", color="Side", labels={"time_s": "Time (s)"}), use_container_width=True)

st.subheader("Movement-screen findings")
st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)
st.caption("Educational screening only. This tool does not diagnose injury risk; interpret results with a qualified clinician and validate against calibrated 3D capture.")

download = result.to_csv(index=False).encode("utf-8")
st.download_button("Download analysed CSV", data=download, file_name="motiondna_kinematics.csv", mime="text/csv")

with st.expander("CSV schema and pipeline notes"):
    st.code("time_s,left_hip_x,left_hip_y,left_hip_z,left_knee_x,...,right_ankle_z")
    st.write("Video upload uses MediaPipe Pose Landmarker locally and derives a 2D/depth-relative landmark stream. Use calibrated 3D coordinates in metres, or multi-view triangulation, for research-grade 3D joint analysis.")
