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
    annotated_frames = []
    source_note = "Showing synthetic demo data."
    if video:
        suffix = Path(video.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(video.getbuffer())
            video_path = temporary.name
        progress = st.progress(0, text="Detecting pose landmarks in video…")
        try:
            raw, preview, derived_fps, detected, annotated_frames = extract_video_landmarks(
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
except Exception as error:
    st.error(f"Could not process video: {error}")
    st.warning("⚠️ **MediaPipe Compatibility Notice**: MediaPipe C++ native bindings are incompatible with Python 3.14. If you are deployed on Streamlit Cloud, go to **App Settings > Advanced settings** and change the **Python version to 3.11 or 3.10**.")
    st.stop()

score, band, findings = score_movement_risk(result, movement)
peak = result["mean_knee_flexion_deg"].max()
asymmetry = (result["left_knee_flexion_deg"] - result["right_knee_flexion_deg"]).abs().max()
velocity = result["mean_knee_velocity_dps"].abs().max()

st.info(source_note)

a, b, c, d = st.columns(4)
a.metric("Screening score", f"{score}/100", band)
b.metric("Peak knee flexion", f"{peak:.1f}°")
c.metric("Max asymmetry", f"{asymmetry:.1f}°")
d.metric("Peak angular velocity", f"{velocity:.0f}°/s")

# Frame-by-Frame Inspector Section
st.divider()
st.subheader("🔍 Frame-by-Frame Inspector")
total_frames = len(result)
if total_frames > 0:
    selected_idx = st.slider("Scrub through frames (Time step)", 0, total_frames - 1, 0, format="Frame %d")
    
    col_img, col_metrics = st.columns([1, 1])
    
    with col_img:
        if annotated_frames and selected_idx < len(annotated_frames):
            st.image(annotated_frames[selected_idx], caption=f"Frame {selected_idx} Pose Overlay", use_container_width=True)
        elif preview is not None:
            st.image(preview, caption="MediaPipe lower-limb pose preview", use_container_width=True)
        else:
            st.info("Uploaded CSV or synthetic trial mode active (no raw video frames available).")
            
    with col_metrics:
        current_time = float(result.loc[selected_idx, "time_s"])
        l_flex = float(result.loc[selected_idx, "left_knee_flexion_deg"])
        r_flex = float(result.loc[selected_idx, "right_knee_flexion_deg"])
        inst_asymmetry = abs(l_flex - r_flex)
        inst_vel = float(result.loc[selected_idx, "mean_knee_velocity_dps"])
        l_offset = float(result.loc[selected_idx, "left_knee_ankle_offset_cm"])
        r_offset = float(result.loc[selected_idx, "right_knee_ankle_offset_cm"])
        
        st.markdown(f"### Frame `{selected_idx}` Metrics *(t = {current_time:.2f}s)*")
        m1, m2 = st.columns(2)
        m1.metric("Left Knee Flexion", f"{l_flex:.1f}°")
        m2.metric("Right Knee Flexion", f"{r_flex:.1f}°")
        
        m3, m4 = st.columns(2)
        m3.metric("Instant Asymmetry", f"{inst_asymmetry:.1f}°")
        m4.metric("Mean Velocity", f"{inst_vel:.0f}°/s")
        
        st.caption(f"Knee-Ankle Offset: Left {l_offset:+.1f} cm | Right {r_offset:+.1f} cm")

st.divider()
st.subheader("Knee Kinematics Over Time")
chart = result.melt(id_vars="time_s", value_vars=["left_knee_flexion_deg", "right_knee_flexion_deg"], var_name="Side", value_name="Knee flexion (°)")
fig = px.line(chart, x="time_s", y="Knee flexion (°)", color="Side", labels={"time_s": "Time (s)"})

if total_frames > 0:
    selected_time = float(result.loc[selected_idx, "time_s"])
    fig.add_vline(
        x=selected_time,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Frame {selected_idx} ({selected_time:.2f}s)",
        annotation_position="top left"
    )

st.plotly_chart(fig, use_container_width=True)

st.subheader("Movement-Screen Findings")
st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)
st.caption("Educational screening only. This tool does not diagnose injury risk; interpret results with a qualified clinician and validate against calibrated 3D capture.")

download = result.to_csv(index=False).encode("utf-8")
st.download_button("Download analysed CSV", data=download, file_name="motiondna_kinematics.csv", mime="text/csv")

with st.expander("CSV schema and pipeline notes"):
    st.code("time_s,left_hip_x,left_hip_y,left_hip_z,left_knee_x,...,right_ankle_z")
    st.write("Video upload uses MediaPipe Pose Landmarker locally and derives a 2D/depth-relative landmark stream. Use calibrated 3D coordinates in metres, or multi-view triangulation, for research-grade 3D joint analysis.")
