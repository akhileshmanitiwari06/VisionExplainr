
# app.py - Streamlit frontend for VisionExplainr
import streamlit as st
from vision_explainr.core import explain_video
import tempfile, os

st.set_page_config("VisionExplainr", layout="wide")
st.title("VisionExplainr — Video Explanation Demo")
st.markdown("Upload a short video (<= 30s) or use sample. This demo uses MediaPipe + heuristics to explain actions.")

uploaded = st.file_uploader("Upload video (mp4/mov)", type=["mp4","mov","avi"])
use_llm = st.checkbox("Use LLM for polished explanations (requires OPENAI_API_KEY)", value=False)
fps = st.slider("Frames per second to process (lower -> faster)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
tts_lang = st.selectbox("TTS language", options=["en","hi"], index=0)

video_path = None
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.flush(); tmp.close()
    video_path = tmp.name
else:
    if st.button("Use example video"):
        example = os.path.join("example_inputs", "sample_video.mp4")
        if os.path.exists(example):
            video_path = example
        else:
            st.warning("No example video found in example_inputs/ - add sample_video.mp4 to test.")

if video_path:
    with st.spinner("Analyzing video..."):
        res = explain_video(video_path, fps=fps, use_llm=use_llm, tts_lang=tts_lang)
    st.success(f"Analysis done — {res['frames_count']} frames processed")
    st.subheader("Detected Events")
    for i,ev in enumerate(res['events']):
        st.markdown(f"**Event {i+1}:** {ev['label']} — {int((ev['end']-ev['start'])/1000)}s")
        st.write(ev['summary'])
    st.subheader("Explanations")
    for i,txt in enumerate(res['explanations']):
        st.write(f"{i+1}. {txt}")
    if res.get('tts'):
        st.audio(res['tts'], format='audio/mp3')
