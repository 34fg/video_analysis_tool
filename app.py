"""Streamlit UI for the video analysis tool.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile

import streamlit as st

from src.config import DEFAULT_SAMPLE_FPS
from src.search import find_moments
from src.video_index import build_video_index

st.set_page_config(page_title="Video Analysis Tool", layout="wide")
st.title("Video Analysis Tool")
st.caption(
    "Upload a video, then ask for any moment: \"a car\", \"a bird\", "
    "\"a woman in a red dress\", \"a gray computer\"..."
)

WORKDIR = os.path.join(tempfile.gettempdir(), "video_analysis_tool")
os.makedirs(WORKDIR, exist_ok=True)

if "index" not in st.session_state:
    st.session_state.index = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

with st.sidebar:
    st.header("1. Load a video")
    uploaded = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv", "webm"])
    sample_fps = st.slider(
        "Sampling rate (frames/sec)", min_value=0.2, max_value=5.0, value=DEFAULT_SAMPLE_FPS, step=0.2
    )
    process_clicked = st.button("Process video", type="primary", disabled=uploaded is None)

if uploaded is not None and process_clicked:
    video_path = os.path.join(WORKDIR, uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    thumbs_dir = os.path.join(WORKDIR, os.path.splitext(uploaded.name)[0] + "_thumbs")
    with st.spinner("Extracting frames and computing embeddings... this can take a while for long videos."):
        index = build_video_index(
            video_path=video_path, thumbnails_dir=thumbs_dir, sample_fps=sample_fps, progress=False
        )
    st.session_state.index = index
    st.session_state.video_path = video_path
    st.session_state.pop("seek_time", None)
    st.success(f"Processed {len(index.timestamps)} frames.")

if st.session_state.index is not None:
    st.header("2. Ask what to find")
    query = st.text_input(
        "What do you want to find in the video?",
        placeholder="e.g. a red car, a bird, a woman in a red dress, a gray computer",
    )
    col_a, col_b, col_c = st.columns(3)
    top_k = col_a.number_input("Max results", min_value=1, max_value=50, value=10)
    std_multiplier = col_b.slider("Sensitivity (lower = more results)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    max_gap = col_c.slider("Merge gap (seconds)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

    if query:
        moments = find_moments(
            st.session_state.index,
            query,
            std_multiplier=std_multiplier,
            max_gap_seconds=max_gap,
            top_k=int(top_k),
        )
        if not moments:
            st.warning("No matching moments found. Try lowering the sensitivity.")
        else:
            st.subheader(f'Found {len(moments)} moment(s) for: "{query}"')
            for i, moment in enumerate(moments, start=1):
                with st.container(border=True):
                    cols = st.columns([1, 3])
                    cols[0].image(moment.thumbnail_path, use_container_width=True)
                    with cols[1]:
                        st.markdown(f"**Moment {i}** — score `{moment.peak_score:.3f}`")
                        st.write(
                            f"Occurs between **{moment.format_time(moment.start_time)}** and "
                            f"**{moment.format_time(moment.end_time)}**"
                        )
                        st.write(f"Exact moment: **{moment.format_time(moment.peak_time)}**")
                        if st.button(f"Jump to {moment.peak_time:.1f}s", key=f"jump_{i}"):
                            st.session_state["seek_time"] = moment.peak_time

            st.header("3. Preview")
            seek_time = st.session_state.get("seek_time", moments[0].peak_time)
            st.video(st.session_state.video_path, start_time=seek_time)
else:
    st.info("Upload and process a video from the sidebar to get started.")
