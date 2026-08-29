"""Extract sampled frames (with timestamps) from a video file using OpenCV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class SampledFrame:
    index: int              # sequential index among sampled frames
    frame_number: int       # original frame number in the source video
    timestamp: float         # seconds from the start of the video
    image_bgr: np.ndarray    # OpenCV BGR image


def iter_sampled_frames(video_path: str, sample_fps: float = 1.0) -> Iterator[SampledFrame]:
    """Yield frames from `video_path` sampled at approximately `sample_fps` frames/sec.

    Sampling (instead of processing every frame) keeps CNN/CLIP inference tractable for
    long videos while still giving second-level precision on when things happen.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(video_fps / sample_fps))

    frame_number = 0
    sampled_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_number % step == 0:
                timestamp = frame_number / video_fps
                yield SampledFrame(
                    index=sampled_index,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    image_bgr=frame,
                )
                sampled_index += 1
            frame_number += 1
    finally:
        cap.release()


def get_video_duration(video_path: str) -> float:
    """Return the duration of `video_path` in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return frame_count / fps if fps else 0.0
