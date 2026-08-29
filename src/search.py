"""Semantic search over a VideoIndex: find timestamps matching a natural-language query."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .clip_model import get_shared_encoder
from .video_index import VideoIndex


@dataclass
class Moment:
    start_time: float
    end_time: float
    peak_time: float
    peak_score: float
    thumbnail_path: str

    @staticmethod
    def format_time(t: float) -> str:
        m, s = divmod(max(t, 0.0), 60)
        h, m = divmod(int(m), 60)
        return f"{h:02d}:{m:02d}:{s:05.2f}"

    def __str__(self) -> str:
        return (
            f"{self.format_time(self.start_time)} - {self.format_time(self.end_time)} "
            f"(peak at {self.format_time(self.peak_time)}, score={self.peak_score:.3f})"
        )


def score_query(index: VideoIndex, query: str, model_name: Optional[str] = None) -> np.ndarray:
    """Return the cosine similarity of `query` against every embedded frame in `index`."""
    encoder = get_shared_encoder(model_name or index.model_name)
    text_emb = encoder.encode_text(query)
    return index.embeddings @ text_emb


def find_moments(
    index: VideoIndex,
    query: str,
    std_multiplier: float = 1.0,
    min_score: Optional[float] = None,
    max_gap_seconds: float = 2.0,
    top_k: Optional[int] = 10,
) -> List[Moment]:
    """Find the moments in the video where `query` best matches the visual content.

    A frame counts as a match if its similarity score exceeds a threshold, computed from
    `min_score` (if given) or as mean + std_multiplier * std over all sampled frames.
    Consecutive matches within `max_gap_seconds` of each other are merged into a single
    Moment, whose "exact" instant is the highest-scoring frame within that group.
    """
    scores = score_query(index, query)
    threshold = (
        min_score if min_score is not None else float(scores.mean() + std_multiplier * scores.std())
    )

    matched_idx = np.where(scores >= threshold)[0]
    if len(matched_idx) == 0:
        return []

    groups: List[List[int]] = []
    current_group = [int(matched_idx[0])]
    for idx in matched_idx[1:]:
        idx = int(idx)
        gap = index.timestamps[idx] - index.timestamps[current_group[-1]]
        if gap <= max_gap_seconds:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)

    moments: List[Moment] = []
    for group in groups:
        group_scores = scores[group]
        peak_idx = group[int(np.argmax(group_scores))]
        moments.append(
            Moment(
                start_time=float(index.timestamps[group[0]]),
                end_time=float(index.timestamps[group[-1]]),
                peak_time=float(index.timestamps[peak_idx]),
                peak_score=float(scores[peak_idx]),
                thumbnail_path=index.thumbnail_paths[peak_idx],
            )
        )

    moments.sort(key=lambda m: m.peak_score, reverse=True)
    if top_k is not None:
        moments = moments[:top_k]
    return moments
