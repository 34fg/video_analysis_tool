"""Shared configuration and constants for the video analysis tool."""
from dataclasses import dataclass

# CLIP model used to embed frames and text queries. It is "open vocabulary": it can
# match arbitrary natural-language descriptions (colors, clothing, objects, animals...)
# rather than being limited to a fixed list of classes.
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"

DEFAULT_SAMPLE_FPS = 1.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_THUMB_MAX_SIZE = 320


@dataclass
class ProcessingConfig:
    sample_fps: float = DEFAULT_SAMPLE_FPS
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = DEFAULT_BATCH_SIZE
    thumb_max_size: int = DEFAULT_THUMB_MAX_SIZE
