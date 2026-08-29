"""Command-line interface for the video analysis tool.

Usage:
    python cli.py process --video path/to/video.mp4 --index path/to/index.npz
    python cli.py query --index path/to/index.npz --query "a red car"
"""
from __future__ import annotations

import argparse
import os

from src.config import DEFAULT_BATCH_SIZE, DEFAULT_MODEL_NAME, DEFAULT_SAMPLE_FPS
from src.search import find_moments
from src.video_index import VideoIndex, build_video_index


def cmd_process(args: argparse.Namespace) -> None:
    thumbs_dir = args.thumbnails_dir or (os.path.splitext(args.index)[0] + "_thumbs")
    index = build_video_index(
        video_path=args.video,
        thumbnails_dir=thumbs_dir,
        sample_fps=args.fps,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    index.save(args.index)
    print(f"Indexed {len(index.timestamps)} frames from '{args.video}'")
    print(f"Saved index to '{args.index}' (thumbnails in '{thumbs_dir}')")


def cmd_query(args: argparse.Namespace) -> None:
    index = VideoIndex.load(args.index)
    moments = find_moments(
        index,
        args.query,
        std_multiplier=args.std_multiplier,
        min_score=args.min_score,
        max_gap_seconds=args.max_gap,
        top_k=args.top,
    )
    if not moments:
        print(f"No moments found matching: '{args.query}'")
        return
    print(f"Moments matching '{args.query}':")
    for i, moment in enumerate(moments, start=1):
        print(f"  {i}. {moment}  [{moment.thumbnail_path}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video analysis tool: search video content with natural language."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser(
        "process", help="Extract frames and build a searchable index for a video."
    )
    process_parser.add_argument("--video", required=True, help="Path to the input video file.")
    process_parser.add_argument("--index", required=True, help="Path to save the output index (.npz).")
    process_parser.add_argument("--fps", type=float, default=DEFAULT_SAMPLE_FPS, help="Frames per second to sample.")
    process_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HuggingFace CLIP model name.")
    process_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    process_parser.add_argument("--thumbnails-dir", default=None, help="Directory to store frame thumbnails.")
    process_parser.set_defaults(func=cmd_process)

    query_parser = subparsers.add_parser(
        "query", help="Search a previously built index with a natural language query."
    )
    query_parser.add_argument("--index", required=True, help="Path to a previously built index (.npz).")
    query_parser.add_argument("--query", required=True, help='Natural language query, e.g. "a red car".')
    query_parser.add_argument("--top", type=int, default=10, help="Maximum number of moments to return.")
    query_parser.add_argument("--std-multiplier", type=float, default=1.0, help="Higher = stricter matching.")
    query_parser.add_argument("--min-score", type=float, default=None, help="Override the automatic threshold.")
    query_parser.add_argument(
        "--max-gap", type=float, default=2.0, help="Max seconds gap to merge matches into one moment."
    )
    query_parser.set_defaults(func=cmd_query)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
