# Video Analysis Tool

Upload/point to a video and ask natural-language questions like *"when is a car visible"*,
*"when is a bird here"*, *"when is a woman in a red dress visible"*, or *"when is a gray
computer visible"* — and get back the exact timestamps.

## How it works

1. **Frame sampling** — [src/frame_extractor.py](src/frame_extractor.py) reads the video with
   OpenCV and samples frames at a configurable rate (default 1 frame/sec), keeping the timestamp
   of each frame.
2. **CNN/vision-language embedding** — [src/clip_model.py](src/clip_model.py) runs each sampled
   frame through OpenAI's **CLIP** model (`openai/clip-vit-base-patch32`). CLIP is trained to map
   images and text descriptions into the same vector space, so it can match *any* free-text
   description (not just a fixed list of classes like "car"/"bird") — including attributes like
   color and clothing (e.g. "a woman in a red dress").
3. **Indexing** — [src/video_index.py](src/video_index.py) stores all frame embeddings, timestamps,
   and small thumbnail images in a single `.npz` index file so a video only needs to be processed
   once.
4. **Semantic search** — [src/search.py](src/search.py) embeds your text query with the same CLIP
   model, scores every frame by cosine similarity, keeps the frames that score above an automatic
   threshold, and groups consecutive matching frames into "moments" (start time, end time, and the
   single highest-scoring "exact" instant).

## Setup

```powershell
cd video_analysis_tool
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> The first run downloads the CLIP model (~600 MB) from Hugging Face and caches it locally.
> A GPU (CUDA) is used automatically if available; otherwise it runs on CPU (slower but works).

## Usage

### Option A: Web UI (recommended)

```powershell
streamlit run app.py
```

Upload a video in the sidebar, click **Process video**, then type a query such as
`a car`, `a bird`, `a woman in a red dress`, or `a gray computer`. Each result shows a
thumbnail, the time range it occurs in, and the exact peak timestamp, with a button to
seek the video player to that moment.

### Option B: Command line

```powershell
# 1. Process a video once (extracts frames + builds the searchable index)
python cli.py process --video path\to\video.mp4 --index path\to\video_index.npz

# 2. Ask as many questions as you want against the saved index
python cli.py query --index path\to\video_index.npz --query "a red car"
python cli.py query --index path\to\video_index.npz --query "a bird"
python cli.py query --index path\to\video_index.npz --query "a woman in a red dress"
```

Useful flags for `query`:
- `--top N` — max number of moments returned (default 10)
- `--std-multiplier` — raise for stricter/fewer matches, lower for more matches (default 1.0)
- `--max-gap` — seconds allowed between matching frames before they're treated as separate
  moments (default 2.0)
- `--min-score` — bypass automatic thresholding and use a fixed similarity cutoff

Useful flags for `process`:
- `--fps` — sampling rate; higher gives more precise timestamps but takes longer to process
  (default 1.0)

## Notes & limitations

- Detection quality depends on CLIP's general visual knowledge; very small, occluded, or
  unusual objects may be missed. Try rephrasing the query or lowering `--std-multiplier` /
  the sensitivity slider if you expect a match that isn't showing up.
- Precision is limited by the sampling rate — with the default 1 fps, timestamps are accurate
  to about a second. Increase `--fps` for finer precision at the cost of processing time.
- This project does not require any cloud API keys; everything runs locally.
