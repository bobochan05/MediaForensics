# Layer 2 README

## Overview

Layer 2 is the multimodal analysis and propagation-intelligence layer built on top of the Layer 1 detector.

Layer 1 answers: "Does this file look fake or real?"

Layer 2 answers the next set of questions:

- where else has this media or a close variant appeared
- how visually or acoustically similar are those appearances
- what context the media is being used in
- what the probable earliest accessible occurrence is
- how fast the media appears to be spreading
- how risky the current spread pattern looks

The Layer 2 pipeline combines local corpus search, external discovery, multimodal verification, context classification, propagation analysis, and risk scoring into one structured response.

## Scope

Layer 2 is responsible for:

- reusing or invoking Layer 1 detection
- generating robust visual embeddings for media
- extracting and embedding audio when present
- building and querying FAISS indexes
- local similarity retrieval
- external discovery and reverse-search style lookup
- context classification with NLP
- spread timeline construction
- origin estimation
- risk scoring
- per-candidate enrichment and analysis persistence

## Code Map

- [layer2_pipeline.py](/C:/Users/Armaa/Downloads/deepfake detector/layer2_pipeline.py): main orchestration pipeline
- [layer2_api.py](/C:/Users/Armaa/Downloads/deepfake detector/layer2_api.py): FastAPI service
- [layer2_schemas.py](/C:/Users/Armaa/Downloads/deepfake detector/layer2_schemas.py): response models
- [layer2_insights.py](/C:/Users/Armaa/Downloads/deepfake detector/layer2_insights.py): enrichment and interpretation helpers
- [similarity/embedding.py](/C:/Users/Armaa/Downloads/deepfake detector/similarity/embedding.py): visual embedding service
- [similarity/faiss_index.py](/C:/Users/Armaa/Downloads/deepfake detector/similarity/faiss_index.py): FAISS-backed vector index wrapper
- [similarity/search.py](/C:/Users/Armaa/Downloads/deepfake detector/similarity/search.py): multimodal local retrieval logic
- [audio/audio_extract.py](/C:/Users/Armaa/Downloads/deepfake detector/audio/audio_extract.py): audio extraction from media
- [audio/audio_features.py](/C:/Users/Armaa/Downloads/deepfake detector/audio/audio_features.py): FFT, Mel, and MFCC feature generation
- [audio/audio_embedding.py](/C:/Users/Armaa/Downloads/deepfake detector/audio/audio_embedding.py): wav2vec and CLAP audio embeddings
- [tracking/external_search.py](/C:/Users/Armaa/Downloads/deepfake detector/tracking/external_search.py): external occurrence search
- [tracking/metadata_parser.py](/C:/Users/Armaa/Downloads/deepfake detector/tracking/metadata_parser.py): timestamp normalization and metadata shaping
- [tracking/timeline.py](/C:/Users/Armaa/Downloads/deepfake detector/tracking/timeline.py): spread timeline and origin estimation
- [tracking/reverse_search_service.py](/C:/Users/Armaa/Downloads/deepfake detector/tracking/reverse_search_service.py): reverse-search helpers and public upload flow
- [tracking/media_resolver.py](/C:/Users/Armaa/Downloads/deepfake detector/tracking/media_resolver.py): candidate media resolution
- [tracking/reverse_image_providers.py](/C:/Users/Armaa/Downloads/deepfake detector/tracking/reverse_image_providers.py): provider-specific reverse-search integrations
- [nlp/context_classifier.py](/C:/Users/Armaa/Downloads/deepfake detector/nlp/context_classifier.py): context classification
- [risk/scoring.py](/C:/Users/Armaa/Downloads/deepfake detector/risk/scoring.py): risk assessment logic
- [inference.py](/C:/Users/Armaa/Downloads/deepfake detector/inference.py): Layer 1 fallback used when Layer 2 needs a detector verdict

## Layer 2 Runtime Architecture

The `Layer2Pipeline` object owns the main runtime dependencies:

- `VisualEmbeddingService`
- `AudioEmbeddingService`
- `ContextClassifier`
- `ExternalSearchClient`
- `FaissVectorIndex("visual")`
- `FaissVectorIndex("audio")`
- `MultimodalSimilaritySearch`

Default configuration from `Layer2Config`:

- `sample_fps = 0.5`
- `max_frames_per_video = 8`
- `max_seed_items = 512`
- `max_audio_seed_items = 32`
- `top_k = 20`
- `device = "auto"`

Directories created automatically under `artifacts/layer2/`:

- `uploads/`
- `analyses/`
- `cache/`
- `indexes/`

## End-to-End Flow

The `analyze_media()` flow in [layer2_pipeline.py](/C:/Users/Armaa/Downloads/deepfake detector/layer2_pipeline.py) is:

1. copy the uploaded or temporary file into `artifacts/layer2/uploads/`
2. generate a unique `analysis_id`
3. reuse Layer 1 verdict if `is_fake` and `confidence` were provided
4. otherwise run Layer 1 inference using [inference.py](/C:/Users/Armaa/Downloads/deepfake detector/inference.py)
5. compute a visual embedding for the media
6. extract audio from the media if possible
7. compute audio embeddings when audio exists
8. ensure local visual and audio indexes are available
9. query local multimodal similarity indexes unless `internet_only=True`
10. query external discovery sources
11. convert matches into normalized occurrence records
12. classify context for each occurrence
13. build a propagation timeline
14. estimate the earliest accessible origin
15. compute risk score and risk level
16. generate Layer 2 insight enrichment
17. persist the uploaded item back into local indexes
18. save the final JSON response under `artifacts/layer2/analyses/`

## Layer 1 Dependency Boundary

Layer 2 depends on Layer 1 in two ways:

- it can call Layer 1 inference directly if no verdict is supplied
- it seeds its initial visual corpus from Layer 1 cached CLIP embeddings when available

The Layer 1 model files Layer 2 expects are:

- `artifacts/fusion_model.pth`
- `artifacts/efficientnet_finetuned.pth`
- `artifacts/dino_finetuned.pth`

Layer 2 uses the same `sample_fps` default of `0.5`, so the analysis stage stays aligned with Layer 1 sampling behavior.

## Visual Similarity Subsystem

Implemented primarily in [similarity/embedding.py](/C:/Users/Armaa/Downloads/deepfake detector/similarity/embedding.py) and [similarity/search.py](/C:/Users/Armaa/Downloads/deepfake detector/similarity/search.py).

Responsibilities:

- sample representative frames from video
- produce one robust media-level visual embedding
- search a FAISS index of prior items
- return scored local matches

Seeding logic:

1. try `artifacts/embeddings/train/clip_embeddings.npy` and `artifacts/embeddings/test/clip_embeddings.npy`
2. group frame embeddings by source file
3. average frame embeddings into one vector per file
4. cap to `max_seed_items`
5. attach metadata per indexed item

Fallback seeding:

- if Layer 1 embedding caches are not present, scan `dataset/` directly and embed a capped subset of files

Stored metadata includes:

- `entry_id`
- source type
- platform
- local path
- optional timestamp
- title/caption
- label
- credibility score
- context defaults
- provenance metadata

## Audio Subsystem

Implemented across:

- [audio/audio_extract.py](/C:/Users/Armaa/Downloads/deepfake detector/audio/audio_extract.py)
- [audio/audio_features.py](/C:/Users/Armaa/Downloads/deepfake detector/audio/audio_features.py)
- [audio/audio_embedding.py](/C:/Users/Armaa/Downloads/deepfake detector/audio/audio_embedding.py)

Audio responsibilities:

- detect whether a video contains audio
- extract waveform and sample rate
- compute engineered features such as:
  - FFT bins
  - Mel spectrogram
  - MFCC
- compute learned embeddings such as:
  - wav2vec 2.0
  - CLAP
- combine audio representation for retrieval and verification

Audio index behavior:

- built lazily from the visual index metadata
- only seeded from local video entries with usable audio
- capped by `max_audio_seed_items`
- persisted as a separate FAISS index

If a file has no audio, Layer 2 still works in visual-only mode.

## Local Multimodal Search

The `MultimodalSimilaritySearch` engine fuses:

- visual similarity from the visual FAISS index
- audio similarity from the audio FAISS index when available

Local matches are normalized into occurrence records and later enriched with:

- context label
- context score distribution
- timestamps
- credibility
- fused similarity values

This gives Layer 2 a private corpus-based memory before it looks outward to the web.

## External Discovery

Implemented mainly in [tracking/external_search.py](/C:/Users/Armaa/Downloads/deepfake detector/tracking/external_search.py) and the reverse-search helpers under `tracking/`.

External search responsibilities:

- accept optional `query_hint`
- use optional `source_url`
- search public sources when possible
- resolve candidate pages and media
- score and verify candidates against the original upload

The current README-safe summary of behavior from the codebase:

- DuckDuckGo HTML search can be used when available
- Reddit public JSON search can be used when available
- reverse-search style helpers can resolve candidate media URLs
- if live search is limited or disabled, the system can fall back to deterministic mocked results when allowed

Important runtime difference:

- `layer2_api.py` default upload flow allows the standard external search path
- the Flask `POST /api/discover` route in [app.py](/C:/Users/Armaa/Downloads/deepfake detector/app.py) calls Layer 2 with `internet_only=True` and `allow_mock_fallback=False`

That means the Flask discovery route intentionally hides local-only results and tries to show public-web discovery only.

## Context Classification

Implemented in [nlp/context_classifier.py](/C:/Users/Armaa/Downloads/deepfake detector/nlp/context_classifier.py).

For each occurrence, Layer 2 builds text from:

- title
- caption
- platform

The classifier assigns:

- `news`
- `meme`
- `propaganda / misinformation`

It also attaches class score distributions to each result.

## Timeline And Origin Analysis

Implemented in [tracking/timeline.py](/C:/Users/Armaa/Downloads/deepfake detector/tracking/timeline.py).

The occurrence list is converted into:

- a chronological spread timeline
- an origin estimate

Timeline points include:

- `timestamp`
- `count`
- `cumulative_count`
- `velocity`
- `spike_score`

Origin estimate fields include:

- `timestamp`
- `source`
- `url`
- `platform`
- `note`

Current behavior uses the earliest accessible occurrence as a practical proxy for probable origin.

## Risk Scoring

Implemented in [risk/scoring.py](/C:/Users/Armaa/Downloads/deepfake detector/risk/scoring.py).

Risk is computed from a mix of:

- Layer 1 fake probability
- number of occurrences
- spread velocity
- source credibility
- misuse probability from context
- visual similarity evidence
- audio similarity evidence

The structured `RiskBreakdown` model includes:

- `fake_probability`
- `occurrence_score`
- `spread_velocity_score`
- `credibility_risk`
- `misuse_probability`
- `visual_similarity_score`
- `audio_similarity_score`
- `final_score`

The final API response exposes:

- `risk_score`
- `risk_level`
- `confidence_explanation`

## Layer 2 Insights Enrichment

Implemented in [layer2_insights.py](/C:/Users/Armaa/Downloads/deepfake detector/layer2_insights.py).

This post-processing layer adds:

- platform distribution across `news`, `social`, `video`, and `other`
- temporal anomaly detection
- cross-modal consistency checks between visual and audio similarity
- per-candidate enrichment

Per-candidate enrichment includes:

- source credibility
- match reasons such as high visual similarity, pHash hints, or title relevance
- simple mutation heuristics:
  - `crop`
  - `resized`
  - `color_adjusted`
  - `none`

## API

Layer 2 exposes a FastAPI service from [layer2_api.py](/C:/Users/Armaa/Downloads/deepfake detector/layer2_api.py).

Run it with:

```powershell
cd "C:\Users\Armaa\Downloads\deepfake detector"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m uvicorn layer2_api:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### `GET /`

Returns service metadata including:

- service name
- visual index size
- audio index size
- available endpoints

### `POST /upload`

Multipart form fields:

- `file`: required
- `query_hint`: optional
- `source_url`: optional
- `is_fake`: optional
- `confidence`: optional

Behavior:

- if `is_fake` and `confidence` are provided, Layer 2 reuses them
- otherwise it runs Layer 1 inference first
- returns the full `AnalysisResponse`

### `GET /similar/{analysis_id}`

Returns only the `similar_content` array for a previously saved analysis.

### `GET /analysis/{analysis_id}`

Returns the full persisted `AnalysisResponse`.

## Response Schema

The main response model in [layer2_schemas.py](/C:/Users/Armaa/Downloads/deepfake detector/layer2_schemas.py) includes:

- `analysis_id`
- `filename`
- `media_type`
- `is_fake`
- `confidence`
- `similar_content`
- `matches`
- `origin_estimate`
- `spread_timeline`
- `risk_score`
- `risk_level`
- `risk_breakdown`
- `confidence_explanation`
- `visual_embedding_dim`
- `audio_embedding_dim`
- `external_matches_used`
- `created_at`
- `layer2_insights`

Each `SimilarContentItem` can include:

- source URL
- media URL
- local path
- visual similarity
- audio similarity
- fused similarity
- platform
- timestamp
- source type
- title
- caption
- context
- context scores
- credibility score
- label
- metadata

## Storage Layout

Layer 2 writes under `artifacts/layer2/`:

- `uploads/`: stored analysis inputs
- `analyses/`: saved JSON responses by `analysis_id`
- `cache/`: search and extraction cache data
- `indexes/`: FAISS indexes and metadata

Expected index files:

- `visual.index`
- `visual_metadata.json`
- `audio.index`
- `audio_metadata.json`

## Integration With Flask App

The user-facing Flask app in [app.py](/C:/Users/Armaa/Downloads/deepfake detector/app.py) uses Layer 2 in its discovery stage.

The two-step UI path is:

1. `POST /api/predict`
   Layer 1 produces `upload_id`, label, confidence, and reasoning.
2. `POST /api/discover`
   Layer 2 performs internet-first discovery, origin estimation, timeline analysis, and risk scoring for that prior upload.

The Flask route also:

- tries to generate a public source URL for reverse-image search
- rebuilds discovery sections such as exact matches and exploratory leads
- intentionally excludes local laptop/internal corpus results from the final web-discovery sections shown to the user

## Operational Notes

- Layer 2 can work with images or videos.
- Audio analysis is opportunistic and activates only when usable audio is available.
- First use may trigger model downloads for CLIP, DistilBERT, wav2vec, or CLAP if those assets are not cached locally.
- Audio extraction prefers `ffmpeg` when available and falls back to local decoding paths.
- Local indexes become more useful over time because analyzed uploads are written back into the index after each run.
- The quality of external discovery depends heavily on public availability, resolvable media URLs, and provider restrictions.
