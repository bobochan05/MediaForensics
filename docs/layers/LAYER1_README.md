# Layer 1 README

## Overview

Layer 1 is the core deepfake detection engine in this repository. It trains and serves a feature-level fusion classifier that operates on both images and videos.

The current Layer 1 stack combines four signals:

- CLIP image embeddings
- DINOv2 image embeddings
- EfficientNet-B0 image embeddings
- FFT-derived frequency inputs

Images are treated as single-frame inputs. Videos are sampled into frames and classified by averaging frame-level fake probabilities into one file-level decision.

## Scope

Layer 1 is responsible for:

- dataset loading and label assignment
- train/test splitting with leakage protection
- image and video frame extraction
- EfficientNet fine-tuning
- DINO fine-tuning
- CLIP feature extraction
- FFT feature generation
- fusion-model training
- cached embedding reuse
- single-file inference
- Flask-based upload and prediction API

Layer 1 does not do propagation tracking, reverse search, timeline analysis, or risk scoring. Those belong to Layer 2.

## Code Map

- [train.py](/C:/Users/Armaa/Downloads/deepfake detector/train.py): end-to-end Layer 1 training entry point
- [inference.py](/C:/Users/Armaa/Downloads/deepfake detector/inference.py): single image/video inference
- [evaluate.py](/C:/Users/Armaa/Downloads/deepfake detector/evaluate.py): holdout-set evaluation from saved artifacts
- [app.py](/C:/Users/Armaa/Downloads/deepfake detector/app.py): Flask UI and API for upload-based detection
- [data_loader.py](/C:/Users/Armaa/Downloads/deepfake detector/data_loader.py): dataset scanning and label mapping
- [video_split.py](/C:/Users/Armaa/Downloads/deepfake detector/video_split.py): stratified split and overlap checks
- [frame_extractor.py](/C:/Users/Armaa/Downloads/deepfake detector/frame_extractor.py): image loading and video frame sampling
- [models/clip_model.py](/C:/Users/Armaa/Downloads/deepfake detector/models/clip_model.py): CLIP branch
- [models/dino_model.py](/C:/Users/Armaa/Downloads/deepfake detector/models/dino_model.py): DINOv2 branch
- [models/efficientnet_model.py](/C:/Users/Armaa/Downloads/deepfake detector/models/efficientnet_model.py): EfficientNet branch
- [models/fft_model.py](/C:/Users/Armaa/Downloads/deepfake detector/models/fft_model.py): FFT preprocessing and frequency branch inputs
- [models/fusion_model.py](/C:/Users/Armaa/Downloads/deepfake detector/models/fusion_model.py): fusion classifier definition and inference helpers
- [utils.py](/C:/Users/Armaa/Downloads/deepfake detector/utils.py): shared utilities and label mapping

## Data Contract

Supported dataset layouts:

```text
dataset/
  original_sequences/
  manipulated_sequences/
```

or:

```text
dataset/
  real/
  fake/
```

Label mapping used by the pipeline:

- `real` or `original_sequences` -> `0`
- `fake` or `manipulated_sequences` -> `1`

Prediction names:

- `0` -> `real`
- `1` -> `fake`

Supported media types:

- `.jpg`
- `.jpeg`
- `.png`
- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`

## Detection Architecture

### CLIP branch

Implemented in [models/clip_model.py](/C:/Users/Armaa/Downloads/deepfake detector/models/clip_model.py).

- backbone: `openai/clip-vit-base-patch32`
- source library: Hugging Face `transformers`
- role: frozen semantic visual encoder
- embedding size: `512`

This branch is used for image/frame embeddings, not prompt-based zero-shot scoring.

### EfficientNet branch

Implemented in [models/efficientnet_model.py](/C:/Users/Armaa/Downloads/deepfake detector/models/efficientnet_model.py).

- backbone: `efficientnet_b0`
- pretrained weights: ImageNet
- feature size: `1280`
- fine-tuning: last two feature blocks are unfrozen
- temporary training head: `FeatureProbe`
- saved artifact: `artifacts/efficientnet_finetuned.pth`

### DINO branch

Implemented in [models/dino_model.py](/C:/Users/Armaa/Downloads/deepfake detector/models/dino_model.py).

- backbone: `dinov2_vits14`
- source: `facebookresearch/dinov2`
- feature size: `384`
- training schedule:
  - warmup with backbone frozen
  - later unfreeze final transformer block and normalization layer
- saved artifact: `artifacts/dino_finetuned.pth`

### FFT branch

Implemented in [models/fft_model.py](/C:/Users/Armaa/Downloads/deepfake detector/models/fft_model.py).

Input processing:

- convert frame to grayscale
- resize to `224 x 224`
- run 2D FFT
- compute magnitude spectrum
- center with `fftshift`
- min/max normalize per sample

The learned FFT branch contributes `128` features to the fusion stage.

### Fusion classifier

Implemented in [models/fusion_model.py](/C:/Users/Armaa/Downloads/deepfake detector/models/fusion_model.py).

Input dimensions:

- CLIP: `512`
- DINO: `384`
- EfficientNet: `1280`
- FFT: `128`

Derived dimensions:

- combined branch feature dimension before FFT concat: `2176`
- total fusion input dimension: `2304`

Classifier topology:

- Linear `2304 -> 512`
- ReLU
- Dropout `0.3`
- Linear `512 -> 256`
- ReLU
- Linear `256 -> 1`

The final output is one fake logit. Sigmoid is applied at prediction time to obtain fake probability.

## End-to-End Training Pipeline

The main workflow in [train.py](/C:/Users/Armaa/Downloads/deepfake detector/train.py) is:

1. Scan the dataset and collect media records.
2. Balance real and fake items to the smaller class count.
3. Perform a stratified train/test split.
4. Reject video-id overlap between train and test.
5. Extract frames from images and videos.
6. Fine-tune EfficientNet with a binary probe head.
7. Fine-tune DINO with a binary probe head and staged unfreezing.
8. Extract CLIP, DINO, EfficientNet, and FFT inputs for every frame.
9. Cache split embeddings under `artifacts/embeddings/`.
10. Train the fusion classifier on frame-level labels.
11. Aggregate frame probabilities into file-level predictions.
12. Save weights, metrics, manifests, and prediction outputs.

## Frame Extraction Behavior

Implemented in [frame_extractor.py](/C:/Users/Armaa/Downloads/deepfake detector/frame_extractor.py).

Current behavior:

- default image size: `224`
- default sampling rate: `0.5 FPS`
- optional override: `--frames_per_video`

Rules:

- image input -> one resized frame
- video input with `--frames_per_video` -> evenly spaced frames
- video input without fixed count -> FPS-based sampling
- unavailable FPS metadata -> fallback to up to 10 evenly spaced frames

## Training Defaults

Actual defaults from `parse_args()` in [train.py](/C:/Users/Armaa/Downloads/deepfake detector/train.py):

- `--batch_size 32`
- `--test_size 0.2`
- `--random_state 42`
- `--sample_fps 0.5`
- `--frames_per_video None`
- `--image_size 224`
- `--epochs 15`
- `--learning_rate 1e-3`
- `--backbone_epochs 15`
- `--backbone_learning_rate 5e-5`
- `--weight_decay 1e-3`
- `--label_smoothing 0.05`
- `--gradient_clip_norm 1.0`
- `--device cuda`

## Optimization Details

Layer 1 is CUDA-first.

- training defaults to `cuda`
- if `cuda` is requested but unavailable, training raises an error instead of silently switching to CPU
- inference defaults to `auto`

Torch runtime behavior:

- `torch.set_float32_matmul_precision("high")`
- cuDNN benchmark mode on CUDA
- TF32 matmul enabled when available
- TF32 cuDNN enabled when available
- pinned `DataLoader` memory on CUDA
- non-blocking GPU transfers
- `optimizer.zero_grad(set_to_none=True)` in the backbone fine-tuning loop

Training details:

- branch fine-tuning uses `AdamW`
- fusion training uses `Adam`
- branch and fusion heads use binary cross-entropy with logits
- label smoothing is applied to binary labels
- branch fine-tuning uses gradient clipping
- class rebalancing inside loaders uses `WeightedRandomSampler`

## Cached Embedding Layout

Each split is written under:

- `artifacts/embeddings/train/`
- `artifacts/embeddings/test/`

Per-split files:

- `clip_embeddings.npy`
- `dino_embeddings.npy`
- `efficientnet_embeddings.npy`
- `frequency_inputs.npy`
- `labels.npy`
- `frame_video_ids.json`
- `frame_video_paths.json`
- `metadata.json`

`metadata.json` stores:

- fusion mode
- sampling parameters
- image size
- frame counts
- EfficientNet weight path and mtime
- DINO weight path and mtime

If `--reuse_embeddings` is passed and the cached metadata matches the current run, feature extraction is skipped and the cached arrays are reused.

## Inference Pipeline

Implemented in [inference.py](/C:/Users/Armaa/Downloads/deepfake detector/inference.py).

For one input file, Layer 1:

1. resolves the runtime device
2. loads CLIP, EfficientNet, DINO, and the fusion classifier
3. extracts image or video frames
4. computes CLIP embeddings
5. computes DINO embeddings
6. computes EfficientNet embeddings
7. computes FFT inputs
8. predicts frame-level probabilities
9. averages fake probability across frames
10. returns:
   - `real` or `fake`
   - confidence

Decision logic:

- fake probability `>= 0.5` -> `fake`
- fake probability `< 0.5` -> `real`
- confidence is distance from the decision boundary in the predicted class direction

## Evaluation

Implemented in [evaluate.py](/C:/Users/Armaa/Downloads/deepfake detector/evaluate.py).

Evaluation uses saved artifacts. It does not retrain.

It:

- loads cached split embeddings
- loads the saved fusion model
- recomputes probabilities
- aggregates frame predictions to file level
- prints metrics
- writes `<split>_evaluation.json`

Reported metrics are file-level metrics after frame aggregation:

- accuracy
- confusion matrix
- precision
- recall
- F1-score
- support

## Layer 1 Flask App

Layer 1 also exposes a Flask-based user-facing app from [app.py](/C:/Users/Armaa/Downloads/deepfake detector/app.py).

Key routes:

- `GET /`
- `GET /minimal`
- `GET /health`
- `POST /api/predict`
- `POST /api/discover`
- `POST /reverse-search`

Layer 1 responsibilities inside the Flask app:

- validate required model files
- accept image/video uploads
- store temporary upload files
- call [inference.py](/C:/Users/Armaa/Downloads/deepfake detector/inference.py) in a subprocess
- return detector label and confidence
- generate human-readable reasoning text around the Layer 1 verdict

Important detail:

- `POST /api/predict` is the direct Layer 1 classification step
- `POST /api/discover` bridges into Layer 2 internet-first discovery using the saved Layer 1 result

## Saved Artifacts

Main Layer 1 artifacts under `artifacts/`:

- `fusion_model.pth`
- `efficientnet_finetuned.pth`
- `dino_finetuned.pth`
- `metrics.json`
- `split_manifest.json`
- `train_video_predictions.json`
- `test_video_predictions.json`
- `train_evaluation.json`
- `test_evaluation.json`

UI-related temporary outputs:

- `artifacts/ui_uploads/`

## Installation

Recommended Windows PowerShell setup:

```powershell
cd "C:\Users\Armaa\Downloads\deepfake detector"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For GPU acceleration, install the CUDA-enabled PyTorch build into the same `.venv`.

## Commands

### Train Layer 1

```powershell
cd "C:\Users\Armaa\Downloads\deepfake detector"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe train.py --dataset_dir "dataset" --output_dir "artifacts"
```

### Evaluate Holdout Split

```powershell
.\.venv\Scripts\python.exe evaluate.py --artifacts_dir "artifacts" --split test
```

### Run Inference On One File

```powershell
.\.venv\Scripts\python.exe inference.py --input_path "dataset\original_sequences\ACTUAL_REAL_IMAGE.jpg" --classifier_path "artifacts\fusion_model.pth" --efficientnet_path "artifacts\efficientnet_finetuned.pth" --dino_path "artifacts\dino_finetuned.pth"
```

### Run Flask App

```powershell
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Operational Notes

- The current implementation uses feature-level fusion, not score-only fusion.
- The holdout result in this repository is the train/test split test set, not a separate external benchmark.
- First run can be slower because pretrained weights may be loaded or downloaded if not already cached locally.
- The app sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so fully offline execution depends on the models already being present in local caches.
