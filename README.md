# BSR Audio Classification

Binary OK/NOK classifier for vehicle BSR (Buzz, Squeak, Rattle) track testing.

**65 audio files × 4-5 min each → ~15,000 training clips → CNN14 fine-tune**

---

## Project Structure

```
Audio-Model-Training/
├── data/
│   ├── raw/                    ← Place your 65 audio files here (.wav/.mp3)
│   ├── annotations/            ← Label Studio exports land here
│   └── segments/               ← Auto-generated mel-spec clips + manifest.csv
├── scripts/
│   ├── setup_label_studio.py   ← Step 1: Create LS project & import files
│   ├── segment_data.py         ← Step 2: Slice files into training clips
│   ├── train.py                ← Step 3: Train CNN14 classifier
│   └── predict.py              ← Step 4: Run inference on new files
├── src/
│   ├── dataset/                ← AnnotationReader, AudioSegmenter, BSRDataset
│   ├── models/                 ← CNN14 backbone + BSRClassifier head
│   ├── augmentation/           ← SpecAugment, Mixup, WaveformAugmenter
│   └── utils/                  ← Metrics (F1, AUC, confusion matrix)
├── configs/config.yaml         ← All hyperparameters
└── checkpoints/                ← Saved model weights
```

---

## Why You Cannot Use Whole-File Labels

These audio files are **mixed recordings** — a file labeled "NOK" contains long
stretches of clean driving interrupted by short defect bursts. A file labeled "OK"
may contain brief road bumps or events that acoustically resemble BSR.

Labeling the entire 4-5 min file as OK/NOK and then segmenting into 2-sec clips
means the model gets thousands of clips with **wrong labels** — it will learn
per-file recording conditions (road surface, speed, microphone placement) instead
of the actual defect signature.

**Verification on the 10 sample files:**
| File | Folder label | Auto-detected anomalous events |
|------|-------------|-------------------------------|
| 74967_ok.wav | OK | **28 candidate events** (30s) |
| 74970_ok.wav | OK | **26 candidate events** (27s) |
| 78555_ok.wav | OK | **47 candidate events** (50s) |
| 78608_ok 2.wav | OK | **71 candidate events** (78s) |
| 77850_notok_ipNoise 9.wav | NOK | 105 events — but 159s is clean |

Those anomalous events in "OK" files could be genuine missed defects or road
events. Either way, labeling them "OK" based on the filename corrupts training.

---

## Step-by-Step Workflow

### 0. Install dependencies
```bash
pip install -r requirements.txt
```

### 1. Auto-detect candidate events (pre-labeling)

This scans every file and flags acoustically anomalous regions using
**per-file baseline normalisation** — so the file's own road noise becomes the
reference. Output is a Label Studio import file with pre-drawn regions.

```bash
python scripts/auto_prelabel.py \
    --audio_dir test_data \
    --output    data/annotations/prelabels.json \
    --threshold 3.5
```

### 2. Verify in Label Studio (human review)

**Start Label Studio:**
```bash
label-studio start
# Opens at http://localhost:8080
# Get API token: http://localhost:8080/user/account
```

**Create project and import pre-annotations:**
```bash
# Create project (first time only)
python scripts/setup_label_studio.py \
    --token YOUR_API_TOKEN \
    --audio_dir test_data \
    --action setup

# Then: Project → Import → select data/annotations/prelabels.json
# Regions are already drawn — you only verify/correct, ~3-5 hrs for 65 files
```

**Hotkeys in Label Studio:**
| Key | Label | Meaning |
|-----|-------|---------|
| `1` | OK | Clean — reject this candidate region |
| `2` | NOK_BSR | Buzz/squeak/rattle — confirm |
| `3` | NOK_IP | Instrument panel creak — confirm |
| `4` | NOK_Sunroof | Sunroof wind/rattle — confirm |
| `5` | Unsure | Edge case (used with low weight, not discarded) |

**Annotation rules:**
- For each pre-drawn region: listen, then label it OR delete it if it's a road bump
- Also check "OK" files — if a region sounds like a genuine rattle, label it NOK_BSR
- The overall_quality field (OK/NOK at file top) is auto-suggested — correct it if needed

**Export when done:**
```bash
python scripts/setup_label_studio.py \
    --token YOUR_API_TOKEN \
    --action export
```

### 2. Segment audio into training clips
```bash
python scripts/segment_data.py \
    --annotations data/annotations/annotations.csv \
    --audio_dir   data/raw
```

Expected output:
```
Segmentation complete:
  OK  clips : 10842
  NOK clips :  3201
  Manifest  : data/segments/manifest.csv
```

### 3. Train the model
```bash
# Default (CNN14, 25 epochs, config.yaml settings)
python scripts/train.py

# Override backbone or batch size
python scripts/train.py --backbone MobileNetV2 --batch_size 16

# Resume from checkpoint
python scripts/train.py --resume checkpoints/epoch_010.pth
```

Training output:
```
Phase 1: Training head only for 5 epochs
Epoch   1/25  train_loss=0.6821  train_f1=0.6102  │  val_loss=0.5941  val_f1=0.6820  val_auc=0.7441
...
Phase 2: Unfreezing backbone — lr=1e-05
Epoch   6/25  train_loss=0.4102  train_f1=0.8341  │  val_loss=0.3821  val_f1=0.8790  val_auc=0.9210
...
✓ New best val_f1=0.9102 → saved checkpoints/best_model.pth
```

Monitor with TensorBoard:
```bash
tensorboard --logdir runs/
```

### 4. Run inference on new recordings
```bash
# Single file
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --input      new_recording.wav \
    --plot

# Batch of files with timeline plots
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --input      data/new_recordings/ \
    --output     predictions/ \
    --plot
```

Output:
```
File                           Verdict  NOK%   MaxP
vehicle_run_001.wav            OK         2.1%  0.3241
vehicle_run_002.wav            NOK       18.4%  0.9821   ← rattle detected
vehicle_run_003.wav            OK         0.8%  0.2104
```

---

## Model Architecture

```
Input: (B, 1, 128, 87)          ← log-mel spectrogram: 128 mel bins × ~87 time frames
        ↓
CNN14 Backbone (PANNs)          ← pretrained on AudioSet (2M clips, 527 classes)
  6 × ConvBlock(Conv→BN→ReLU)
  Global avg+max pool
        ↓
Embedding: (B, 2048)
        ↓
Head: Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→1)
        ↓
Output: (B,) raw logit          ← sigmoid → P(NOK)
```

## Training Strategy

| Phase | Epochs | What trains | LR |
|-------|--------|-------------|-----|
| 1 | 1–5 | Head only | 1e-3 |
| 2 | 6–25 | Full model | backbone=1e-5, head=1e-3 |

**Loss:** Confidence-weighted BCEWithLogitsLoss
**Sampler:** WeightedRandomSampler (balances OK/NOK per batch)
**Augmentation:** SpecAugment (time+freq masking) + Mixup

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| PANNs CNN14 over scratch CNN | 65 files too few for scratch; CNN14 already understands mechanical transients |
| 2-sec clips with 1-sec overlap | Captures transient BSR events; overlap ensures no event is split at boundary |
| File-level group splits | Prevents data leakage from temporal correlation across clips of same file |
| Confidence-weighted loss | Uncertain labels ('Unsure') get down-weighted — doesn't hurt model |
| WeightedRandomSampler | BSR tracks are mostly quiet (OK); sampler prevents OK-class dominance |
