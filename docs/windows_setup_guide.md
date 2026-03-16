# Full Setup Guide — Windows + GPU

Complete step-by-step for installing the entire BSR pipeline on Windows
with CUDA GPU acceleration.

---

## Step 1 — Check Your GPU

Open **Command Prompt** (Win+R → `cmd`) and run:
```cmd
nvidia-smi
```

Note down:
- **GPU name** (e.g. RTX 3060, RTX 4090)
- **CUDA Version** shown in top-right corner (e.g. 12.1)

If `nvidia-smi` is not recognised, download and install the latest
NVIDIA driver from: https://www.nvidia.com/Download/index.aspx

---

## Step 2 — Python Setup

### Option A — Plain Python 3.10 (recommended if starting fresh)

> Use exactly Python 3.10 — Label Studio and PyTorch both have best support here.

1. Download: https://www.python.org/downloads/release/python-31011/
   → Scroll down → **Windows installer (64-bit)**
2. Run installer
   - **CHECK** "Add Python to PATH" at the bottom before clicking Install
3. Verify:
```cmd
python --version
# Should print: Python 3.10.x
```

### Option B — Using miniconda / Anaconda (if already installed)

```powershell
# Create a dedicated conda env with Python 3.10
conda create -n bsr-audio-classification python=3.10 -y
conda activate bsr-audio-classification

# IMPORTANT: install setuptools first — prevents pkg_resources error
pip install --upgrade setuptools pip

# Fix sqlite3.dll issue (Label Studio needs this on Windows + conda)
conda install -c anaconda sqlite -y
```

---

## Step 3 — Create a Virtual Environment

Open **Command Prompt** in your project folder:
```cmd
cd C:\Users\YourName\Audio-Model-Training

python -m venv venv
venv\Scripts\activate
```

Your prompt will now show `(venv)` — always activate this before running anything.

---

## Step 4 — Install PyTorch WITH CUDA

Go to https://pytorch.org/get-started/locally/ and use the selector, OR
use the commands below based on your CUDA version from Step 1:

**CUDA 12.1 (most common for RTX 30/40 series):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8 (older cards, GTX 10/16/20 series):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU is working:**
```cmd
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: True
#               NVIDIA GeForce RTX XXXX
```

---

## Step 5 — Install Audio Libraries

On Windows, `librosa` needs `soundfile` which needs the `libsndfile` binary.
The easiest way is via `pip` — the wheel includes the DLL automatically:

```cmd
pip install librosa soundfile audioread numpy pandas scikit-learn matplotlib seaborn tqdm pyyaml scipy joblib
```

**Verify audio loading works:**
```cmd
python -c "import librosa; print('librosa OK')"
```

---

## Step 6 — Install Label Studio

```cmd
pip install label-studio==1.9.1
```

**Verify:**
```cmd
label-studio --version
```

---

## Step 7 — Install Remaining Project Dependencies

```cmd
pip install tensorboard audiomentations label-studio-sdk==0.0.32
```

> Skip `panns-inference` for now — we load the CNN14 weights manually
> (the pip package has dependency conflicts on Windows). The project handles this.

---

## Step 8 — Configure Label Studio for Local Audio Files

Label Studio needs permission to serve audio from your local disk.

1. Create a file at `C:\Users\YourName\.label-studio\label_studio.conf`
   (create the folder if it doesn't exist)

2. Add this content:
```
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:/
```

**Or set as environment variables before launching (easier):**
```cmd
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:/
label-studio start
```

> The `DOCUMENT_ROOT=C:/` lets Label Studio serve any WAV file from your C: drive.
> Without this, the audio player in the browser will show a blank waveform.

---

## Step 9 — First Launch of Label Studio

```cmd
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:/
label-studio start
```

- Browser opens at **http://localhost:8080**
- Create an account (local only, no internet needed)
- Get your API token: top-right avatar → **Account & Settings** → copy the token

---

## Step 10 — Run the Pre-Labeling Script

```cmd
python scripts\auto_prelabel.py --audio_dir test_data --output data\annotations\prelabels.json
```

---

## Step 11 — Import Pre-Annotations into Label Studio

```cmd
python scripts\setup_label_studio.py --token YOUR_TOKEN --audio_dir test_data --action setup
```

Then in the browser:
- Open the project
- Click **Import** → drag in `data\annotations\prelabels.json`
- Regions will appear pre-drawn on every file

---

## Step 12 — Annotate

Open any task. You will see the waveform with highlighted candidate regions.

Workflow per file (takes ~3-5 min each):
1. Press **Space** to play
2. Click each highlighted region → press hotkey to label:
   - `1` = OK (road bump, not a defect — delete this region)
   - `2` = NOK_BSR (rattle confirmed)
   - `3` = NOK_IP (instrument panel creak)
   - `4` = NOK_Sunroof (sunroof/glass)
   - `5` = Unsure
3. Also listen to sections WITHOUT regions — if you hear something, draw a new one
4. Set **overall_quality** at the top of the page
5. Click **Submit** → next file

---

## Step 13 — Export Annotations

```cmd
python scripts\setup_label_studio.py --token YOUR_TOKEN --action export
```

---

## Step 14 — Segment Data

```cmd
python scripts\segment_data.py --annotations data\annotations\annotations.csv --audio_dir test_data
```

---

## Step 15 — Train

```cmd
python scripts\train.py
```

GPU will be used automatically if Step 4 was done correctly.
You will see `Device: cuda` in the first line of output.

Monitor training in TensorBoard:
```cmd
tensorboard --logdir runs
# Open http://localhost:6006
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pkg_resources'`
This happens with conda environments. Fix:
```powershell
pip install --upgrade setuptools
```

### `Can't load sqlite3.dll from current directory`
Label Studio + conda on Windows. Two options:
```powershell
# Option 1 (preferred)
conda install -c anaconda sqlite -y

# Option 2 — copy DLL manually if option 1 doesn't help
copy "$env:USERPROFILE\AppData\Local\miniconda3\Library\bin\sqlite3.dll" `
     "$env:USERPROFILE\AppData\Local\miniconda3\envs\bsr-audio-classification\lib\site-packages\label_studio"
```

### Both errors at once — nuclear option
If both errors persist together, pin Label Studio to 1.8.0 which has
no sqlite conflict on conda:
```powershell
pip install label-studio==1.8.0
```

### `label-studio` not found after install
```cmd
# Add Python Scripts to PATH manually:
set PATH=%PATH%;C:\Users\YourName\AppData\Local\Programs\Python\Python310\Scripts
```

### Audio not playing in Label Studio (blank waveform)
- Make sure `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` is set
- The audio path in the task must use forward slashes: `C:/Users/...` not `C:\Users\...`

### `torch.cuda.is_available()` returns False
- Reinstall PyTorch with the correct CUDA version for your driver
- Run `nvidia-smi` and check the "CUDA Version" column in the top-right

### `ImportError: No module named 'soundfile'` on audio load
```cmd
pip install soundfile
```

### Slow training (model not using GPU)
Check at start of training script output — should print `Device: cuda`.
If it prints `Device: cpu`, the CUDA install failed — redo Step 4.

---

## Quick Reference — Daily Commands

```cmd
# Activate environment (always do this first)
cd C:\Users\YourName\Audio-Model-Training
venv\Scripts\activate

# Start Label Studio
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:/
label-studio start

# Run training
python scripts\train.py

# Run inference on new files
python scripts\predict.py --checkpoint checkpoints\best_model.pth --input new_recording.wav --plot
```
