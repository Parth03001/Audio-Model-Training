@echo off
:: BSR Audio Classification — Windows Setup Script
:: Run this once after cloning the repo.
:: Double-click OR run from Command Prompt.

echo ============================================================
echo  BSR Audio Classification - Windows Setup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Download Python 3.10 from https://www.python.org/downloads/
    echo Make sure to CHECK "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/6] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/6] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [3/6] Installing PyTorch with CUDA 12.1...
echo       If your GPU uses CUDA 11.8, edit this line in setup_windows.bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [4/6] Installing audio libraries...
pip install librosa==0.10.1 soundfile==0.12.1 audioread==3.0.1

echo.
echo [5/6] Installing Label Studio...
pip install label-studio==1.9.1 label-studio-sdk==0.0.32

echo.
echo [6/6] Installing remaining dependencies...
pip install numpy pandas scikit-learn matplotlib seaborn tqdm pyyaml scipy joblib tensorboard audiomentations

echo.
echo ============================================================
echo  Verifying installation...
echo ============================================================
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
python -c "import librosa; print('librosa: OK')"
python -c "import label_studio_sdk; print('Label Studio SDK: OK')"

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  Next steps:
echo  1. Run start_label_studio.bat to launch Label Studio
echo  2. Follow docs\windows_setup_guide.md from Step 9
echo ============================================================
pause
