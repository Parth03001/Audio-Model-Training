@echo off
:: Runs the full training pipeline.
:: Activates venv and launches train.py — GPU used automatically if available.

call venv\Scripts\activate.bat

echo Checking GPU...
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND - training on CPU')"
echo.

python scripts\train.py %*

pause
