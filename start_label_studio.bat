@echo off
:: Launches Label Studio with local file serving enabled.
:: Run this every time you want to annotate.

call venv\Scripts\activate.bat

set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%~dp0

echo Starting Label Studio at http://localhost:8080
echo Local file root: %~dp0
echo.
echo To get your API token:
echo   1. Open http://localhost:8080
echo   2. Click your avatar (top-right)
echo   3. Account ^& Settings ^> copy the token
echo.
echo After getting your token, run:
echo   python scripts\setup_label_studio.py --token YOUR_TOKEN --audio_dir %~dp0audio_dataset --action setup
echo.

label-studio start
