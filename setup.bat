@echo off
REM One-command setup for Windows.   setup.bat [--train]
setlocal enabledelayedexpansion
cd /d "%~dp0"
set REQ=requirements.txt
set DOCARGS=
if "%1"=="--train" (set REQ=requirements-train.txt& set DOCARGS=--train)

set PY=
for %%V in (3.11 3.12 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set PY=py -%%V
  )
)
if not defined PY (
  python -c "import sys;assert (3,10)<=sys.version_info[:2]<=(3,12)" >nul 2>&1 && set PY=python
)

if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys;assert (3,10)<=sys.version_info[:2]<=(3,12)" >nul 2>&1 || rmdir /s /q .venv
)
if not exist .venv\Scripts\python.exe (
  if defined PY (
    echo Using !PY!
    !PY! -m venv .venv || goto :fail
  ) else (
    where uv >nul 2>&1 && (
      echo No Python 3.10-3.12 found; uv will download Python 3.11
      uv venv --python 3.11 --seed .venv || goto :fail
    ) || (
      echo No Python 3.10-3.12 found. Python 3.13+ is not supported by mediapipe 0.10.21.
      echo Install Python 3.11 from https://www.python.org/downloads/ ^(tick "Add to PATH"^) and re-run setup.bat
      goto :fail
    )
  )
)

.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
echo Installing %REQ% (pinned with constraints.txt)...
.venv\Scripts\python.exe -m pip install -r %REQ% -c constraints.txt || (
  echo Exact transitive pins unavailable; retrying with direct pins only...
  .venv\Scripts\python.exe -m pip install -r %REQ% || goto :fail
)
if not exist models\hand_landmarker.task .venv\Scripts\python.exe scripts\download_models.py
if not exist models\pose_landmarker_lite.task .venv\Scripts\python.exe scripts\download_models.py
.venv\Scripts\python.exe scripts\doctor.py %DOCARGS% || goto :fail
echo.
echo Done. Activate with:  .venv\Scripts\activate
echo Then:  python predict.py path\to\clip.mp4    or    python demo.py
exit /b 0
:fail
echo Setup failed. See the message above.
exit /b 1
