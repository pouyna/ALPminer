@echo off
REM ==================================================================
REM  ALPminer -- one-click GUI launcher (Windows)
REM
REM  Double-click this file, or run it from a terminal in the code
REM  folder. It reuses the existing virtual environment when it is
REM  healthy and rebuilds it automatically when it is missing or
REM  broken (e.g. after copying the folder to another computer or
REM  upgrading Python), then starts the GUI server.
REM
REM  Keep this window OPEN while you use the app. Close it, or press
REM  Ctrl-C, to stop the server.
REM ==================================================================
setlocal
cd /d "%~dp0"

REM reuse the venv when it works; rebuild only when missing or broken
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import alpminer.schema" >nul 2>&1 && goto venv_ok
  echo Existing environment is broken or from another machine; rebuilding...
)
echo.
echo === Creating a clean virtual environment (.venv) ===
py -3 -m venv --clear .venv || (echo. & echo *** venv creation failed -- see above *** & pause & exit /b 1)

:venv_ok
echo.
echo === Activating the environment ===
call ".venv\Scripts\activate.bat" || (echo. & echo *** activation failed -- see above *** & pause & exit /b 1)

echo.
echo === Installing/refreshing ALPminer (fast when unchanged) ===
python -m pip install -q -e . || (echo. & echo *** install failed -- see above *** & pause & exit /b 1)

echo.
echo === Starting the ALPminer GUI -- keep this window open; close it to stop ===
alpminer gui

echo.
echo (The GUI server has stopped.)
pause
