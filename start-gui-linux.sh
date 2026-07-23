#!/usr/bin/env bash
# ==================================================================
#  ALPminer -- one-click GUI launcher (Linux)
#
#  Run it from a terminal in the code folder:
#      ./start-gui-linux.sh
#  (double-click-to-run depends on your desktop's file manager).
#  It reuses the existing virtual environment when it is healthy and
#  rebuilds it automatically when it is missing or broken (e.g. after
#  copying the folder to another computer or upgrading Python), then
#  starts the GUI server.
#
#  First time only: the file may need to be made executable with
#      chmod +x start-gui-linux.sh
#  (a download or unzip can drop the executable bit).
#
#  Keep this window OPEN while you use the app. Close it, or press
#  Ctrl-C, to stop the server.
# ==================================================================
set -u

# work from this script's own folder, wherever it is launched from
cd "$(dirname "$0")" || exit 1

PYTHON="${PYTHON:-python3}"
pause() { printf '\n'; read -r -p "Press Enter to close..." _junk; }

# reuse the venv when it works; rebuild only when missing or broken
if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import alpminer.schema" >/dev/null 2>&1; then
  echo "=== Using the existing virtual environment (.venv) ==="
else
  [ -d ".venv" ] && echo "Existing environment is broken or from another machine; rebuilding..."
  echo
  echo "=== Creating a clean virtual environment (.venv) ==="
  "$PYTHON" -m venv --clear .venv || { echo; echo "*** venv creation failed -- see above ***"; pause; exit 1; }
fi

echo
echo "=== Activating the environment ==="
# shellcheck disable=SC1091
. .venv/bin/activate || { echo; echo "*** activation failed -- see above ***"; pause; exit 1; }

echo
echo "=== Installing/refreshing ALPminer (fast when unchanged) ==="
python -m pip install -q -e . || { echo; echo "*** install failed -- see above ***"; pause; exit 1; }

echo
echo "=== Starting the ALPminer GUI -- keep this window open; close it to stop ==="
alpminer gui

echo
echo "(The GUI server has stopped.)"
pause
