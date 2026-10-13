#!/bin/sh
# Development dependency setup; source/set data are in the pinned public payload.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
venv=${1:-/opt/frontier-foul-play}
foul_play_build_python=${FOUL_PLAY_BUILD_PYTHON:-python3.12}
"$foul_play_build_python" -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip==25.0.1
if [ -f "$here/engine/engine-build.json" ] && "$venv/bin/python" -c \
    'import platform,sys; sys.exit(not (sys.platform == "linux" and platform.machine() == "x86_64" and sys.version_info[:2] == (3,12)))'; then
    "$venv/bin/python" - "$here/engine" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
receipt = json.loads((root / 'engine-build.json').read_text())
wheel = root / 'poke_engine-0.0.48-cp312-cp312-linux_x86_64.whl'
if hashlib.sha256(wheel.read_bytes()).hexdigest() != receipt['wheel_sha256']:
    raise SystemExit('Task engine wheel digest mismatch')
PY
    "$venv/bin/python" -m pip install --no-cache-dir --force-reinstall --only-binary=poke-engine \
        --find-links "$here/engine" -r "$here/foul-play-requirements.txt"
else
    command -v cargo >/dev/null || { echo 'Install Rust >=1.89 first (https://rustup.rs).' >&2; exit 1; }
    # Other platforms and verifier image builds use the pinned source with Tera.
    "$venv/bin/python" -m pip install --no-cache-dir --force-reinstall --no-binary=poke-engine -r "$here/foul-play-requirements.txt"
fi
echo "Use FOUL_PLAY_PYTHON=$venv/bin/python when running validation."
