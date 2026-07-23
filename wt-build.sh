#!/usr/bin/env bash
# Rebuild the engine inside this worktree and drop the extension module beside the
# Python sources, so `PYTHONPATH=<worktree>/python` runs entirely from here — immune
# to whatever the shared checkout's editable install currently points at.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")"
rm -rf dist
maturin build --release -o dist "$@"
python3 - <<'PY'
import glob, os, shutil, zipfile

wheel = glob.glob("dist/*.whl")[0]
with zipfile.ZipFile(wheel) as z:
    for name in (n for n in z.namelist() if n.endswith(".so")):
        dest = os.path.join("python", name)
        with z.open(name) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        os.chmod(dest, 0o755)
        print("installed", dest)
PY
