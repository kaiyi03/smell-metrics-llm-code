#!/usr/bin/env python3
"""One command to set up and launch the evaluation dashboard locally.

    python run.py

The first run creates a local environment (.venv) and installs the dashboard's
dependencies -- a minute or two, once. Every run after that just starts the
dashboard and opens your browser. The only prerequisite is Python 3.9+.

(This is the easy path. If you would rather not install anything at all, open the
hosted version instead -- see deploy/DEPLOY.md.)
"""

import os
import subprocess
import sys
import venv

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
PY = (os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt"
      else os.path.join(VENV, "bin", "python"))
REQS = os.path.join(ROOT, "deploy", "requirements-space.txt")
STAMP = os.path.join(VENV, ".dashboard-deps-installed")


def pip(*args):
    subprocess.run([PY, "-m", "pip", *args], check=True)


def main():
    if sys.version_info < (3, 9):
        sys.exit(f"Python 3.9+ required; this is {sys.version.split()[0]}")

    if not os.path.exists(PY):
        print("[1/2] creating a local environment (.venv) ... one moment")
        venv.create(VENV, with_pip=True)

    if not os.path.exists(STAMP):
        print("[2/2] installing dependencies (one-time, ~1-2 min) ...")
        pip("install", "--upgrade", "pip", "-q")
        pip("install", "-q", "-r", REQS)
        # codebleu pins an old tree-sitter; install it without deps so it doesn't
        # downgrade the others (it works against the newer one). Best-effort.
        subprocess.run([PY, "-m", "pip", "install", "-q", "--no-deps", "codebleu"])
        open(STAMP, "w").close()
        print("      done.\n")

    print("starting the dashboard (your browser will open; press Ctrl+C here to stop)\n")
    subprocess.run([PY, os.path.join(ROOT, "dashboard", "app.py")])


if __name__ == "__main__":
    main()
