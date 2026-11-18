"""Raccourcis de développement multiplateformes (Windows / Linux / macOS).

    python tasks.py install
    python tasks.py test
    python tasks.py train-synthetic
    python tasks.py train --category metal_nut [--data data/mvtec_ad]
    python tasks.py train-all
    python tasks.py classifier
    python tasks.py yolo
    python tasks.py api
    python tasks.py front
    python tasks.py up
    python tasks.py export-demo     # pré-calcule la démo statique (frontend/public/demo)
    python tasks.py deploy          # build statique + publication sur GitHub Pages (branche gh-pages)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Utilise le Python du venv du projet s'il existe, même si le venv n'est pas activé
_VENV = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PY = str(_VENV) if _VENV.exists() else sys.executable
NPM = "npm.cmd" if os.name == "nt" else "npm"


def run(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, **(env or {})})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=["install", "test", "synthetic", "train-synthetic", "train", "train-all",
                                    "classifier", "yolo", "api", "front", "up", "export-demo", "deploy"])
    p.add_argument("--data", default="data/mvtec_ad")
    p.add_argument("--category", default="metal_nut")
    a = p.parse_args()
    ml = ROOT / "ml" / "scripts"

    if sys.version_info < (3, 10):
        sys.exit(f"Python {sys.version.split()[0]} détecté : AeroVision demande Python 3.10 ou plus "
                 "(3.11 recommandé). Recrée le venv avec : py -3.11 -m venv .venv")

    if a.task == "install":
        run([PY, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        run([PY, "-m", "pip", "install", "-e", "ml[dev]", "-r", "backend/requirements.txt"])
    elif a.task == "test":
        run([PY, "-m", "pytest", "-q", "tests"], cwd=ROOT / "ml")
        run([PY, "-m", "pytest", "-q", "tests"], cwd=ROOT / "backend")
    elif a.task in ("synthetic", "train-synthetic"):
        run([PY, str(ml / "make_synthetic_dataset.py"), "--out", "data/synthetic"])
        if a.task == "train-synthetic":
            run([PY, str(ml / "train_patchcore.py"), "--data-root", "data/synthetic", "--category",
                 "synthetic_plate", "--backbone", "resnet18", "--out-dir", "models/patchcore"])
    elif a.task == "train":
        run([PY, str(ml / "train_patchcore.py"), "--data-root", a.data, "--category", a.category])
    elif a.task == "train-all":
        run([PY, str(ml / "train_patchcore.py"), "--data-root", a.data, "--category", "all",
             "--coreset-ratio", "0.01"])
    elif a.task == "classifier":
        run([PY, str(ml / "train_classifier.py"), "--data-root", a.data])
    elif a.task == "yolo":
        run([PY, str(ml / "mvtec_to_yolo.py"), "--data-root", a.data, "--out", "data/yolo_defects"])
        run([PY, str(ml / "train_yolo.py"), "--data", "data/yolo_defects/dataset.yaml"])
    elif a.task == "api":
        run([PY, "-m", "uvicorn", "app.main:app", "--reload"], cwd=ROOT / "backend",
            env={"PYTHONPATH": str(ROOT / "ml"), "AEROVISION_MODELS_DIR": str(ROOT / "models"),
                 "AEROVISION_STORAGE_DIR": str(ROOT / "data" / "app" / "storage"),
                 "AEROVISION_DATABASE_URL": f"sqlite:///{(ROOT / 'data' / 'app' / 'aerovision.db').as_posix()}"})
    elif a.task == "front":
        run([NPM, "install"], cwd=ROOT / "frontend")
        run([NPM, "run", "dev"], cwd=ROOT / "frontend")
    elif a.task == "export-demo":
        run([PY, str(ROOT / "scripts" / "export_static_demo.py")])
    elif a.task == "deploy":
        run([NPM, "install"], cwd=ROOT / "frontend")
        run([NPM, "run", "deploy"], cwd=ROOT / "frontend")
    elif a.task == "up":
        run(["docker", "compose", "up", "--build"])


if __name__ == "__main__":
    main()
