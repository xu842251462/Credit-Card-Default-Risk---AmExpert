#!/usr/bin/env python3
"""Basic project health checks for the AmExpert credit-default repository."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "Credit-Card-Default-Risk---AmExpert-CodeLab-main"
NOTEBOOK_PATH = PROJECT_DIR / "Credit Card Default Risk - Prediction.ipynb"
DATASET_DIR = PROJECT_DIR / "dataset"

EXPECTED_DATASETS = [
    "train.csv",
    "test.csv",
    "sample_submission.csv",
]

RUNTIME_PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "imblearn",
    "xgboost",
    "tensorflow",
]


def check_path(path: Path, label: str) -> None:
    status = "OK" if path.exists() else "MISSING"
    print(f"[{status}] {label}: {path.relative_to(ROOT)}")


def count_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def read_headers(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def installed(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def iter_notebook_imports(nb_path: Path) -> Iterable[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                yield stripped


def main() -> int:
    print("== Project file checks ==")
    check_path(PROJECT_DIR, "project directory")
    check_path(NOTEBOOK_PATH, "main notebook")

    print("\n== Dataset checks ==")
    for filename in EXPECTED_DATASETS:
        csv_path = DATASET_DIR / filename
        check_path(csv_path, filename)
        if csv_path.exists():
            print(f"    rows: {count_rows(csv_path):,}")
            print(f"    columns: {len(read_headers(csv_path))}")

    train_path = DATASET_DIR / "train.csv"
    test_path = DATASET_DIR / "test.csv"
    if train_path.exists() and test_path.exists():
        train_headers = read_headers(train_path)
        test_headers = read_headers(test_path)
        target_col = "credit_card_default"
        print("\n== Target column checks ==")
        print(f"[{'OK' if target_col in train_headers else 'MISSING'}] '{target_col}' in train.csv")
        print(f"[{'OK' if target_col not in test_headers else 'UNEXPECTED'}] '{target_col}' not in test.csv")

    print("\n== Python environment checks ==")
    for pkg in RUNTIME_PACKAGES:
        print(f"[{'OK' if installed(pkg) else 'MISSING'}] {pkg}")

    print("\n== Notebook import scan ==")
    imports = sorted(set(iter_notebook_imports(NOTEBOOK_PATH))) if NOTEBOOK_PATH.exists() else []
    if not imports:
        print("No imports found or notebook missing.")
    else:
        print(f"Found {len(imports)} unique import lines in notebook.")
        for line in imports:
            print(f"  - {line}")

    print("\nCheck complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
