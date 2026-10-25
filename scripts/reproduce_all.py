#!/usr/bin/env python3
"""
Full Thesis Model Reproduction Runner (Token-Free Autonomous Script)
=====================================================================
Master's Thesis: Study of Implied Volatility Surface with Machine Learning Techniques
Author: Sérgio da Cunha Sobral | Supervisor: Profª. Drª. Diana Mendes (ISCTE-IUL)

Usage:
------
Run full replication overnight (0 tokens consumed, runs directly in your local terminal):
    python scripts/reproduce_all.py

Run quick smoke test to verify all models without full GridSearch:
    python scripts/reproduce_all.py --smoke-test
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

NOTEBOOKS = [
    "03_notebooks/03_modeling/01_model_ols.ipynb",
    "03_notebooks/03_modeling/02_model_polynomial.ipynb",
    "03_notebooks/03_modeling/03_model_splines.ipynb",
    "03_notebooks/03_modeling/04_model_random_forest.ipynb",
    "03_notebooks/03_modeling/05_model_xgboost.ipynb",
    "03_notebooks/03_modeling/06_model_hybrid_random_forest.ipynb",
    "03_notebooks/03_modeling/07_model_hybrid_xgboost.ipynb",
]

def main():
    parser = argparse.ArgumentParser(description="Full Thesis Model Replication Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid unit/integration tests instead of full 30h grid search")
    args = parser.parse_args()

    print("=" * 80)
    print("  SPX IMPLIED VOLATILITY SURFACE - MACHINE LEARNING REPLICATION")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    if args.smoke_test:
        print("\n[MODE] Running Fast Smoke Test (tests/test_pipeline_reproducibility.py)...")
        cmd = [sys.executable, "tests/test_pipeline_reproducibility.py"]
        t0 = time.time()
        ret = subprocess.call(cmd)
        elapsed = time.time() - t0
        print(f"\n[DONE] Smoke test finished in {elapsed:.2f}s with status code {ret}.")
        sys.exit(ret)

    print("\n[MODE] Full Notebook Sequential Execution")
    print("WARNING: Re-running all 7 models from scratch with 8.86M contracts and GridSearch")
    print("may take 15-30+ hours of compute depending on CPU/GPU hardware.\n")

    t_start_all = time.time()
    for idx, nb in enumerate(NOTEBOOKS, 1):
        if not os.path.exists(nb):
            print(f"Error: Notebook not found: {nb}")
            continue

        print(f"[{idx}/{len(NOTEBOOKS)}] Executing {nb}...")
        t0 = time.time()
        # Run notebook headless using jupyter nbconvert
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            nb
        ]
        try:
            subprocess.check_call(cmd)
            elapsed = time.time() - t0
            print(f"  ✓ Completed in {elapsed/60:.2f} minutes.\n")
        except Exception as e:
            print(f"  ✗ Failed to execute {nb}: {e}\n")

    total_elapsed = time.time() - t_start_all
    print("=" * 80)
    print(f"  ALL EXPERIMENTS FINISHED. Total Elapsed Time: {total_elapsed/3600:.2f} hours.")
    print("=" * 80)

if __name__ == "__main__":
    main()

