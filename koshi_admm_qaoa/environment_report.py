"""Write a machine-readable execution-environment fingerprint.

The report deliberately records only reproducibility metadata. It does not
contain credentials, environment variables, host names, or user names.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "environment.json"
PACKAGES = (
    "clarabel",
    "cvxpy",
    "matplotlib",
    "networkx",
    "numpy",
    "pytest",
    "qiskit",
    "qiskit-aer",
    "qiskit-algorithms",
    "qiskit-ibm-runtime",
    "qiskit-optimization",
    "scipy",
)


def _git(args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_tree_dirty() -> bool:
    """Ignore products that the evidence pipeline is expected to rewrite."""
    status = _git(
        [
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            ".",
            ":(exclude)results/**",
            ":(exclude)generated/**",
            ":(exclude)figures/**",
            ":(exclude)RESULTS_SUMMARY.md",
        ]
    )
    return bool(status)


def build_report() -> dict:
    try:
        import psutil

        memory_bytes = int(psutil.virtual_memory().total)
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
    except ModuleNotFoundError:
        memory_bytes = None
        physical_cores = None
        logical_cores = os.cpu_count()
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_architecture": platform.architecture()[0],
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "hardware": {
            "processor": platform.processor(),
            "physical_cpu_cores": physical_cores,
            "logical_cpu_cores": logical_cores,
            "memory_bytes": memory_bytes,
        },
        "packages": versions,
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["branch", "--show-current"]),
            "source_working_tree_dirty": _source_tree_dirty(),
            "dirty_scope": (
                "tracked and untracked source files under koshi_admm_qaoa; "
                "results, generated, figures, and RESULTS_SUMMARY.md excluded"
            ),
        },
        "timing_contract": {
            "benchmark_time": "wall-clock seconds measured around each solver call",
            "includes": "local algorithm execution and primitive simulation",
            "excludes": (
                "artifact rendering; for QPU runs, queue time is recorded "
                "separately and is not combined with local wall time"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = build_report()
    if args.check:
        saved = json.loads(args.output.read_text())
        if saved["packages"] != current["packages"]:
            raise RuntimeError("saved package versions differ from this environment")
        if saved["python"]["version"] != current["python"]["version"]:
            raise RuntimeError("saved Python version differs from this environment")
        print("environment report check passed")
        return
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(current, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
