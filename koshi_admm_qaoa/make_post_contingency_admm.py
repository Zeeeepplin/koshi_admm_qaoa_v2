"""Run and archive the predeclared post-contingency ADMM experiment.

This script is intentionally separate from the legacy ``make_admm_figure.py``.
It writes complete iteration histories before any table or figure is generated.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from admm_hybrid import run_admm
from benchmark import _git_commit, _versions
from network_data import build_full_network
from study_protocol import build_protocol, validate_protocol


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "admm_post_contingency_v1.json"


def _active_switch_branches(network, faulted):
    """Return switchable branch indices that remain available in the scenario."""
    faulted_set = set(faulted)
    return [index for index in network.switch_indices() if index not in faulted_set]


def _qaoa_admm_seed_run(task):
    primary, seed = task
    admm = primary["admm"]
    faulted = [primary["contingency"]["forced_open_branch_index"]]
    return run_admm(
        build_full_network(),
        rho=admm["rho"],
        max_iter=admm["maximum_iterations"],
        eps_primal=admm["primal_tolerance"],
        eps_dual=admm["dual_tolerance"],
        z_solver="qaoa",
        qaoa_reps=1,
        qaoa_kind="noiseless",
        qaoa_seed=seed,
        qaoa_maxiter=primary["qaoa"]["optimizer_max_iterations"],
        qaoa_optimizer_tol=primary["qaoa"]["optimizer_tolerance"],
        faulted=faulted,
        verbose=False,
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _save(artifact):
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(_jsonable(artifact), indent=2, allow_nan=False) + "\n")


def run():
    protocol = build_protocol()
    validate_protocol(protocol)
    primary = protocol["prospective_primary_protocol"]
    checkpoint = None
    if OUTPUT.exists():
        try:
            candidate = json.loads(OUTPUT.read_text())
            if candidate.get("protocol_sha256") == protocol["protocol_sha256"]:
                checkpoint = candidate
        except (OSError, ValueError, TypeError):
            checkpoint = None
    admm = primary["admm"]
    faulted = [primary["contingency"]["forced_open_branch_index"]]
    network = build_full_network()
    switch_branches = _active_switch_branches(network, faulted)
    run_commit = _git_commit()
    artifact = {
        "schema_version": 1,
        "artifact_id": "koshi-admm-post-contingency-v1",
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_revision": protocol["protocol_revision"],
        "protocol_sha256": protocol["protocol_sha256"],
        "provenance": {
            "source_commit": run_commit,
            "package_versions": _versions(),
            "generator": "make_post_contingency_admm.run",
        },
        "objective_convention": {
            "objective_symbol": "f_Q",
            "objective_sense": "minimize",
            "binary_to_pauli_mapping": "z_ell = (1 - Z_ell) / 2",
            "returned_array_order": "switch_branches order",
        },
        "scenario": primary["contingency"],
        "faulted_branches": faulted,
        "variable_order": [
            {
                "position": position,
                "branch_index": branch_index,
                "name": network.branches[branch_index].name,
            }
            for position, branch_index in enumerate(switch_branches)
        ],
        "exact_z_update_rho_sensitivity": [],
        "qaoa_z_update_seed_runs": [],
    }
    if checkpoint is not None:
        checkpoint_commit = checkpoint.get("provenance", {}).get("source_commit")
        expected_rhos = {1.5, 3.0, 6.0}
        for result in checkpoint.get("exact_z_update_rho_sensitivity", []):
            rho = result.get("configuration", {}).get("rho")
            if rho in expected_rhos and result.get("history", {}).get("primal"):
                result["source_commit"] = result.get(
                    "source_commit", checkpoint_commit
                )
                artifact["exact_z_update_rho_sensitivity"].append(result)
        expected_seeds = set(primary["qaoa"]["seeds"])
        for result in checkpoint.get("qaoa_z_update_seed_runs", []):
            seed = result.get("configuration", {}).get("qaoa_seed")
            if seed in expected_seeds and result.get("history", {}).get("primal"):
                result["source_commit"] = result.get(
                    "source_commit", checkpoint_commit
                )
                artifact["qaoa_z_update_seed_runs"].append(result)
        if (
            artifact["exact_z_update_rho_sensitivity"]
            or artifact["qaoa_z_update_seed_runs"]
        ):
            artifact["provenance"]["resumed_from_source_commit"] = checkpoint_commit
    _save(artifact)

    for rho in (1.5, 3.0, 6.0):
        if any(
            result["configuration"]["rho"] == rho
            for result in artifact["exact_z_update_rho_sensitivity"]
        ):
            continue
        result = run_admm(
            network,
            rho=rho,
            max_iter=admm["maximum_iterations"],
            eps_primal=admm["primal_tolerance"],
            eps_dual=admm["dual_tolerance"],
            z_solver="exact",
            faulted=faulted,
            verbose=False,
        )
        result["source_commit"] = run_commit
        artifact["exact_z_update_rho_sensitivity"].append(result)
        _save(artifact)

    completed_seeds = {
        result["configuration"]["qaoa_seed"]
        for result in artifact["qaoa_z_update_seed_runs"]
    }
    tasks = [
        (primary, seed)
        for seed in primary["qaoa"]["seeds"]
        if seed not in completed_seeds
    ]
    with ProcessPoolExecutor(
        max_workers=primary["execution"]["parallel_workers"],
        max_tasks_per_child=1,
    ) as pool:
        for result in pool.map(_qaoa_admm_seed_run, tasks):
            result["source_commit"] = run_commit
            artifact["qaoa_z_update_seed_runs"].append(result)
            _save(artifact)

    artifact["exact_z_update_rho_sensitivity"].sort(
        key=lambda result: result["configuration"]["rho"]
    )
    artifact["qaoa_z_update_seed_runs"].sort(
        key=lambda result: result["configuration"]["qaoa_seed"]
    )
    artifact["status"] = "complete"
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _save(artifact)
    return artifact


if __name__ == "__main__":
    run()
    print(f"Saved {OUTPUT}")
