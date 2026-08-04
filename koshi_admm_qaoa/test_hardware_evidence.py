"""Regression tests for hardware terminology and evidence validation."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from hardware_evidence import (
    HARDWARE_EVIDENCE_REVISION,
    HARDWARE_EVIDENCE_SCHEMA_VERSION,
    build_problem_record,
    enumerate_exact_optima,
    sha256_file,
    summarize_counts,
    validate_evidence_package,
    write_evidence_manifest,
    write_latex_tables,
)


def _meta():
    # E(00)=0, E(10)=-1, E(01)=-2 (exact), E(11)=1.
    return {
        "n_qubits": 2,
        "switch_branches": [2, 3],
        "linear": np.array([-1.0, -2.0]),
        "Q": np.array([[0.0, 4.0], [0.0, 0.0]]),
        "K_target": 1,
        "topology_encoding": {"revision": "test"},
    }


def test_modal_best_sample_expectation_and_exact_are_distinct():
    meta = _meta()
    exact = enumerate_exact_optima(meta)
    # Qiskit raw "10" reverses to variable-order bits [0,1], a QUBO minimizer.
    counts = {"00": 6, "10": 3, "01": 1}
    summary = summarize_counts(counts, meta, exact)

    assert summary["modal_sample"]["selected_raw_bitstring"] == "00"
    assert summary["modal_sample"]["objective"] == 0.0
    assert not summary["modal_sample"]["is_exact_optimum"]
    assert summary["best_energy_sample"]["selected_raw_bitstring"] == "10"
    assert summary["best_energy_sample"]["objective"] == -2.0
    assert summary["best_energy_sample"]["is_exact_optimum"]
    assert summary["exact_optimum"]["objective"] == -2.0
    assert summary["exact_optimum"]["observed_probability"] == 0.3
    assert np.isclose(summary["sampled_objective"]["expectation"], -0.7)
    assert summary["sampled_objective"]["standard_error"] > 0


def test_complete_evidence_package_recomputes_count_metrics(tmp_path):
    meta = _meta()
    exact = enumerate_exact_optima(meta)
    net = SimpleNamespace(name="test-network", n_bus=3, n_branch=3)
    problem = build_problem_record(net, meta, exact)
    circuit_path = tmp_path / "transpiled_circuit.qpy"
    circuit_path.write_bytes(b"test-qpy-placeholder")
    circuit_hash = sha256_file(circuit_path)
    bound_path = tmp_path / "bound_qaoa_ansatz.qpy"
    bound_path.write_bytes(b"test-bound-qpy-placeholder")
    for name in (
        "backend_configuration.json",
        "backend_properties.json",
        "backend_target.json",
    ):
        (tmp_path / name).write_text("{}\n")
    runs = {}
    for name, counts in {
        "baseline": {"00": 6, "10": 3, "01": 1},
        "noise_managed": {"10": 7, "00": 2, "01": 1},
    }.items():
        counts_path = tmp_path / f"counts_{name}.json"
        counts_path.write_text(json.dumps(counts) + "\n")
        job_path = tmp_path / f"job_{name}.json"
        job_path.write_text(json.dumps({"job_id": f"job-{name}"}) + "\n")
        runs[name] = {
            "condition": {
                "description": name,
                "measurement_twirling": name == "noise_managed",
                "dynamical_decoupling": name == "noise_managed",
            },
            "job_id": f"job-{name}",
            "backend_name": "fake-backend",
            "calibration_timestamp_utc": "2026-07-24T00:00:00+00:00",
            "shots": sum(counts.values()),
            "problem_fingerprint_sha256": problem["problem_fingerprint_sha256"],
            "transpiled_circuit_sha256": circuit_hash,
            "counts_file": counts_path.name,
            "counts_sha256": sha256_file(counts_path),
            "job_metadata_file": job_path.name,
            "job_metadata_sha256": sha256_file(job_path),
            "summary": summarize_counts(counts, meta, exact),
            "same_instance_physics": {
                "modal_sample_same_n": {
                    "connected": True,
                    "radial": True,
                    "fixed_topology_soc_validated": True,
                    "nonlinear_ac_validated": True,
                    "loss_mw": 1.0,
                },
                "best_energy_sample_same_n": {
                    "connected": True,
                    "radial": True,
                    "fixed_topology_soc_validated": True,
                    "nonlinear_ac_validated": True,
                    "loss_mw": 0.9,
                },
                "exact_optimum_same_n": {
                    "connected": True,
                    "radial": True,
                    "fixed_topology_soc_validated": True,
                    "nonlinear_ac_validated": True,
                    "loss_mw": 0.9,
                },
            },
        }

    evidence = {
        "schema_version": HARDWARE_EVIDENCE_SCHEMA_VERSION,
        "evidence_revision": HARDWARE_EVIDENCE_REVISION,
        "problem": problem,
        "execution": {
            "backend_name": "fake-backend",
            "bound_ansatz": {
                "file": bound_path.name,
                "sha256": sha256_file(bound_path),
            },
            "backend_configuration_file": "backend_configuration.json",
            "backend_properties_file": "backend_properties.json",
            "backend_target_file": "backend_target.json",
            "transpiled_circuit": {
                "file": circuit_path.name,
                "sha256": circuit_hash,
            }
        },
        "runs": runs,
    }
    (tmp_path / "hardware_results.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    write_evidence_manifest(tmp_path)
    result = validate_evidence_package(tmp_path)
    assert result["valid"]
    assert result["runs"] == ["baseline", "noise_managed"]
    table_path = write_latex_tables(tmp_path, tmp_path / "hardware_tables.tex")
    table = table_path.read_text()
    assert "Modal $f_Q$" in table
    assert "Best-energy sample" in table
    assert "\\widehat p_{\\rm shot}^\\star" in table
    assert "Same-instance validation" in table