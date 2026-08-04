"""Submit and archive a terminology-safe IBM Quantum hardware experiment.

The historical paper conflated four different quantities: sampled-objective
mean, modal sample, best-energy observed sample, and the exact-QUBO minimum.
This driver
records all four separately through ``hardware_evidence.py`` and refuses to
describe a modal sample as QUBO-optimal unless it belongs to the exact
minimizer set.

Two executions use the same bound and transpiled circuit:

* ``baseline`` disables measurement twirling and dynamical decoupling;
* ``noise_managed`` enables measurement twirling and XpXm dynamical decoupling.

SamplerV2 returns raw samples and does not perform TREX expectation-value
mitigation. The second condition is therefore labelled noise management:
measurement twirling tailors noise, while dynamical decoupling suppresses it.
One job per condition is descriptive evidence only; it does
not support a causal claim that noise management improves performance.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hardware_evidence import (
    HARDWARE_EVIDENCE_REVISION,
    HARDWARE_EVIDENCE_SCHEMA_VERSION,
    build_problem_record,
    enumerate_exact_optima,
    qiskit_bitstring_to_bits,
    sha256_record,
    sha256_file,
    summarize_counts,
    validate_evidence_package,
    write_evidence_manifest,
)
from network_data import build_full_network, scaled_network
from qubo_builder import build_reconfig_qubo, z_to_dict
import power_model as pm


IBM_CHANNEL = "ibm_quantum_platform"
ROOT = Path(__file__).resolve().parent


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")
    return path


def _safe_call(obj, name):
    attribute = getattr(obj, name, None)
    if attribute is None:
        return None
    try:
        return attribute() if callable(attribute) else attribute
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def _calibration_timestamp(properties) -> str:
    value = getattr(properties, "last_update_date", None)
    if value is None and hasattr(properties, "to_dict"):
        value = properties.to_dict().get("last_update_date")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if value is not None:
        return str(value)
    return "unavailable-in-backend-properties"


def _physics_record(net, bits, meta, faulted=None):
    result = pm.ac_feasibility(
        net, z_to_dict(np.asarray(bits, dtype=int), meta), faulted=faulted
    )
    keys = (
        "model_revision", "status", "soc_feasible", "loss_mw", "shed_mw",
        "connected", "radial", "fixed_topology_soc_validated",
        "nonlinear_ac_validated", "engineering_validated", "diagnostics",
        "nonlinear_ac",
    )
    return _jsonable({key: result.get(key) for key in keys})


def _extract_counts(pub_result):
    data = pub_result.data
    keys = list(data.keys())
    if not keys:
        raise ValueError("SamplerV2 PubResult contains no classical data register")
    register_name = keys[0]
    register = getattr(data, register_name)
    return register.get_counts(), register_name


def _build_hardware_problem(n_switches=None):
    """Default to the predeclared post-contingency instance.

    Passing ``n_switches`` is retained only for explicit legacy diagnostics.
    """
    if n_switches is None:
        from study_protocol import build_protocol, validate_protocol

        protocol = build_protocol()
        validate_protocol(protocol)
        primary = protocol["prospective_primary_protocol"]
        faulted = [primary["contingency"]["forced_open_branch_index"]]
        net = build_full_network()
        qp, meta = build_reconfig_qubo(
            net,
            faulted=faulted,
            **{
                key: primary["qubo_parameters"][key]
                for key in (
                    "lambda_card", "lambda_cycle", "lambda_iso", "loss_bias"
                )
            },
        )
        return net, qp, meta, faulted, primary["contingency"], protocol
    net = scaled_network(n_switches)
    qp, meta = build_reconfig_qubo(net)
    return net, qp, meta, [], {"legacy_scaled_n": int(n_switches)}, None


def run_phase3(
    n_switches=None,
    reps=1,
    shots=4096,
    min_qubits=None,
    token=None,
):
    """Run two same-circuit hardware conditions and archive a v2 evidence package."""
    from qiskit import qpy
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    import solvers

    token = token or os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError(
            "Set IBM_QUANTUM_TOKEN or pass token= explicitly; credentials are not "
            "written to the evidence package."
        )

    net, qp, meta, faulted, scenario, protocol = _build_hardware_problem(n_switches)
    exact_reference = enumerate_exact_optima(meta)
    problem = build_problem_record(
        net,
        meta,
        exact_reference,
        faulted_branches=faulted,
        scenario=scenario,
    )
    if protocol is not None:
        problem["protocol_revision"] = protocol["protocol_revision"]
        problem["protocol_sha256"] = protocol["protocol_sha256"]
        unsigned_problem = dict(problem)
        unsigned_problem.pop("problem_fingerprint_sha256", None)
        problem["problem_fingerprint_sha256"] = sha256_record(unsigned_problem)
    print(
        f"Phase 3: n={meta['n_qubits']}, exact QUBO objective="
        f"{exact_reference['objective']:.6f}, degeneracy={exact_reference['degeneracy']}"
    )

    # Local parameter training is distinct from hardware sampling.  The
    # MinimumEigenOptimizer result is recorded as a returned candidate, not as
    # an expectation unless the primitive explicitly exposes one.
    training_seed = 42
    solvers.set_algorithm_seed(training_seed)
    initial_point = solvers.variational_initial_point(training_seed, 2 * reps)
    qaoa = QAOA(
        sampler=solvers._make_sampler("noiseless", 2048, 42, 0, 0),
        optimizer=COBYLA(maxiter=100, tol=1.0e-4),
        reps=reps,
        initial_point=initial_point,
    )
    local_result = MinimumEigenOptimizer(qaoa).solve(qp)
    optimal_point = np.asarray(
        local_result.min_eigen_solver_result.optimal_point, dtype=float
    )
    bound_ansatz = qaoa.ansatz.assign_parameters(optimal_point)
    if bound_ansatz.num_clbits == 0:
        bound_ansatz.measure_all()

    service = QiskitRuntimeService(channel=IBM_CHANNEL, token=token)
    required_qubits = max(meta["n_qubits"], int(min_qubits or 0))
    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=required_qubits
    )
    pass_manager = generate_preset_pass_manager(
        optimization_level=3, backend=backend
    )
    isa = pass_manager.run(bound_ansatz)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = ROOT / "results" / f"hardware_{timestamp}"
    evidence_dir.mkdir(parents=True, exist_ok=False)

    bound_path = evidence_dir / "bound_qaoa_ansatz.qpy"
    transpiled_path = evidence_dir / "transpiled_circuit.qpy"
    with bound_path.open("wb") as stream:
        qpy.dump(bound_ansatz, stream)
    with transpiled_path.open("wb") as stream:
        qpy.dump(isa, stream)
    circuit_hash = sha256_file(transpiled_path)

    backend_configuration = _safe_call(backend, "configuration")
    backend_properties_object = _safe_call(backend, "properties")
    backend_target = _safe_call(backend, "target")
    _write_json(evidence_dir / "backend_configuration.json", backend_configuration)
    _write_json(evidence_dir / "backend_properties.json", backend_properties_object)
    _write_json(evidence_dir / "backend_target.json", backend_target)
    calibration_timestamp = _calibration_timestamp(backend_properties_object)

    exact_bits = exact_reference["bitstrings_variable_order"][0]
    exact_physics = _physics_record(net, exact_bits, meta, faulted)
    run_records = {}
    conditions = {
        "baseline": {
            "measurement_twirling": False,
            "dynamical_decoupling": False,
            "description": "raw SamplerV2 condition without enabled noise-management options",
        },
        "noise_managed": {
            "measurement_twirling": True,
            "dynamical_decoupling": True,
            "dd_sequence": "XpXm",
            "description": (
                "SamplerV2 measurement twirling plus XpXm dynamical decoupling; "
                "not TREX expectation-value mitigation"
            ),
        },
    }

    for condition_name, condition in conditions.items():
        sampler = RuntimeSampler(mode=backend)
        sampler.options.twirling.enable_measure = condition["measurement_twirling"]
        sampler.options.dynamical_decoupling.enable = condition[
            "dynamical_decoupling"
        ]
        if condition["dynamical_decoupling"]:
            sampler.options.dynamical_decoupling.sequence_type = condition["dd_sequence"]
        options_record = _jsonable(sampler.options)
        submitted_at = datetime.now(timezone.utc).isoformat()
        job = sampler.run([isa], shots=shots)
        job_id = job.job_id()
        print(f"  {condition_name}: submitted job {job_id}")
        primitive_result = job.result()
        completed_at = datetime.now(timezone.utc).isoformat()
        pub_result = primitive_result[0]
        counts, register_name = _extract_counts(pub_result)
        counts = {str(key): int(value) for key, value in counts.items()}
        summary = summarize_counts(counts, meta, exact_reference)

        counts_path = _write_json(
            evidence_dir / f"counts_{condition_name}.json", counts
        )
        job_record = {
            "job_id": job_id,
            "status": _safe_call(job, "status"),
            "metrics": _safe_call(job, "metrics"),
            "usage": _safe_call(job, "usage"),
            "creation_date": _safe_call(job, "creation_date"),
            "submitted_at_utc": submitted_at,
            "completed_at_utc": completed_at,
            "primitive_result_metadata": _jsonable(getattr(pub_result, "metadata", None)),
            "classical_register": register_name,
            "sampler_options": options_record,
        }
        job_path = _write_json(
            evidence_dir / f"job_{condition_name}.json", job_record
        )

        modal_bits = summary["modal_sample"]["bits_variable_order"]
        best_bits = summary["best_energy_sample"]["bits_variable_order"]
        physics = {
            "modal_sample_same_n": _physics_record(net, modal_bits, meta, faulted),
            "best_energy_sample_same_n": _physics_record(net, best_bits, meta, faulted),
            "exact_optimum_same_n": exact_physics,
        }
        run_records[condition_name] = {
            "condition": condition,
            "job_id": job_id,
            "backend_name": str(backend.name),
            "calibration_timestamp_utc": calibration_timestamp,
            "submitted_at_utc": submitted_at,
            "completed_at_utc": completed_at,
            "shots": int(shots),
            "problem_fingerprint_sha256": problem["problem_fingerprint_sha256"],
            "transpiled_circuit_sha256": circuit_hash,
            "counts_file": counts_path.name,
            "counts_sha256": sha256_file(counts_path),
            "job_metadata_file": job_path.name,
            "job_metadata_sha256": sha256_file(job_path),
            "summary": summary,
            "same_instance_physics": physics,
        }
        modal = summary["modal_sample"]
        best = summary["best_energy_sample"]
        sampled = summary["sampled_objective"]
        print(
            f"    modal f_Q={modal['objective']:.6f}, "
            f"best-sample f_Q={best['objective']:.6f}, "
            f"mu_Q_hat={sampled['mean']:.6f} +/- "
            f"{sampled['standard_error']:.6f}, "
            f"p_shot_star={summary['exact_optimum']['observed_probability']:.4f}"
        )

    baseline = run_records["baseline"]["summary"]
    noise_managed = run_records["noise_managed"]["summary"]
    observed_comparison = {
        "interpretation": (
            "descriptive difference between one baseline job and one noise-managed "
            "job; not a causal mitigation estimate"
        ),
        "sampled_objective_mean_difference_noise_managed_minus_baseline": (
            noise_managed["sampled_objective"]["mean"]
            - baseline["sampled_objective"]["mean"]
        ),
        "best_sample_gap_difference_noise_managed_minus_baseline": (
            noise_managed["best_energy_sample"]["objective_gap_from_exact"]
            - baseline["best_energy_sample"]["objective_gap_from_exact"]
        ),
        "shot_level_exact_set_probability_difference_noise_managed_minus_baseline": (
            noise_managed["exact_optimum"]["observed_probability"]
            - baseline["exact_optimum"]["observed_probability"]
        ),
    }

    evidence = {
        "schema_version": HARDWARE_EVIDENCE_SCHEMA_VERSION,
        "evidence_revision": HARDWARE_EVIDENCE_REVISION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_status": (
            "raw evidence package; manuscript claims require independent package "
            "validation and repeated-job uncertainty analysis"
        ),
        "problem": problem,
        "local_training": {
            "returned_candidate_bits": local_result.x.astype(int).tolist(),
            "returned_candidate_objective": float(local_result.fval),
            "trained_parameters": optimal_point.tolist(),
            "initial_point": initial_point.tolist(),
            "optimizer": "COBYLA",
            "optimizer_max_iterations": 100,
            "training_sampler": "local noiseless Aer SamplerV2",
            "training_shots": 2048,
            "training_seed": training_seed,
        },
        "execution": {
            "backend_name": str(backend.name),
            "calibration_timestamp_utc": calibration_timestamp,
            "shots_per_condition": int(shots),
            "qaoa_depth": int(reps),
            "transpiler_optimization_level": 3,
            "bound_ansatz": {
                "file": bound_path.name,
                "sha256": sha256_file(bound_path),
            },
            "transpiled_circuit": {
                "file": transpiled_path.name,
                "sha256": circuit_hash,
                "depth": int(isa.depth()),
                "num_qubits": int(isa.num_qubits),
                "num_clbits": int(isa.num_clbits),
                "nonlocal_gates": int(isa.num_nonlocal_gates()),
                "operation_counts": _jsonable(dict(isa.count_ops())),
            },
            "backend_configuration_file": "backend_configuration.json",
            "backend_properties_file": "backend_properties.json",
            "backend_target_file": "backend_target.json",
        },
        "runs": run_records,
        "observed_comparison": observed_comparison,
    }
    evidence_path = _write_json(evidence_dir / "hardware_results.json", evidence)
    write_evidence_manifest(evidence_dir)
    validation = validate_evidence_package(evidence_dir)
    print(f"Evidence saved to {evidence_path}")
    print(f"Offline validation: {validation}")
    return evidence


def local_selftest(n_switches=None, reps=1, shots=4096):
    """Exercise training/transpilation and metric separation without a QPU job."""
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    import solvers

    net, qp, meta, _, _, _ = _build_hardware_problem(n_switches)
    exact = enumerate_exact_optima(meta)
    training_seed = 42
    solvers.set_algorithm_seed(training_seed)
    initial_point = solvers.variational_initial_point(training_seed, 2 * reps)
    qaoa = QAOA(
        sampler=solvers._make_sampler("noiseless", 2048, 42, 0, 0),
        optimizer=COBYLA(maxiter=60, tol=1.0e-4),
        reps=reps,
        initial_point=initial_point,
    )
    result = MinimumEigenOptimizer(qaoa).solve(qp)
    point = result.min_eigen_solver_result.optimal_point
    ansatz = qaoa.ansatz.assign_parameters(point)
    if ansatz.num_clbits == 0:
        ansatz.measure_all()
    backend = FakeSherbrooke()
    isa = generate_preset_pass_manager(
        optimization_level=3, backend=backend
    ).run(ansatz)
    pub_result = AerSampler(default_shots=shots).run([isa]).result()[0]
    counts, _ = _extract_counts(pub_result)
    summary = summarize_counts(counts, meta, exact)
    print(
        f"[selftest] f_Q_star={exact['objective']:.6f}; "
        f"modal_f_Q={summary['modal_sample']['objective']:.6f}; "
        f"best_sample_f_Q={summary['best_energy_sample']['objective']:.6f}; "
        f"mu_Q_hat={summary['sampled_objective']['mean']:.6f} +/- "
        f"{summary['sampled_objective']['standard_error']:.6f}; "
        f"ISA depth={isa.depth()}, 2q gates={isa.num_nonlocal_gates()}"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--n-switches",
        type=int,
        default=None,
        help="explicit legacy scaled instance; omit for the predeclared branch-3 contingency",
    )
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--min-qubits", type=int, default=None)
    args = parser.parse_args()
    if args.selftest:
        local_selftest(args.n_switches, args.reps, args.shots)
    else:
        run_phase3(
            n_switches=args.n_switches,
            reps=args.reps,
            shots=args.shots,
            min_qubits=args.min_qubits,
        )
