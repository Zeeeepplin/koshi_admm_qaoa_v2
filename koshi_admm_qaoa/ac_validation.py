"""Nonlinear AC recovery and branch-flow angle diagnostics.

This module is deliberately independent of CVXPY so that the nonlinear
validation can be unit-tested even when the conic-optimization environment is
not installed. Transformer nominal ratios are absorbed by the bus voltage
bases; ``tap_ratio_pu`` is the explicit real off-nominal from-side tap.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from scipy.optimize import least_squares

from network_data import Network, S_BASE


def _is_closed(net: Network, k: int, z: Dict[int, int], faulted: set[int]) -> bool:
    if k in faulted:
        return False
    branch = net.branches[k]
    return not branch.switchable or bool(int(z.get(k, 1)))


def _wrap_angle(values):
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def build_ybus(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Build the series-only per-unit Ybus with real off-nominal taps."""
    faulted_set = set(faulted or [])
    ybus = np.zeros((net.n_bus, net.n_bus), dtype=complex)
    for k, branch in enumerate(net.branches):
        if not _is_closed(net, k, z, faulted_set):
            continue
        impedance = complex(branch.r_pu, branch.x_pu)
        if abs(impedance) < 1e-14:
            raise ValueError(f"branch {k} has zero series impedance")
        admittance = 1.0 / impedance
        tap = float(branch.tap_ratio_pu)
        i, j = branch.frm, branch.to
        ybus[i, i] += admittance / (tap * tap)
        ybus[i, j] -= admittance / tap
        ybus[j, i] -= admittance / tap
        ybus[j, j] += admittance
    return ybus


def recover_branch_flow_angles(
    net: Network,
    z: Dict[int, int],
    p_pu,
    q_pu,
    v_sq_pu,
    faulted: Optional[Iterable[int]] = None,
    tolerance_rad: float = 1e-5,
) -> dict:
    """Least-squares angle recovery from branch-flow variables.

    For the series-side convention used in ``power_model.py``,
    ``theta_i - theta_j = atan2(xP-rQ, v_i/t^2-rP-xQ)``.
    A nonzero least-squares residual exposes inconsistent cycle angles.
    """
    p_pu = np.asarray(p_pu, dtype=float)
    q_pu = np.asarray(q_pu, dtype=float)
    v_sq_pu = np.asarray(v_sq_pu, dtype=float)
    faulted_set = set(faulted or [])
    non_slack = [i for i in range(net.n_bus) if i != net.slack]
    position = {bus: col for col, bus in enumerate(non_slack)}
    rows = []
    implied = []
    branch_indices = []
    for k, branch in enumerate(net.branches):
        if not _is_closed(net, k, z, faulted_set):
            continue
        tap = float(branch.tap_ratio_pu)
        source_v = v_sq_pu[branch.frm] / (tap * tap)
        real = source_v - branch.r_pu * p_pu[k] - branch.x_pu * q_pu[k]
        imag = branch.x_pu * p_pu[k] - branch.r_pu * q_pu[k]
        beta = float(np.arctan2(imag, real))
        row = np.zeros(len(non_slack))
        if branch.frm != net.slack:
            row[position[branch.frm]] += 1.0
        if branch.to != net.slack:
            row[position[branch.to]] -= 1.0
        rows.append(row)
        implied.append(beta)
        branch_indices.append(k)

    theta = np.zeros(net.n_bus)
    if rows:
        matrix = np.vstack(rows)
        rhs = np.asarray(implied)
        solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        theta[non_slack] = solution
        residual = _wrap_angle(matrix @ solution - rhs)
    else:
        residual = np.array([], dtype=float)
    max_residual = float(np.max(np.abs(residual))) if residual.size else 0.0
    return {
        "angles_rad": theta.tolist(),
        "branch_indices": branch_indices,
        "branch_residual_rad": residual.tolist(),
        "max_residual_rad": max_residual,
        "recoverable": bool(max_residual <= tolerance_rad),
        "tolerance_rad": float(tolerance_rad),
    }


def nonlinear_ac_power_flow(
    net: Network,
    z: Dict[int, int],
    p_shed_mw=None,
    q_shed_mvar=None,
    initial_v_sq_pu=None,
    initial_angles_rad=None,
    faulted: Optional[Iterable[int]] = None,
    mismatch_tolerance_mva: float = 1e-3,
    vmin_pu: float = 0.90,
    vmax_pu: float = 1.10,
) -> dict:
    """Run a fixed-injection nonlinear AC power flow of the series-only model.

    All non-slack buses are PQ buses. Positive ``p_load_mw`` and
    ``q_load_mvar`` are demands; negative values are fixed injections. The
    slack voltage is 1∠0 pu and its active/reactive injection is recovered.
    """
    nb = net.n_bus
    non_slack = [i for i in range(nb) if i != net.slack]
    p_shed = np.zeros(nb) if p_shed_mw is None else np.asarray(p_shed_mw, dtype=float)
    q_shed = np.zeros(nb) if q_shed_mvar is None else np.asarray(q_shed_mvar, dtype=float)
    if p_shed.shape != (nb,) or q_shed.shape != (nb,):
        raise ValueError("shed arrays must have one value per bus")

    p_spec = np.array(
        [(-bus.p_load_mw + p_shed[bus.idx]) / S_BASE for bus in net.buses]
    )
    q_spec = np.array(
        [(-bus.q_load_mvar + q_shed[bus.idx]) / S_BASE for bus in net.buses]
    )
    ybus = build_ybus(net, z, faulted)

    theta0 = np.zeros(nb)
    if initial_angles_rad is not None:
        theta0 = np.asarray(initial_angles_rad, dtype=float).copy()
    vm0 = np.ones(nb)
    if initial_v_sq_pu is not None:
        vm0 = np.sqrt(np.maximum(np.asarray(initial_v_sq_pu, dtype=float), 1e-8))
    x0 = np.concatenate([theta0[non_slack], vm0[non_slack]])

    def unpack(state):
        theta = np.zeros(nb)
        vm = np.ones(nb)
        theta[non_slack] = state[: len(non_slack)]
        vm[non_slack] = state[len(non_slack) :]
        return theta, vm

    def mismatches(state):
        theta, vm = unpack(state)
        voltage = vm * np.exp(1j * theta)
        injection = voltage * np.conj(ybus @ voltage)
        return np.concatenate(
            [injection.real[non_slack] - p_spec[non_slack],
             injection.imag[non_slack] - q_spec[non_slack]]
        )

    lower = np.concatenate(
        [np.full(len(non_slack), -np.pi), np.full(len(non_slack), 0.50)]
    )
    upper = np.concatenate(
        [np.full(len(non_slack), np.pi), np.full(len(non_slack), 1.50)]
    )
    result = least_squares(
        mismatches,
        np.clip(x0, lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=3000,
    )
    theta, vm = unpack(result.x)
    voltage = vm * np.exp(1j * theta)
    injection = voltage * np.conj(ybus @ voltage)
    mismatch = mismatches(result.x)
    max_mismatch_mva = (
        float(np.max(np.abs(mismatch)) * S_BASE) if mismatch.size else 0.0
    )

    faulted_set = set(faulted or [])
    branches = []
    p_from_pu = [None] * net.n_branch
    q_from_pu = [None] * net.n_branch
    current_sq_pu = [None] * net.n_branch
    total_loss = 0.0
    max_apparent_util = 0.0
    max_current_util = 0.0
    for k, branch in enumerate(net.branches):
        if not _is_closed(net, k, z, faulted_set):
            continue
        tap = float(branch.tap_ratio_pu)
        admittance = 1.0 / complex(branch.r_pu, branch.x_pu)
        series_current = (voltage[branch.frm] / tap - voltage[branch.to]) * admittance
        s_from = (voltage[branch.frm] / tap) * np.conj(series_current)
        s_to = voltage[branch.to] * np.conj(-series_current)
        loss = s_from + s_to
        apparent_util = max(abs(s_from), abs(s_to)) / branch.rating_pu
        current_util = abs(series_current) / branch.rating_pu
        max_apparent_util = max(max_apparent_util, float(apparent_util))
        max_current_util = max(max_current_util, float(current_util))
        total_loss += float(loss.real * S_BASE)
        p_from_pu[k] = float(s_from.real)
        q_from_pu[k] = float(s_from.imag)
        current_sq_pu[k] = float(abs(series_current) ** 2)
        branches.append(
            {
                "branch": k,
                "s_from_mva": [float(s_from.real * S_BASE), float(s_from.imag * S_BASE)],
                "s_to_mva": [float(s_to.real * S_BASE), float(s_to.imag * S_BASE)],
                "current_pu": float(abs(series_current)),
                "apparent_power_utilization": float(apparent_util),
                "current_utilization": float(current_util),
            }
        )

    voltage_ok = bool(vm.min() >= vmin_pu - 1e-7 and vm.max() <= vmax_pu + 1e-7)
    thermal_ok = bool(max(max_apparent_util, max_current_util) <= 1.0 + 1e-7)
    converged = bool(result.success and max_mismatch_mva <= mismatch_tolerance_mva)
    slack_injection = injection[net.slack] * S_BASE
    return {
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "nfev": int(result.nfev),
        "converged": converged,
        "validated": bool(converged and voltage_ok and thermal_ok),
        "max_mismatch_mva": max_mismatch_mva,
        "mismatch_tolerance_mva": float(mismatch_tolerance_mva),
        "voltage_ok": voltage_ok,
        "thermal_ok": thermal_ok,
        "vmin_pu": float(vm.min()),
        "vmax_pu": float(vm.max()),
        "angles_rad": theta.tolist(),
        "voltage_magnitudes_pu": vm.tolist(),
        "slack_p_mw": float(slack_injection.real),
        "slack_q_mvar": float(slack_injection.imag),
        "loss_mw": float(total_loss),
        "max_apparent_power_utilization": float(max_apparent_util),
        "max_current_utilization": float(max_current_util),
        "p_from_pu_by_branch": p_from_pu,
        "q_from_pu_by_branch": q_from_pu,
        "current_sq_pu_by_branch": current_sq_pu,
        "branches": branches,
    }