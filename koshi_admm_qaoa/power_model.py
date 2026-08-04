"""Branch-flow SOC model and fixed-topology physical validation.

Continuous-model revision v2 corrects the issues identified in the technical
review:

* active/reactive shedding is tied at the bus demand power factor;
* every branch uses its own MVA-derived current and apparent-power limits;
* transformer off-nominal taps are explicit (nominal ratios are in bus bases);
* switchable voltage-drop equations use a two-sided Big-M constraint;
* the slack injection and every nodal P/Q balance are explicit;
* fixed-topology solutions report cone slack, voltage-drop residuals, thermal
  utilization, angle recovery, and a nonlinear AC power-flow recovery.

The SOC result is a relaxation. ``soc_feasible`` and
``nonlinear_ac_validated`` are intentionally distinct.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import numpy as np

try:
    import cvxpy as cp
except ModuleNotFoundError:  # Topology and nonlinear checks remain importable.
    cp = None

from ac_validation import nonlinear_ac_power_flow, recover_branch_flow_angles
from network_data import Network, S_BASE


MODEL_REVISION = "continuous-model-v2"
V_MIN_SQ = 0.81       # squared pu (0.90 pu)
V_MAX_SQ = 1.21       # squared pu (1.10 pu)
M_V_SQ_PU = 2.0       # two-sided voltage-drop deactivation bound, squared pu
SHED_PENALTY = 1.0e4  # objective weight on pu active-power shedding
CONE_TIGHTNESS_TOL = 1.0e-6
VOLTAGE_RESIDUAL_TOL = 1.0e-7
ANGLE_RECOVERY_TOL_RAD = 1.0e-5


def _require_cvxpy():
    if cp is None:
        raise ModuleNotFoundError(
            "cvxpy with the CLARABEL solver is required for the SOC model"
        )


def _incidence(net: Network):
    inc_in = {i: [] for i in range(net.n_bus)}
    inc_out = {i: [] for i in range(net.n_bus)}
    for branch in net.branches:
        inc_out[branch.frm].append(branch.idx)
        inc_in[branch.to].append(branch.idx)
    return inc_in, inc_out


def _build_common(net: Network, faulted: Optional[List[int]] = None):
    _require_cvxpy()
    faulted = list(faulted or [])
    nb, nl = net.n_bus, net.n_branch
    P = cp.Variable(nl, name="P_branch_pu")
    Q = cp.Variable(nl, name="Q_branch_pu")
    Vsq = cp.Variable(nb, name="V_sq_pu")
    Isq = cp.Variable(nl, name="I_sq_pu")
    Pshed = cp.Variable(nb, name="P_shed_pu")
    Qshed = cp.Variable(nb, name="Q_shed_pu")
    Pslack = cp.Variable(name="P_slack_pu")
    Qslack = cp.Variable(name="Q_slack_pu")
    r = np.array([branch.r_pu for branch in net.branches])
    x = np.array([branch.x_pu for branch in net.branches])
    constraints = [
        Vsq[net.slack] == 1.0,
        Isq >= 0,
        Vsq >= V_MIN_SQ,
        Vsq <= V_MAX_SQ,
        Pshed >= 0,
    ]

    # Shedding is allowed only at positive-demand buses and preserves the
    # original demand power factor exactly. Qshed can be negative only if a
    # future positive-P demand has capacitive (negative-Q) demand.
    for bus in net.buses:
        pd = bus.p_load_mw / S_BASE
        qd = bus.q_load_mvar / S_BASE
        if bus.is_slack or pd <= 0:
            constraints += [Pshed[bus.idx] == 0, Qshed[bus.idx] == 0]
        else:
            constraints += [
                Pshed[bus.idx] <= pd,
                Qshed[bus.idx] == (qd / pd) * Pshed[bus.idx],
            ]

    inc_in, inc_out = _incidence(net)
    for i, bus in enumerate(net.buses):
        incoming, outgoing = inc_in[i], inc_out[i]
        loss_p = cp.sum(cp.multiply(r[incoming], Isq[incoming])) if incoming else 0
        loss_q = cp.sum(cp.multiply(x[incoming], Isq[incoming])) if incoming else 0
        p_in = (cp.sum(P[incoming]) if incoming else 0) - loss_p
        q_in = (cp.sum(Q[incoming]) if incoming else 0) - loss_q
        p_out = cp.sum(P[outgoing]) if outgoing else 0
        q_out = cp.sum(Q[outgoing]) if outgoing else 0
        p_grid = Pslack if i == net.slack else 0
        q_grid = Qslack if i == net.slack else 0
        pd = bus.p_load_mw / S_BASE
        qd = bus.q_load_mvar / S_BASE
        constraints += [
            p_in + p_grid - p_out == pd - Pshed[i],
            q_in + q_grid - q_out == qd - Qshed[i],
        ]

    for branch_index in faulted:
        constraints += [
            Isq[branch_index] == 0,
            P[branch_index] == 0,
            Q[branch_index] == 0,
        ]
    return (
        P, Q, Vsq, Isq, Pshed, Qshed, Pslack, Qslack,
        r, x, constraints, faulted,
    )


def _branch_physics(net, k, P, Q, Vsq, Isq, gate, constraints, faulted):
    """Append tap-aware branch physics for a binary or relaxed branch gate."""
    if k in faulted:
        return
    branch = net.branches[k]
    i, j = branch.frm, branch.to
    r, x = branch.r_pu, branch.x_pu
    tap = float(branch.tap_ratio_pu)
    source_v = Vsq[i] / (tap * tap)
    vdrop = source_v - 2 * (r * P[k] + x * Q[k]) + (r * r + x * x) * Isq[k]
    current_limit_sq = float(branch.current_limit_sq_pu)
    apparent_limit = float(branch.rating_pu)

    fixed_closed = isinstance(gate, (int, float)) and gate == 1 and not branch.switchable
    if fixed_closed:
        constraints += [
            Vsq[j] == vdrop,
            Isq[k] <= current_limit_sq,
        ]
    else:
        constraints += [
            Vsq[j] - vdrop <= M_V_SQ_PU * (1 - gate),
            Vsq[j] - vdrop >= -M_V_SQ_PU * (1 - gate),
            Isq[k] <= current_limit_sq * gate,
        ]

    # Sending/receiving-end MVA ratings and the series-current rating.
    constraints.append(cp.SOC(apparent_limit * gate, cp.hstack([P[k], Q[k]])))
    constraints.append(
        cp.SOC(
            apparent_limit * gate,
            cp.hstack([P[k] - r * Isq[k], Q[k] - x * Isq[k]]),
        )
    )
    constraints.append(
        cp.SOC(
            source_v + Isq[k],
            cp.hstack([2 * P[k], 2 * Q[k], source_v - Isq[k]]),
        )
    )


def _empty_result(status, error=None):
    result = {
        "model_revision": MODEL_REVISION,
        "status": status,
        "soc_feasible": False,
        "feasible": False,
        "loss_mw": np.inf,
        "shed_mw": np.inf,
        "reactive_shed_mvar": np.inf,
        "obj": np.inf,
        "nonlinear_ac_validated": False,
    }
    if error is not None:
        result["error"] = str(error)
    return result


def _soc_diagnostics(net, z, p, q, v_sq, i_sq, faulted=None):
    faulted_set = set(faulted or [])
    cone_slack = [None] * net.n_branch
    voltage_residual = [None] * net.n_branch
    apparent_utilization = [None] * net.n_branch
    current_utilization = [None] * net.n_branch
    closed_indices = []
    for k, branch in enumerate(net.branches):
        closed = k not in faulted_set and (
            not branch.switchable or bool(int(z.get(k, 1)))
        )
        if not closed:
            continue
        closed_indices.append(k)
        tap = float(branch.tap_ratio_pu)
        source_v = v_sq[branch.frm] / (tap * tap)
        cone_slack[k] = float(source_v * i_sq[k] - p[k] ** 2 - q[k] ** 2)
        predicted_v = (
            source_v
            - 2 * (branch.r_pu * p[k] + branch.x_pu * q[k])
            + (branch.r_pu ** 2 + branch.x_pu ** 2) * i_sq[k]
        )
        voltage_residual[k] = float(v_sq[branch.to] - predicted_v)
        sending_apparent = np.hypot(p[k], q[k])
        receiving_apparent = np.hypot(
            p[k] - branch.r_pu * i_sq[k],
            q[k] - branch.x_pu * i_sq[k],
        )
        apparent_utilization[k] = float(
            max(sending_apparent, receiving_apparent) / branch.rating_pu
        )
        current_utilization[k] = float(
            np.sqrt(max(i_sq[k], 0.0)) / branch.rating_pu
        )

    closed_cone = np.array([cone_slack[k] for k in closed_indices], dtype=float)
    closed_vres = np.array([voltage_residual[k] for k in closed_indices], dtype=float)
    closed_sutil = np.array([apparent_utilization[k] for k in closed_indices], dtype=float)
    closed_iutil = np.array([current_utilization[k] for k in closed_indices], dtype=float)
    angles = recover_branch_flow_angles(
        net,
        z,
        p,
        q,
        v_sq,
        faulted=faulted,
        tolerance_rad=ANGLE_RECOVERY_TOL_RAD,
    )
    max_cone_slack = float(np.max(closed_cone)) if closed_cone.size else 0.0
    min_cone_slack = float(np.min(closed_cone)) if closed_cone.size else 0.0
    max_vres = float(np.max(np.abs(closed_vres))) if closed_vres.size else 0.0
    max_sutil = float(np.max(closed_sutil)) if closed_sutil.size else 0.0
    max_iutil = float(np.max(closed_iutil)) if closed_iutil.size else 0.0
    return {
        "closed_branch_indices": closed_indices,
        "cone_slack_pu2_by_branch": cone_slack,
        "max_cone_slack_pu2": max_cone_slack,
        "min_cone_slack_pu2": min_cone_slack,
        "cone_tight": bool(
            max(abs(max_cone_slack), abs(min_cone_slack)) <= CONE_TIGHTNESS_TOL
        ),
        "cone_tightness_tolerance_pu2": CONE_TIGHTNESS_TOL,
        "voltage_drop_residual_pu2_by_branch": voltage_residual,
        "max_abs_voltage_drop_residual_pu2": max_vres,
        "voltage_drop_consistent": bool(max_vres <= VOLTAGE_RESIDUAL_TOL),
        "apparent_power_utilization_by_branch": apparent_utilization,
        "current_utilization_by_branch": current_utilization,
        "max_apparent_power_utilization": max_sutil,
        "max_current_utilization": max_iutil,
        "thermal_limits_satisfied": bool(max(max_sutil, max_iutil) <= 1.0 + 1e-7),
        "angle_recovery": angles,
    }


def solve_socp_fixed(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[List[int]] = None,
    run_nonlinear_recovery: bool = False,
) -> dict:
    """Solve the fixed-topology SOC relaxation and optionally recover AC power flow."""
    try:
        common = _build_common(net, faulted)
    except Exception as exc:
        return _empty_result(f"error:{type(exc).__name__}", exc)
    (
        P, Q, Vsq, Isq, Pshed, Qshed, Pslack, Qslack,
        r, _, constraints, faulted,
    ) = common
    for k, branch in enumerate(net.branches):
        gate = 1 if not branch.switchable else int(z.get(k, 1))
        _branch_physics(net, k, P, Q, Vsq, Isq, gate, constraints, faulted)
    loss = cp.sum(cp.multiply(r, Isq))
    problem = cp.Problem(
        cp.Minimize(loss + SHED_PENALTY * cp.sum(Pshed)), constraints
    )
    try:
        problem.solve(solver=cp.CLARABEL)
    except Exception as exc:
        return _empty_result(f"error:{type(exc).__name__}", exc)
    ok = problem.status in ("optimal", "optimal_inaccurate") and Isq.value is not None
    if not ok:
        return _empty_result(problem.status)

    p = np.asarray(P.value, dtype=float)
    q = np.asarray(Q.value, dtype=float)
    v_sq = np.asarray(Vsq.value, dtype=float)
    i_sq = np.asarray(Isq.value, dtype=float)
    p_shed_mw = np.asarray(Pshed.value, dtype=float) * S_BASE
    q_shed_mvar = np.asarray(Qshed.value, dtype=float) * S_BASE
    diagnostics = _soc_diagnostics(net, z, p, q, v_sq, i_sq, faulted)
    result = {
        "model_revision": MODEL_REVISION,
        "status": problem.status,
        "soc_feasible": True,
        "feasible": True,  # backwards-compatible alias: SOC solver feasibility only
        "loss_mw": float(np.sum(r * i_sq) * S_BASE),
        "shed_mw": float(np.sum(p_shed_mw)),
        "reactive_shed_mvar": float(np.sum(q_shed_mvar)),
        "p_shed_mw_by_bus": p_shed_mw.tolist(),
        "q_shed_mvar_by_bus": q_shed_mvar.tolist(),
        "slack_p_mw": float(Pslack.value * S_BASE),
        "slack_q_mvar": float(Qslack.value * S_BASE),
        "obj": float(problem.value),
        "vmin_pu": float(np.sqrt(max(v_sq.min(), 0.0))),
        "vmax_pu": float(np.sqrt(max(v_sq.max(), 0.0))),
        "p_branch_pu": p.tolist(),
        "q_branch_pu": q.tolist(),
        "v_sq_pu": v_sq.tolist(),
        "i_sq_pu": i_sq.tolist(),
        "diagnostics": diagnostics,
        "nonlinear_ac": None,
        "nonlinear_ac_validated": False,
    }
    if run_nonlinear_recovery:
        angle_initial = diagnostics["angle_recovery"]["angles_rad"]
        nonlinear = nonlinear_ac_power_flow(
            net,
            z,
            p_shed_mw=p_shed_mw,
            q_shed_mvar=q_shed_mvar,
            initial_v_sq_pu=v_sq,
            initial_angles_rad=angle_initial,
            faulted=faulted,
            vmin_pu=np.sqrt(V_MIN_SQ),
            vmax_pu=np.sqrt(V_MAX_SQ),
        )
        result["nonlinear_ac"] = nonlinear
        result["nonlinear_ac_validated"] = bool(nonlinear["validated"])
    return result


def solve_socp_admm_x(
    net: Network,
    z_target: np.ndarray,
    u: np.ndarray,
    rho: float,
    faulted: Optional[List[int]] = None,
):
    """ADMM x-update with the corrected continuous feasible set."""
    faulted_set = set(faulted or [])
    sw = [index for index in net.switch_indices() if index not in faulted_set]
    common = _build_common(net, faulted)
    (
        P, Q, Vsq, Isq, Pshed, _, _, _, r, _, constraints, faulted,
    ) = common
    alpha = cp.Variable(len(sw), name="alpha")
    constraints += [alpha >= 0, alpha <= 1]
    position = {branch_index: a for a, branch_index in enumerate(sw)}
    for k, branch in enumerate(net.branches):
        if k in faulted_set:
            gate = 0
        else:
            gate = alpha[position[k]] if branch.switchable else 1
        _branch_physics(net, k, P, Q, Vsq, Isq, gate, constraints, faulted)
    loss = cp.sum(cp.multiply(r, Isq))
    penalty = (rho / 2) * cp.sum_squares(alpha - z_target + u)
    problem = cp.Problem(
        cp.Minimize(loss + SHED_PENALTY * cp.sum(Pshed) + penalty),
        constraints,
    )
    problem.solve(solver=cp.CLARABEL)
    if alpha.value is None:
        raise RuntimeError(f"ADMM x-update failed with status {problem.status}")
    return (
        np.clip(alpha.value, 0, 1),
        float(np.sum(r * Isq.value) * S_BASE),
        float(np.sum(Pshed.value) * S_BASE),
    )


# ---------------- topology utilities --------------------------------------
def closed_graph(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[List[int]] = None,
) -> nx.MultiGraph:
    faulted_set = set(faulted or [])
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(net.n_bus))
    for k, branch in enumerate(net.branches):
        closed = k not in faulted_set and (
            not branch.switchable or int(z.get(k, 1))
        )
        if closed:
            graph.add_edge(branch.frm, branch.to, key=k)
    return graph


def is_connected(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[List[int]] = None,
) -> bool:
    """Return whether every nonzero-injection bus is in the slack component."""
    graph = closed_graph(net, z, faulted)
    component = nx.node_connected_component(graph, net.slack)
    return all(
        bus.idx in component
        for bus in net.buses
        if bus.p_load_mw != 0 or bus.q_load_mvar != 0 or bus.is_slack
    )


def is_radial(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[List[int]] = None,
) -> bool:
    """Radial means the energized slack component is a connected tree."""
    graph = closed_graph(net, z, faulted)
    component = nx.node_connected_component(graph, net.slack)
    edge_count = sum(
        1 for u, v, _ in graph.edges(keys=True) if u in component and v in component
    )
    return edge_count == len(component) - 1 and is_connected(net, z, faulted)


def connectivity_repair(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[List[int]] = None,
) -> Dict[int, int]:
    """Backward-compatible topology repair returning only the repaired state.

    The implementation now projects onto a spanning tree and can both close
    bridges and open cycle-forming switches.  Use ``radiality_repair`` when the
    repair actions and success flag must be archived.
    """
    return radiality_repair(net, z, faulted)["z"]


def radiality_repair(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[List[int]] = None,
) -> dict:
    """Return a cycle-safe spanning-tree projection and its exact action log."""
    from radiality import project_to_spanning_tree

    return project_to_spanning_tree(net, z, faulted)


def ac_feasibility(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[List[int]] = None,
) -> dict:
    """Run topology, SOC, tightness, angle, and nonlinear AC checks."""
    result = solve_socp_fixed(
        net,
        z,
        faulted,
        run_nonlinear_recovery=True,
    )
    connected = is_connected(net, z, faulted)
    radial = is_radial(net, z, faulted)
    result["connected"] = connected
    result["radial"] = radial
    diagnostics = result.get("diagnostics") or {}
    angle = diagnostics.get("angle_recovery") or {}
    angle["recoverable"] = bool(connected and angle.get("recoverable", False))
    if diagnostics:
        diagnostics["angle_recovery"] = angle
    nonlinear_valid = bool(result.get("nonlinear_ac_validated", False))
    result["nonlinear_power_flow_validated"] = nonlinear_valid
    result["nonlinear_ac_validated"] = bool(connected and nonlinear_valid)
    result["fixed_topology_soc_validated"] = bool(
        result.get("soc_feasible", False)
        and connected
        and diagnostics.get("cone_tight", False)
        and diagnostics.get("thermal_limits_satisfied", False)
        and diagnostics.get("voltage_drop_consistent", False)
        and angle.get("recoverable", False)
    )
    result["engineering_validated"] = bool(
        connected and radial and result["nonlinear_ac_validated"]
    )
    return result


if __name__ == "__main__":
    from network_data import build_full_network

    network = build_full_network()
    all_closed = {k: 1 for k in network.switch_indices()}
    validation = ac_feasibility(network, all_closed)
    keys = (
        "status", "soc_feasible", "loss_mw", "shed_mw", "connected",
        "radial", "nonlinear_ac_validated", "engineering_validated",
    )
    print("All-closed (meshed):", {key: validation.get(key) for key in keys})
