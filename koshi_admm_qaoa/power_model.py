"""
power_model.py
==============
AC branch-flow (DistFlow) second-order-cone (SOC) model for the Eastern-Nepal
sub-system, plus radiality / connectivity utilities and an AC-feasibility check.

Two entry points:
  * solve_socp_fixed(net, z)      -> exact continuous OPF for a FIXED switch
                                      configuration z (0/1 per switchable branch).
                                      Used for ground-truth benchmarking and for
                                      AC-feasibility validation of any topology.
  * solve_socp_admm_x(net, ...)   -> the ADMM x-update: switches relaxed to
                                      alpha in [0,1] with an L2 consensus penalty
                                      towards the quantum iterate z (+ dual u).

Both minimise real-power losses + a large load-shedding penalty subject to the
SOC branch-flow constraints; open branches carry no flow (Big-M).

Relaxation honesty: the SOC branch-flow relaxation is provably exact only for
radial operation. We therefore (a) allow meshed intermediate iterates inside
ADMM, and (b) validate the FINAL configuration with is_radial()/is_connected()
and a fixed-z AC solve.  CLARABEL is used (open-source conic solver).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import cvxpy as cp
import networkx as nx
from network_data import Network, S_BASE

V_MIN_SQ = 0.81   # 0.90 pu
V_MAX_SQ = 1.21   # 1.10 pu
M_I = 40.0        # Big-M on squared current (pu) for an open branch
M_V = 2.0         # Big-M on the voltage-drop slack for an open branch
SHED_PENALTY = 1.0e4


def _incidence(net: Network):
    inc_in = {i: [] for i in range(net.n_bus)}
    inc_out = {i: [] for i in range(net.n_bus)}
    for b in net.branches:
        inc_out[b.frm].append(b.idx)
        inc_in[b.to].append(b.idx)
    return inc_in, inc_out


def _build_common(net: Network, faulted: Optional[List[int]] = None):
    faulted = faulted or []
    nb, nl = net.n_bus, net.n_branch
    P = cp.Variable(nl); Q = cp.Variable(nl)
    Vsq = cp.Variable(nb); Isq = cp.Variable(nl)
    Pshed = cp.Variable(nb); Qshed = cp.Variable(nb)
    r = np.array([b.r_pu for b in net.branches])
    x = np.array([b.x_pu for b in net.branches])
    cons = [Vsq[net.slack] == 1.0, Isq >= 0, Vsq >= V_MIN_SQ, Vsq <= V_MAX_SQ,
            Pshed >= 0, Qshed >= 0]
    # shedding only on load buses (never shed generation / slack)
    for bus in net.buses:
        pd = bus.p_load_mw / S_BASE
        qd = bus.q_load_mvar / S_BASE
        if bus.is_slack or pd <= 0:
            cons += [Pshed[bus.idx] == 0, Qshed[bus.idx] == 0]
        else:
            cons += [Pshed[bus.idx] <= pd, Qshed[bus.idx] <= qd]
    inc_in, inc_out = _incidence(net)
    for i in range(nb):
        if i == net.slack:
            continue
        li, lo = inc_in[i], inc_out[i]
        lossP = cp.sum(cp.multiply(r[li], Isq[li])) if li else 0
        lossQ = cp.sum(cp.multiply(x[li], Isq[li])) if li else 0
        Pin = (cp.sum(P[li]) if li else 0) - lossP
        Qin = (cp.sum(Q[li]) if li else 0) - lossQ
        Pout = cp.sum(P[lo]) if lo else 0
        Qout = cp.sum(Q[lo]) if lo else 0
        pd = net.buses[i].p_load_mw / S_BASE
        qd = net.buses[i].q_load_mvar / S_BASE
        cons += [Pin - Pout == pd - Pshed[i], Qin - Qout == qd - Qshed[i]]
    for f in faulted:
        cons += [Isq[f] == 0, P[f] == 0, Q[f] == 0]
    return P, Q, Vsq, Isq, Pshed, Qshed, r, x, cons, faulted


def _branch_physics(net, k, P, Q, Vsq, Isq, gate, cons, faulted):
    """gate = 1 (closed / fixed line), 0 (open), or a cvxpy expr in [0,1]."""
    if k in faulted:
        return
    b = net.branches[k]
    i, j, r, x = b.frm, b.to, b.r_pu, b.x_pu
    vdrop = Vsq[i] - 2 * (r * P[k] + x * Q[k]) + (r * r + x * x) * Isq[k]
    if isinstance(gate, (int, float)) and gate == 1 and not b.switchable:
        cons += [Vsq[j] == vdrop, Isq[k] <= M_I]
    else:
        cons += [Vsq[j] - vdrop <= M_V * (1 - gate),
                 Vsq[j] - vdrop >= -M_V * (1 - gate),
                 Isq[k] <= M_I * gate]
    cons.append(cp.SOC(Vsq[i] + Isq[k], cp.vstack([2 * P[k], 2 * Q[k], Vsq[i] - Isq[k]])))


def solve_socp_fixed(net: Network, z: Dict[int, int],
                     faulted: Optional[List[int]] = None) -> dict:
    """Exact continuous OPF for a fixed switch configuration z {branch_idx:0/1}."""
    P, Q, Vsq, Isq, Pshed, Qshed, r, x, cons, faulted = _build_common(net, faulted)
    for k, b in enumerate(net.branches):
        gate = 1 if not b.switchable else int(z.get(k, 1))
        _branch_physics(net, k, P, Q, Vsq, Isq, gate, cons, faulted)
    loss = cp.sum(cp.multiply(r, Isq))
    obj = cp.Minimize(loss + SHED_PENALTY * cp.sum(Pshed))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception as e:
        return {"status": f"error:{e}", "feasible": False, "loss_mw": np.inf,
                "shed_mw": np.inf, "obj": np.inf}
    ok = prob.status in ("optimal", "optimal_inaccurate") and Isq.value is not None
    return {
        "status": prob.status, "feasible": bool(ok),
        "loss_mw": float(np.sum(r * Isq.value) * S_BASE) if ok else np.inf,
        "shed_mw": float(np.sum(Pshed.value) * S_BASE) if ok else np.inf,
        "obj": float(prob.value) if ok else np.inf,
        "vmin_pu": float(np.sqrt(Vsq.value.min())) if ok else None,
        "vmax_pu": float(np.sqrt(Vsq.value.max())) if ok else None,
    }


def solve_socp_admm_x(net: Network, z_target: np.ndarray, u: np.ndarray,
                      rho: float, faulted: Optional[List[int]] = None):
    """ADMM x-update: switches relaxed to alpha in [0,1] with consensus penalty."""
    sw = net.switch_indices()
    P, Q, Vsq, Isq, Pshed, Qshed, r, x, cons, faulted = _build_common(net, faulted)
    alpha = cp.Variable(len(sw)); cons += [alpha >= 0, alpha <= 1]
    amap = {k: a for a, k in enumerate(sw)}
    for k, b in enumerate(net.branches):
        gate = alpha[amap[k]] if b.switchable else 1
        _branch_physics(net, k, P, Q, Vsq, Isq, gate, cons, faulted)
    loss = cp.sum(cp.multiply(r, Isq))
    penalty = (rho / 2) * cp.sum_squares(alpha - z_target + u)
    prob = cp.Problem(cp.Minimize(loss + SHED_PENALTY * cp.sum(Pshed) + penalty), cons)
    prob.solve(solver=cp.CLARABEL)
    if alpha.value is None:
        raise RuntimeError("ADMM x-update infeasible")
    return (np.clip(alpha.value, 0, 1),
            float(np.sum(r * Isq.value) * S_BASE),
            float(np.sum(Pshed.value) * S_BASE))


# ---------------- topology utilities ----------------
def closed_graph(net: Network, z: Dict[int, int]) -> nx.MultiGraph:
    G = nx.MultiGraph(); G.add_nodes_from(range(net.n_bus))
    for k, b in enumerate(net.branches):
        closed = 1 if not b.switchable else int(z.get(k, 1))
        if closed:
            G.add_edge(b.frm, b.to, key=k)
    return G


def is_connected(net: Network, z: Dict[int, int]) -> bool:
    """All buses with load/gen energized (in the slack's component)."""
    G = closed_graph(net, z)
    comp = nx.node_connected_component(G, net.slack)
    for bus in net.buses:
        if (bus.p_load_mw != 0 or bus.is_slack) and bus.idx not in comp:
            return False
    return True


def is_radial(net: Network, z: Dict[int, int]) -> bool:
    """Radial = the energized sub-graph is a forest (no closed loops)."""
    G = closed_graph(net, z)
    n_edges = G.number_of_edges()
    n_nodes = len(nx.node_connected_component(G, net.slack))
    # edges within slack component
    e = sum(1 for u, v, k in G.edges(keys=True)
            if u in nx.node_connected_component(G, net.slack))
    return e == n_nodes - 1 and is_connected(net, z)


def connectivity_repair(net: Network, z: Dict[int, int]) -> Dict[int, int]:
    """Greedy anti-islanding repair: close the lowest-impedance open switchable
    branches needed to bring every load/gen bus into the slack's component.
    Standard feasibility-repair step for reconfiguration heuristics."""
    z = dict(z)
    def needed(zz):
        G = closed_graph(net, zz)
        comp = nx.node_connected_component(G, net.slack)
        return [b.idx for b in net.buses
                if (b.p_load_mw != 0 or b.is_slack) and b.idx not in comp]
    guard = 0
    while needed(z) and guard < net.n_branch + 5:
        guard += 1
        G = closed_graph(net, z)
        comp = nx.node_connected_component(G, net.slack)
        # candidate open switches bridging energized <-> unenergized
        cands = []
        for k, b in enumerate(net.branches):
            if b.switchable and int(z.get(k, 1)) == 0:
                a, c = b.frm, b.to
                if (a in comp) ^ (c in comp):
                    cands.append((b.r_pu + b.x_pu, k))
        if not cands:
            break
        cands.sort()
        z[cands[0][1]] = 1
    return z


def ac_feasibility(net: Network, z: Dict[int, int],
                   faulted: Optional[List[int]] = None) -> dict:
    """Full validation of a candidate topology."""
    res = solve_socp_fixed(net, z, faulted)
    res["connected"] = is_connected(net, z)
    res["radial"] = is_radial(net, z)
    return res


if __name__ == "__main__":
    from network_data import build_full_network
    net = build_full_network()
    z_all = {k: 1 for k in net.switch_indices()}   # all switches closed (fully meshed)
    r = ac_feasibility(net, z_all)
    print("All-closed (meshed):", {k: r[k] for k in
          ("status", "feasible", "loss_mw", "shed_mw", "connected", "radial", "vmin_pu")})
