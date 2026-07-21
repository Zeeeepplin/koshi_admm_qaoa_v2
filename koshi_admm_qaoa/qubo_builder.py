"""
qubo_builder.py
===============
Builds the *radiality-coupled* QUBO for the switch sub-problem.

WHY THIS IS THE KEY FIX
-----------------------
The original implementation used a SEPARABLE objective
    sum_k (rho/2)(alpha_k - z_k + u_k)^2
which has NO z_i z_j cross-terms -> it decouples into independent single-qubit
problems whose optimum is just z_k = round(alpha_k+u_k).  On such a QUBO, QAOA
does nothing that a one-line rounding could not do.

Here the QUBO additionally encodes RADIALITY / CONNECTIVITY, which is what makes
network reconfiguration a genuine combinatorial problem:
  * spanning-tree CARDINALITY:  lambda_card ( sum_k z_k - K_target )^2
        -> couples EVERY pair of switch variables (off-diagonal 2*lambda terms).
  * fundamental-CYCLE penalties: for every independent loop C,
        lambda_cycle * sum_{i<j in C} z_i z_j
        -> couples switches that share a loop (soft "open >=1 branch per loop").

Both terms introduce real z_i z_j coupling, so the Ising Hamiltonian is
non-diagonal-in-a-trivial-way and QAOA/QRAO actually have work to do.
Radiality here is a SOFT bias; the final topology is hard-validated by
power_model.is_radial()/ac_feasibility() (documented, honest).

The builder returns a qiskit-optimization QuadraticProgram (for QAOA/QRAO/exact)
and a fast numpy evaluator (for brute force / simulated annealing).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
from qiskit_optimization import QuadraticProgram
from network_data import Network


def _fixed_rank(net: Network) -> int:
    """Rank (spanning-forest edge count) of the always-closed (fixed) sub-graph."""
    G = nx.Graph()
    G.add_nodes_from(range(net.n_bus))
    for b in net.branches:
        if not b.switchable:
            G.add_edge(b.frm, b.to)
    return net.n_bus - nx.number_connected_components(G)


def fundamental_cycles(net: Network) -> List[List[int]]:
    """Independent loops expressed as lists of SWITCHABLE branch indices.

    Parallel circuits (same endpoints) give 2-branch loops; larger loops come
    from a cycle basis of the simplified graph mapped back to branch indices.
    """
    cycles: List[List[int]] = []
    sw = set(net.switch_indices())

    # (a) parallel-circuit 2-loops among switchable branches
    seen: Dict[Tuple[int, int], List[int]] = {}
    for b in net.branches:
        if b.switchable:
            key = (min(b.frm, b.to), max(b.frm, b.to))
            seen.setdefault(key, []).append(b.idx)
    for key, idxs in seen.items():
        for a in range(len(idxs)):
            for c in range(a + 1, len(idxs)):
                cycles.append([idxs[a], idxs[c]])

    # (b) larger loops from a cycle basis of the simple graph
    Gs = nx.Graph()
    edge_pick: Dict[Tuple[int, int], int] = {}
    for b in net.branches:
        key = (min(b.frm, b.to), max(b.frm, b.to))
        if key not in edge_pick:            # one representative edge
            edge_pick[key] = b.idx
            Gs.add_edge(*key)
    for cyc_nodes in nx.cycle_basis(Gs):
        bl = []
        for a in range(len(cyc_nodes)):
            u, v = cyc_nodes[a], cyc_nodes[(a + 1) % len(cyc_nodes)]
            k = edge_pick[(min(u, v), max(u, v))]
            if k in sw:
                bl.append(k)
        if len(bl) >= 2:
            cycles.append(bl)
    return cycles


def _bus_incident_switch(net: Network):
    """For each bus, the switchable branch indices incident to it, and whether the
    bus has ANY always-closed (fixed) branch keeping it energized."""
    inc = {i: [] for i in range(net.n_bus)}
    has_fixed = {i: False for i in range(net.n_bus)}
    for b in net.branches:
        if b.switchable:
            inc[b.frm].append(b.idx); inc[b.to].append(b.idx)
        else:
            has_fixed[b.frm] = True; has_fixed[b.to] = True
    return inc, has_fixed


def build_reconfig_qubo(
    net: Network,
    alpha: Optional[np.ndarray] = None,      # ADMM consensus target (per switch)
    u: Optional[np.ndarray] = None,          # ADMM scaled dual (per switch)
    rho: float = 0.0,                        # ADMM penalty weight
    lambda_card: float = 3.0,                # spanning-tree cardinality (reconfig mode)
    lambda_cycle: float = 1.5,               # loop / radiality coupling
    lambda_iso: float = 6.0,                 # anti-islanding (connectivity) coupling
    loss_bias: float = 5.0,
) -> Tuple[QuadraticProgram, dict]:
    """Return (QuadraticProgram, meta).  meta has the numpy Q/linear + K_target.

    Coupling (off-diagonal) comes from THREE physically-motivated terms:
      * cardinality (all-pairs), * fundamental cycles (loop pairs),
      * isolation / anti-islanding (pairs of switches sharing a bus).
    Set lambda_card=0 for meshed (transmission) ADMM operation; keep it >0 for the
    classical radial-reconfiguration benchmark.
    """
    sw = net.switch_indices()
    n = len(sw)
    pos = {k: i for i, k in enumerate(sw)}      # branch idx -> variable index
    r = np.array([net.branches[k].r_pu for k in sw])

    K_target = (net.n_bus - 1) - _fixed_rank(net)   # closed switchable in a tree
    K_target = max(0, min(n, K_target))

    lin = np.zeros(n)
    Q = np.zeros((n, n))

    # (1) ADMM consensus (separable part)
    if alpha is not None and rho > 0:
        u = np.zeros(n) if u is None else u
        delta = alpha + u                        # (alpha - z + u)^2, z^2=z
        lin += (rho / 2.0) * (1.0 - 2.0 * delta)

    # (2) loss / closing bias: mild preference to close low-resistance branches
    lin += loss_bias * (r / (r.max() + 1e-9))

    # (3) spanning-tree cardinality  lambda_card (sum z - K)^2   -> couples all pairs
    if lambda_card > 0:
        lin += lambda_card * (1.0 - 2.0 * K_target)
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] += 2.0 * lambda_card

    # (4) fundamental-cycle penalties (loop coupling -> radiality bias)
    for cyc in fundamental_cycles(net):
        idx = [pos[k] for k in cyc if k in pos]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = sorted((idx[a], idx[b]))
                Q[i, j] += lambda_cycle

    # (5) isolation / anti-islanding: for a bus with no fixed feed, penalise
    #     opening ALL its switchable branches -> lambda_iso * prod(1 - z_k),
    #     decomposed pairwise (adds +lambda_iso z_i z_j coupling, -lambda_iso z_i).
    if lambda_iso > 0:
        inc, has_fixed = _bus_incident_switch(net)
        for bus, brs in inc.items():
            if has_fixed[bus] or bus == net.slack:
                continue
            idx = [pos[k] for k in brs if k in pos]
            if len(idx) < 2:
                if idx:                      # single feed -> must stay closed
                    lin[idx[0]] -= lambda_iso
                continue
            for a in range(len(idx)):
                lin[idx[a]] -= lambda_iso
                for b in range(a + 1, len(idx)):
                    i, j = sorted((idx[a], idx[b]))
                    Q[i, j] += lambda_iso

    # ---- assemble QuadraticProgram ----
    qp = QuadraticProgram("koshi_reconfig")
    for k in sw:
        qp.binary_var(name=f"z{k}")
    quad = {(f"z{sw[i]}", f"z{sw[j]}"): Q[i, j]
            for i in range(n) for j in range(i + 1, n) if abs(Q[i, j]) > 1e-12}
    linear = {f"z{sw[i]}": lin[i] for i in range(n)}
    qp.minimize(linear=linear, quadratic=quad)

    meta = {
        "switch_branches": sw, "var_pos": pos, "K_target": K_target,
        "linear": lin, "Q": Q, "n_qubits": n,
        "n_offdiag": int(np.count_nonzero(np.triu(Q, 1))),
    }
    return qp, meta


def qubo_energy(z: np.ndarray, meta: dict) -> float:
    """Fast evaluator: z is a 0/1 vector aligned with meta['switch_branches']."""
    lin, Q = meta["linear"], meta["Q"]
    return float(lin @ z + z @ np.triu(Q, 1) @ z)


def z_to_dict(z: np.ndarray, meta: dict) -> Dict[int, int]:
    return {k: int(z[i]) for i, k in enumerate(meta["switch_branches"])}


if __name__ == "__main__":
    from network_data import build_full_network
    net = build_full_network()
    qp, meta = build_reconfig_qubo(net)
    print(f"QUBO: {meta['n_qubits']} qubits, K_target={meta['K_target']}, "
          f"off-diagonal (coupling) terms = {meta['n_offdiag']}")
    print("=> SEPARABLE?" , meta["n_offdiag"] == 0,
          "(the original toy QUBO had 0 coupling terms; this one is coupled)")
    print("cycles found:", len(fundamental_cycles(net)))
