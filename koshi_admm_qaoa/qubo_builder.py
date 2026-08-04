"""Build the historical topology-coupled QUBO for the switch sub-problem.

The cardinality, pairwise cycle, and pairwise anti-islanding terms make the
actual QUBO nonuniform and nonseparable.  They do *not* enforce a spanning tree:

* cardinality alone does not imply connectivity;
* ``sum z_i z_j`` penalizes a closed two-edge parallel loop, but
  over-penalizes valid selections on longer cycles; and
* the quadratic anti-islanding expression equals ``prod(1-z_k)`` (up to its
  omitted constant) only for degree-one and degree-two buses.

The terms are retained as explicitly labelled heuristic biases so the frozen
quantum results remain reproducible.  Exact connectivity/radiality constraints,
the sorting/prefix baseline, and cycle-safe projection are in ``radiality.py``.

The builder returns a qiskit-optimization QuadraticProgram (for QAOA/QRAO/exact)
and a fast numpy evaluator (for brute force / simulated annealing). Physical
branches are indexed by ``k``; binary-vector positions are indexed by ``ell``
and follow ``meta['variable_order_branch_indices']``.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
try:
    from qiskit_optimization import QuadraticProgram
except ModuleNotFoundError:  # Allow artifact auditing without the quantum stack.
    QuadraticProgram = None
from network_data import Network


def _fixed_rank(net: Network, faulted=None) -> int:
    """Rank (spanning-forest edge count) of the always-closed (fixed) sub-graph."""
    faulted = set(faulted or [])
    G = nx.Graph()
    G.add_nodes_from(range(net.n_bus))
    for b in net.branches:
        if not b.switchable and b.idx not in faulted:
            G.add_edge(b.frm, b.to)
    return net.n_bus - nx.number_connected_components(G)


def _fixed_topology_status(net: Network, faulted=None) -> dict:
    faulted = set(faulted or [])
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(net.n_bus))
    for branch in net.branches:
        if not branch.switchable and branch.idx not in faulted:
            graph.add_edge(branch.frm, branch.to, key=branch.idx)
    components = nx.number_connected_components(graph)
    cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + components
    available = nx.MultiGraph()
    available.add_nodes_from(range(net.n_bus))
    for branch in net.branches:
        if branch.idx not in faulted:
            available.add_edge(branch.frm, branch.to, key=branch.idx)
    return {
        "fixed_closed_count": int(graph.number_of_edges()),
        "fixed_component_count": int(components),
        "fixed_cycle_rank": int(cycle_rank),
        "fixed_is_forest": bool(cycle_rank == 0),
        "available_graph_connected": bool(nx.is_connected(available)),
    }


def fundamental_cycles(net: Network, faulted=None) -> List[List[int]]:
    """Heuristic loop list expressed as SWITCHABLE branch indices.

    Parallel circuits (same endpoints) give 2-branch loops; larger loops come
    from a cycle basis of the simplified graph mapped back to branch indices.
    """
    cycles: List[List[int]] = []
    faulted = set(faulted or [])
    sw = {index for index in net.switch_indices() if index not in faulted}

    # (a) parallel-circuit 2-loops among switchable branches
    seen: Dict[Tuple[int, int], List[int]] = {}
    for b in net.branches:
        if b.switchable and b.idx not in faulted:
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
        if b.idx in faulted:
            continue
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


def _bus_incident_switch(net: Network, faulted=None):
    """For each bus, the switchable branch indices incident to it, and whether the
    bus has ANY always-closed (fixed) branch keeping it energized."""
    inc = {i: [] for i in range(net.n_bus)}
    has_fixed = {i: False for i in range(net.n_bus)}
    faulted = set(faulted or [])
    for b in net.branches:
        if b.idx in faulted:
            continue
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
    faulted: Optional[List[int]] = None,
    build_program: bool = True,
) -> Tuple[QuadraticProgram, dict]:
    """Return (QuadraticProgram, meta).  meta has the numpy Q/linear + K_target.

    Coupling (off-diagonal) comes from three topology-motivated heuristics:
      * cardinality (all-pairs), * fundamental cycles (loop pairs),
      * isolation / anti-islanding (pairs of switches sharing a bus).
    Set lambda_card=0 for meshed (transmission) ADMM operation; keep it >0 for the
    classical radial-reconfiguration benchmark.
    """
    faulted = sorted({int(index) for index in (faulted or [])})
    faulted_set = set(faulted)
    sw = [index for index in net.switch_indices() if index not in faulted_set]
    n = len(sw)
    pos = {k: i for i, k in enumerate(sw)}      # branch idx -> variable index
    r = np.array([net.branches[k].r_pu for k in sw])

    fixed_status = _fixed_topology_status(net, faulted)
    # Historical rank-based target retained for numerical reproducibility.  It
    # is a valid tree edge count only when the fixed subgraph is a forest.  The
    # exact formulation in radiality.py rejects cyclic mandatory subgraphs.
    K_target = (net.n_bus - 1) - _fixed_rank(net, faulted)
    K_target = max(0, min(n, K_target))

    lin_components = {
        "consensus": np.zeros(n),
        "loss": np.zeros(n),
        "cardinality": np.zeros(n),
        "cycle": np.zeros(n),
        "anti_islanding": np.zeros(n),
    }
    q_components = {name: np.zeros((n, n)) for name in lin_components}

    # (1) ADMM consensus (separable part)
    if alpha is not None and rho > 0:
        u = np.zeros(n) if u is None else u
        delta = alpha + u                        # (alpha - z + u)^2, z^2=z
        lin_components["consensus"] += (rho / 2.0) * (1.0 - 2.0 * delta)

    # (2) loss / closing bias: mild preference to close low-resistance branches
    lin_components["loss"] += loss_bias * (r / (r.max() + 1e-9))

    # (3) spanning-tree cardinality  lambda_card (sum z - K)^2   -> couples all pairs
    if lambda_card > 0:
        lin_components["cardinality"] += lambda_card * (1.0 - 2.0 * K_target)
        for i in range(n):
            for j in range(i + 1, n):
                q_components["cardinality"][i, j] += 2.0 * lambda_card

    # (4) pairwise cycle heuristic. Structurally exact for detecting a closed
    # two-edge parallel loop, but still only a finite penalty;
    # for longer cycles it also penalizes valid tree subsets with |C|-1 edges.
    for cyc in fundamental_cycles(net, faulted):
        idx = [pos[k] for k in cyc if k in pos]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = sorted((idx[a], idx[b]))
                q_components["cycle"][i, j] += lambda_cycle

    # (5) pairwise anti-islanding heuristic. It matches prod(1-z_k), up to the
    # omitted constant, for degree one or two; it is not a higher-degree product
    # decomposition and is never treated as a hard connectivity constraint.
    if lambda_iso > 0:
        inc, has_fixed = _bus_incident_switch(net, faulted)
        for bus, brs in inc.items():
            if has_fixed[bus] or bus == net.slack:
                continue
            idx = [pos[k] for k in brs if k in pos]
            if len(idx) < 2:
                if idx:                      # single feed -> must stay closed
                    lin_components["anti_islanding"][idx[0]] -= lambda_iso
                continue
            for a in range(len(idx)):
                lin_components["anti_islanding"][idx[a]] -= lambda_iso
                for b in range(a + 1, len(idx)):
                    i, j = sorted((idx[a], idx[b]))
                    q_components["anti_islanding"][i, j] += lambda_iso

    constant = 0.0
    lin = sum(lin_components.values(), np.zeros(n))
    Q = sum(q_components.values(), np.zeros((n, n)))

    def component_stats(values, quadratic=False):
        data = values[np.triu_indices(n, 1)] if quadratic else values
        nz = data[np.abs(data) > 1e-12]
        return {
            "nonzero": int(nz.size),
            "coefficient_min": float(nz.min()) if nz.size else None,
            "coefficient_max": float(nz.max()) if nz.size else None,
        }

    # ---- assemble QuadraticProgram ----
    qp = None
    if build_program:
        if QuadraticProgram is None:
            raise ModuleNotFoundError(
                "qiskit-optimization is required to build the QuadraticProgram; "
                "use build_program=False for component auditing."
            )
        qp = QuadraticProgram("koshi_reconfig")
        for k in sw:
            qp.binary_var(name=f"z{k}")
        quad = {(f"z{sw[i]}", f"z{sw[j]}"): Q[i, j]
                for i in range(n) for j in range(i + 1, n) if abs(Q[i, j]) > 1e-12}
        linear = {f"z{sw[i]}": lin[i] for i in range(n)}
        qp.minimize(constant=constant, linear=linear, quadratic=quad)

    meta = {
        "constant": constant,
        "switch_branches": sw,  # backward-compatible alias
        "variable_order_branch_indices": sw,
        "var_pos": pos, "K_target": K_target,
        "faulted_branches": faulted,
        "linear": lin, "Q": Q,
        "upper_triangular_quadratic_coefficients": np.triu(Q, 1),
        "n_binary_variables": n,
        "n_qubits": n,  # standard-QAOA qubits; legacy artifact name
        "n_offdiag": int(np.count_nonzero(np.triu(Q, 1))),
        "linear_components": lin_components,
        "quadratic_components": q_components,
        "component_stats": {
            name: {
                "linear": component_stats(lin_components[name]),
                "quadratic": component_stats(q_components[name], quadratic=True),
            }
            for name in lin_components
        },
        "topology_encoding": {
            "revision": "heuristic-topology-qubo-v1",
            "hard_connectivity_enforced": False,
            "hard_radiality_enforced": False,
            "cardinality_target_definition": "n_bus - 1 - rank(fixed subgraph)",
            "cardinality_target_is_tree_valid": bool(fixed_status["fixed_is_forest"]),
            "fixed_subgraph": fixed_status,
            "exact_formulation": "radiality.solve_spanning_tree_milp",
            "faulted_branches_excluded_from_decision_vector": faulted,
        },
        "notation_and_objective_convention": {
            "physical_branch_index": "k",
            "binary_vector_position": "ell",
            "binary_variable": "z_ell",
            "variable_order": "variable_order_branch_indices",
            "objective_symbol": "f_Q",
            "objective_sense": "minimize",
            "binary_to_pauli_mapping": "z_ell = (1 - Z_ell) / 2",
            "returned_array_order": "QuadraticProgram variable order",
        },
    }
    return qp, meta


def qubo_energy(z: np.ndarray, meta: dict) -> float:
    """Evaluate ``f_Q(z)`` in archived variable order."""
    lin, Q = meta["linear"], meta["Q"]
    return float(meta.get("constant", 0.0) + lin @ z + z @ np.triu(Q, 1) @ z)


def qubo_to_ising_coefficients(meta: dict) -> dict:
    """Return exact coefficients for the mapping ``z=(1-Z)/2``.

    ``Q`` stores each pair once in its strict upper triangle. For a basis-state
    bit vector ``z``, ``f_Q(z) = kappa + h @ s + s @ J @ s`` with
    ``s = 1 - 2*z``.
    """
    linear = np.asarray(meta["linear"], dtype=float)
    upper = np.triu(np.asarray(meta["Q"], dtype=float), 1)
    if upper.shape != (linear.size, linear.size):
        raise ValueError("Q shape does not match the linear coefficient vector")
    incident_pair_sum = upper.sum(axis=0) + upper.sum(axis=1)
    return {
        "kappa": float(
            meta.get("constant", 0.0)
            + 0.5 * linear.sum()
            + 0.25 * upper.sum()
        ),
        "h": -0.5 * linear - 0.25 * incident_pair_sum,
        "J": 0.25 * upper,
        "binary_to_pauli_mapping": "z_ell = (1 - Z_ell) / 2",
        "objective_sense": "minimize",
    }


def ising_diagonal_energy(z: np.ndarray, coefficients: dict) -> float:
    """Evaluate the mapped Ising operator on computational-basis bits ``z``."""
    bits = np.asarray(z, dtype=int)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("z must contain only binary values")
    spins = 1.0 - 2.0 * bits
    return float(
        coefficients["kappa"]
        + np.asarray(coefficients["h"], dtype=float) @ spins
        + spins @ np.asarray(coefficients["J"], dtype=float) @ spins
    )


def z_to_dict(z: np.ndarray, meta: dict) -> Dict[int, int]:
    """Map variable-order entries to physical branch indices."""
    order = meta.get("variable_order_branch_indices", meta["switch_branches"])
    return {k: int(z[i]) for i, k in enumerate(order)}


if __name__ == "__main__":
    from network_data import build_full_network
    net = build_full_network()
    qp, meta = build_reconfig_qubo(net)
    print(f"QUBO: {meta['n_qubits']} qubits, K_target={meta['K_target']}, "
          f"off-diagonal (coupling) terms = {meta['n_offdiag']}")
    print("=> SEPARABLE?" , meta["n_offdiag"] == 0)
    print("hard radiality enforced?", meta["topology_encoding"]["hard_radiality_enforced"])
    print("cycles found:", len(fundamental_cycles(net)))
