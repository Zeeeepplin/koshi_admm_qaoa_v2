"""Exact topology constraints and structure-aware classical baselines.

The quantum benchmark deliberately retains the historical quadratic
cycle/anti-islanding objective so that its saved numbers remain comparable.
Those quadratic terms are heuristics: they do not enforce connectivity or
radiality.  This module provides the exact topology layer used to audit and
repair their output.

For a network whose non-switchable closed branches form a forest, a topology
is a spanning tree exactly when

* the total number of closed switchable branches is
  ``n_bus - 1 - n_fixed_closed``; and
* a single-commodity flow sends one unit from the slack to every other bus.

``solve_spanning_tree_milp`` implements that formulation.  The remaining
functions provide exact small-instance enumeration, the O(n log n)
sorting/prefix solver for a uniform-cardinality QUBO, and a cycle-safe
projection of an arbitrary bitstring onto a spanning tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Dict, Iterable, Optional, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from network_data import Network
from qubo_builder import qubo_energy, z_to_dict


TOPOLOGY_REVISION = "exact-radiality-v1"


@dataclass
class _DisjointSet:
    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, n: int) -> "_DisjointSet":
        return cls(list(range(n)), [0] * n)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _faulted_set(faulted: Optional[Iterable[int]]) -> set[int]:
    return {int(index) for index in (faulted or [])}


def _closed_graph(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[Iterable[int]] = None,
) -> nx.MultiGraph:
    faulted_set = _faulted_set(faulted)
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(net.n_bus))
    for branch in net.branches:
        closed = branch.idx not in faulted_set and (
            not branch.switchable or bool(int(z.get(branch.idx, 1)))
        )
        if closed:
            graph.add_edge(branch.frm, branch.to, key=branch.idx)
    return graph


def topology_status(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[Iterable[int]] = None,
) -> dict:
    """Return unambiguous whole-network connectivity/radiality diagnostics."""
    graph = _closed_graph(net, z, faulted)
    connected = bool(net.n_bus > 0 and nx.is_connected(graph))
    closed_count = int(graph.number_of_edges())
    radial = bool(connected and closed_count == net.n_bus - 1)
    cycle_rank = int(closed_count - net.n_bus + nx.number_connected_components(graph))
    return {
        "topology_revision": TOPOLOGY_REVISION,
        "connected": connected,
        "radial": radial,
        "closed_branch_count": closed_count,
        "required_tree_branch_count": net.n_bus - 1,
        "component_count": int(nx.number_connected_components(graph)),
        "cycle_rank": cycle_rank,
    }


def fixed_forest_status(
    net: Network,
    faulted: Optional[Iterable[int]] = None,
) -> dict:
    """Describe the mandatory non-switchable subgraph.

    A spanning tree containing every fixed branch exists only if this graph is
    acyclic and the available switchable branches can connect its components.
    """
    faulted_set = _faulted_set(faulted)
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(net.n_bus))
    fixed_indices = []
    for branch in net.branches:
        if not branch.switchable and branch.idx not in faulted_set:
            graph.add_edge(branch.frm, branch.to, key=branch.idx)
            fixed_indices.append(branch.idx)
    components = nx.number_connected_components(graph)
    cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + components
    target = net.n_bus - 1 - graph.number_of_edges()
    return {
        "fixed_branch_indices": fixed_indices,
        "fixed_closed_count": int(graph.number_of_edges()),
        "fixed_component_count": int(components),
        "fixed_cycle_rank": int(cycle_rank),
        "fixed_is_forest": bool(cycle_rank == 0),
        "target_closed_switches": int(target),
    }


def project_to_spanning_tree(
    net: Network,
    z: Dict[int, int],
    faulted: Optional[Iterable[int]] = None,
    change_penalty: float = 1.0e3,
) -> dict:
    """Project ``z`` onto a spanning tree while retaining fixed branches.

    Kruskal's algorithm first inserts all mandatory fixed branches.  Candidate
    closed switches are cheaper than candidate open switches, so the projection
    minimizes the number of 0->1 changes before using impedance as a stable
    tie-breaker.  Unlike the old close-only repair, it also opens cycle-forming
    switches.
    """
    faulted_set = _faulted_set(faulted)
    original = {index: int(z.get(index, 1)) for index in net.switch_indices()}
    dsu = _DisjointSet.create(net.n_bus)
    fixed = []
    for branch in net.branches:
        if not branch.switchable and branch.idx not in faulted_set:
            if not dsu.union(branch.frm, branch.to):
                return {
                    "success": False,
                    "reason": "mandatory fixed branches contain a cycle",
                    "z": original,
                    "opened_switches": [],
                    "closed_switches": [],
                    **topology_status(net, original, faulted),
                }
            fixed.append(branch.idx)

    candidates = []
    for branch in net.branches:
        if not branch.switchable or branch.idx in faulted_set:
            continue
        was_open = 1 - original[branch.idx]
        cost = change_penalty * was_open + branch.r_pu + branch.x_pu
        candidates.append((cost, branch.idx, branch))
    candidates.sort(key=lambda item: (item[0], item[1]))

    selected = set()
    for _, branch_index, branch in candidates:
        if dsu.union(branch.frm, branch.to):
            selected.add(branch_index)
        if len(fixed) + len(selected) == net.n_bus - 1:
            break

    repaired = {
        index: int(index in selected and index not in faulted_set)
        for index in net.switch_indices()
    }
    status = topology_status(net, repaired, faulted)
    opened = sorted(index for index, value in original.items() if value and not repaired[index])
    closed = sorted(index for index, value in original.items() if not value and repaired[index])
    return {
        "success": bool(status["radial"]),
        "reason": None if status["radial"] else "available switches cannot form a spanning tree",
        "z": repaired,
        "opened_switches": opened,
        "closed_switches": closed,
        "n_switch_changes": len(opened) + len(closed),
        **status,
    }


def solve_spanning_tree_milp(
    net: Network,
    switch_costs: Optional[Sequence[float]] = None,
    faulted: Optional[Iterable[int]] = None,
) -> dict:
    """Solve the exact single-commodity-flow spanning-tree formulation.

    The binary variables are the switch states.  One signed continuous flow is
    assigned to every physical branch; the slack supplies ``n_bus - 1`` units
    and each other bus consumes one.  Flow can traverse a switchable branch only
    when that switch is closed.  Together with the exact edge cardinality this
    is equivalent to a connected spanning tree.
    """
    started = perf_counter()
    faulted_set = _faulted_set(faulted)
    switch_indices = net.switch_indices()
    n_switch, n_branch = len(switch_indices), net.n_branch
    position = {branch_index: offset for offset, branch_index in enumerate(switch_indices)}
    fixed = fixed_forest_status(net, faulted)
    if not fixed["fixed_is_forest"]:
        return {
            "success": False,
            "status": "infeasible: mandatory fixed branches contain a cycle",
            "time_s": perf_counter() - started,
            "topology_revision": TOPOLOGY_REVISION,
        }
    target = fixed["target_closed_switches"]
    if target < 0 or target > sum(index not in faulted_set for index in switch_indices):
        return {
            "success": False,
            "status": "infeasible: spanning-tree cardinality is outside switch bounds",
            "time_s": perf_counter() - started,
            "topology_revision": TOPOLOGY_REVISION,
        }

    if switch_costs is None:
        costs = np.array(
            [net.branches[index].r_pu + net.branches[index].x_pu for index in switch_indices],
            dtype=float,
        )
    else:
        costs = np.asarray(switch_costs, dtype=float)
        if costs.shape != (n_switch,):
            raise ValueError(f"switch_costs must have shape {(n_switch,)}, got {costs.shape}")

    n_variables = n_switch + n_branch
    objective = np.zeros(n_variables)
    objective[:n_switch] = costs
    lower = np.full(n_variables, -np.inf)
    upper = np.full(n_variables, np.inf)
    lower[:n_switch] = 0.0
    upper[:n_switch] = 1.0
    flow_limit = float(max(net.n_bus - 1, 1))
    lower[n_switch:] = -flow_limit
    upper[n_switch:] = flow_limit
    for branch in net.branches:
        if branch.idx in faulted_set:
            lower[n_switch + branch.idx] = 0.0
            upper[n_switch + branch.idx] = 0.0
            if branch.switchable:
                upper[position[branch.idx]] = 0.0

    rows = []
    row_lower = []
    row_upper = []

    # Signed flow balance: outgoing minus incoming is N-1 at the slack and -1
    # at every other bus.
    for bus in range(net.n_bus):
        row = np.zeros(n_variables)
        for branch in net.branches:
            column = n_switch + branch.idx
            if branch.frm == bus:
                row[column] += 1.0
            if branch.to == bus:
                row[column] -= 1.0
        rhs = float(net.n_bus - 1 if bus == net.slack else -1)
        rows.append(row)
        row_lower.append(rhs)
        row_upper.append(rhs)

    # Exact tree edge count after mandatory fixed branches are included.
    row = np.zeros(n_variables)
    for index, offset in position.items():
        if index not in faulted_set:
            row[offset] = 1.0
    rows.append(row)
    row_lower.append(float(target))
    row_upper.append(float(target))

    # -M z_k <= f_k <= M z_k for every switchable branch.
    for branch_index, offset in position.items():
        flow_column = n_switch + branch_index
        upper_row = np.zeros(n_variables)
        upper_row[flow_column] = 1.0
        upper_row[offset] = -flow_limit
        rows.append(upper_row)
        row_lower.append(-np.inf)
        row_upper.append(0.0)

        lower_row = np.zeros(n_variables)
        lower_row[flow_column] = -1.0
        lower_row[offset] = -flow_limit
        rows.append(lower_row)
        row_lower.append(-np.inf)
        row_upper.append(0.0)

    constraints = LinearConstraint(
        np.vstack(rows), np.asarray(row_lower), np.asarray(row_upper)
    )
    integrality = np.zeros(n_variables, dtype=int)
    integrality[:n_switch] = 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"presolve": True},
    )
    elapsed = perf_counter() - started
    if not result.success or result.x is None:
        return {
            "success": False,
            "status": str(result.message),
            "time_s": elapsed,
            "topology_revision": TOPOLOGY_REVISION,
        }
    bits = np.rint(result.x[:n_switch]).astype(int)
    z = {index: int(bits[offset]) for index, offset in position.items()}
    status = topology_status(net, z, faulted)
    return {
        "success": bool(result.success and status["radial"]),
        "status": str(result.message),
        "objective": float(result.fun),
        "time_s": elapsed,
        "x": bits,
        "z": z,
        "flow": np.asarray(result.x[n_switch:], dtype=float),
        "target_closed_switches": int(target),
        **status,
    }


def solve_uniform_cardinality_qubo(meta: dict, tolerance: float = 1.0e-10) -> dict:
    """Solve a linear-plus-uniform-all-pairs QUBO by sorting and prefix sums.

    This is the exact O(n log n) baseline requested by the technical review.
    It intentionally rejects the actual cycle/anti-islanding-coupled QUBO when
    its off-diagonal coefficients are nonuniform.
    """
    started = perf_counter()
    linear = np.asarray(meta["linear"], dtype=float)
    quadratic = np.triu(np.asarray(meta["Q"], dtype=float), 1)
    n = linear.size
    offdiag = quadratic[np.triu_indices(n, 1)]
    uniform = float(offdiag[0]) if offdiag.size else 0.0
    if offdiag.size and not np.allclose(offdiag, uniform, atol=tolerance, rtol=0.0):
        raise ValueError(
            "sorting/prefix baseline applies only to a uniform all-pairs QUBO; "
            "the supplied QUBO has nonuniform cycle/anti-islanding coupling"
        )
    order = np.argsort(linear, kind="stable")
    prefix = np.concatenate(([0.0], np.cumsum(linear[order])))
    weights = np.arange(n + 1)
    energies = prefix + uniform * weights * (weights - 1) / 2.0
    best_weight = int(np.argmin(energies))
    bits = np.zeros(n, dtype=int)
    bits[order[:best_weight]] = 1
    return {
        "success": True,
        "solver": "uniform-cardinality sorting/prefix",
        "time_s": perf_counter() - started,
        "objective": float(energies[best_weight]),
        "x": bits,
        "hamming_weight": best_weight,
        "uniform_offdiagonal_coefficient": uniform,
    }


def solve_exact_enumeration(meta: dict, predicate=None) -> dict:
    """Exactly enumerate a small QUBO, optionally filtering configurations."""
    started = perf_counter()
    n = int(meta["n_qubits"])
    best_energy = np.inf
    best_bits = None
    best_bitstrings = []
    feasible_count = 0
    for values in product((0, 1), repeat=n):
        bits = np.fromiter(values, dtype=int, count=n)
        if predicate is not None and not predicate(bits):
            continue
        feasible_count += 1
        energy = qubo_energy(bits, meta)
        if energy < best_energy - 1.0e-12:
            best_energy = energy
            best_bits = bits.copy()
            best_bitstrings = [bits.astype(int).tolist()]
        elif abs(energy - best_energy) <= 1.0e-12:
            best_bitstrings.append(bits.astype(int).tolist())
    return {
        "success": best_bits is not None,
        "solver": "exact binary enumeration",
        "time_s": perf_counter() - started,
        "objective": float(best_energy) if best_bits is not None else None,
        "x": best_bits,
        "all_minimizers_variable_order": best_bitstrings,
        "minimizer_count": len(best_bitstrings),
        "feasible_configurations": feasible_count,
    }


def solve_exact_radial_qubo(
    net: Network,
    meta: dict,
    faulted: Optional[Iterable[int]] = None,
) -> dict:
    """Minimize the historical QUBO exactly over only spanning-tree states."""
    return solve_exact_enumeration(
        meta,
        predicate=lambda bits: topology_status(
            net, z_to_dict(bits, meta), faulted
        )["radial"],
    )


def surrogate_radiality_audit(
    net: Network,
    meta: dict,
    faulted: Optional[Iterable[int]] = None,
) -> dict:
    """Enumerate false positives/negatives of the pairwise cycle surrogate."""
    n = int(meta["n_qubits"])
    target = int(meta["K_target"])
    cycle_q = np.triu(meta["quadratic_components"]["cycle"], 1)
    total = cardinality_states = radial_states = 0
    cycle_false_positive = cycle_false_negative = 0
    global_best = np.inf
    global_min_radial = 0
    global_min_count = 0
    radial_best = np.inf
    for values in product((0, 1), repeat=n):
        bits = np.fromiter(values, dtype=int, count=n)
        total += 1
        status = topology_status(net, z_to_dict(bits, meta), faulted)
        radial = status["radial"]
        if radial:
            radial_states += 1
        card_ok = int(bits.sum()) == target
        if card_ok:
            cardinality_states += 1
        cycle_value = float(bits @ cycle_q @ bits)
        if radial and cycle_value > 1.0e-12:
            cycle_false_positive += 1
        if card_ok and not radial and abs(cycle_value) <= 1.0e-12:
            cycle_false_negative += 1
        energy = qubo_energy(bits, meta)
        if energy < global_best - 1.0e-12:
            global_best = energy
            global_min_count = 1
            global_min_radial = int(radial)
        elif abs(energy - global_best) <= 1.0e-12:
            global_min_count += 1
            global_min_radial += int(radial)
        if radial:
            radial_best = min(radial_best, energy)
    return {
        "topology_revision": TOPOLOGY_REVISION,
        "total_configurations": total,
        "target_cardinality_configurations": cardinality_states,
        "radial_configurations": radial_states,
        "cycle_surrogate_false_positive_radial_states": cycle_false_positive,
        "cycle_surrogate_false_negative_nonradial_target_cardinality_states": (
            cycle_false_negative
        ),
        "global_qubo_minimum": float(global_best),
        "global_qubo_minimizers": global_min_count,
        "global_qubo_minimizers_that_are_radial": global_min_radial,
        "best_radial_qubo_objective": (
            float(radial_best) if np.isfinite(radial_best) else None
        ),
        "hard_radiality_objective_gap": (
            float(radial_best - global_best) if np.isfinite(radial_best) else None
        ),
    }
