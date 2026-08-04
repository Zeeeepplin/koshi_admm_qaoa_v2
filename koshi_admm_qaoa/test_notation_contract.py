"""Regression tests for notation, ordering, and QUBO--Ising conventions."""
from __future__ import annotations

import itertools

import numpy as np

from artifact_pipeline import _tex_text
from network_data import scaled_network
from qubo_builder import (
    build_reconfig_qubo,
    ising_diagonal_energy,
    qubo_energy,
    qubo_to_ising_coefficients,
    z_to_dict,
)


def test_qubo_to_ising_mapping_preserves_every_basis_energy():
    _, meta = build_reconfig_qubo(scaled_network(4), build_program=False)
    coefficients = qubo_to_ising_coefficients(meta)
    for values in itertools.product((0, 1), repeat=meta["n_binary_variables"]):
        bits = np.asarray(values, dtype=int)
        assert np.isclose(
            qubo_energy(bits, meta),
            ising_diagonal_energy(bits, coefficients),
            atol=1.0e-12,
        )


def test_vector_positions_map_to_archived_physical_branch_order():
    _, meta = build_reconfig_qubo(scaled_network(4), build_program=False)
    bits = np.asarray([1, 0, 1, 0], dtype=int)
    physical = z_to_dict(bits, meta)
    order = meta["variable_order_branch_indices"]
    assert [physical[index] for index in order] == bits.tolist()
    convention = meta["notation_and_objective_convention"]
    assert convention["objective_sense"] == "minimize"
    assert convention["binary_to_pauli_mapping"] == "z_ell = (1 - Z_ell) / 2"


def test_provenance_values_are_safe_for_latex_tables():
    assert _tex_text(r"12% A&B_x\{y}") == (
        r"12\% A\&B\_x\textbackslash{}\{y\}"
    )
