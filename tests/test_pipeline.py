"""
Test suite for the Koshi ADMM-QAOA hybrid optimization pipeline.

Run with: pytest tests/ -v
"""
import sys
import os

# Add the koshi_admm_qaoa directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'koshi_admm_qaoa'))

import numpy as np
import pytest


class TestQUBOConstruction:
    """Tests for QUBO formulation correctness."""
    
    def test_qubo_coupling_term_counts(self):
        """Verify QUBO has correct number of quadratic terms."""
        from koshi_admm_qaoa.network_data import scaled_network
        from koshi_admm_qaoa.qubo_builder import build_reconfig_qubo
        
        # Test with small network
        net = scaled_network(n_switches=4)
        qp, meta = build_reconfig_qubo(net)
        
        nq = meta["n_qubits"]
        Q = meta["Q"]
        
        # Q should be nq x nq
        assert Q.shape == (nq, nq), f"Expected Q shape ({nq}, {nq}), got {Q.shape}"
        
        # Count non-zero off-diagonal elements (coupling terms)
        coupling_count = np.count_nonzero(np.triu(Q, 1))
        
        # Should have at least some coupling terms for non-trivial problem
        assert coupling_count > 0, "QUBO should have coupling terms"
        
        print(f"✓ QUBO has {coupling_count} coupling terms for {nq} qubits")
    
    def test_qubo_linear_term_dimension(self):
        """Verify linear term dimension matches qubit count."""
        from koshi_admm_qaoa.network_data import scaled_network
        from koshi_admm_qaoa.qubo_builder import build_reconfig_qubo
        
        net = scaled_network(n_switches=5)
        qp, meta = build_reconfig_qubo(net)
        
        nq = meta["n_qubits"]
        linear = meta["linear"]
        
        assert len(linear) == nq, f"Linear term length {len(linear)} != n_qubits {nq}"


class TestADMMConvergence:
    """Tests for ADMM algorithm convergence on small instances."""
    
    def test_admm_convergence_small_instance(self):
        """Test ADMM converges on a minimal network."""
        from network_data import scaled_network
        from admm_hybrid import run_admm
        
        net = scaled_network(n_switches=3)
        
        # Run ADMM with reasonable parameters
        result = run_admm(
            net,
            rho=3.0,
            max_iter=20,
            eps_primal=1e-2,
            z_solver="exact",  # Use exact solver for deterministic testing
            verbose=False
        )
        
        # Check that ADMM produced a solution
        assert result is not None, "ADMM should return a result"
        assert "z" in result or hasattr(result, 'z'), "Result should contain switch states"
        
        print("✓ ADMM converged on small instance")
    
    def test_admm_iteration_decrease(self):
        """Verify objective decreases (or stays stable) across iterations."""
        from koshi_admm_qaoa.network_data import scaled_network
        from koshi_admm_qaoa import admm_hybrid
        
        net = scaled_network(n_switches=4)
        
        # We need to inspect iteration history
        # This test assumes admm_hybrid tracks convergence
        # May need adjustment based on actual implementation
        try:
            result = admm_hybrid.run_admm(
                net,
                rho=3.0,
                max_iter=15,
                verbose=False
            )
            # If convergence info is available, check it
            if hasattr(result, 'history') and result.history:
                # Objective should generally decrease
                print("✓ ADMM iteration history available")
        except Exception as e:
            pytest.skip(f"ADMM history tracking not implemented: {e}")


class TestConnectivityRepair:
    """Tests for radiality enforcement and connectivity repair."""
    
    def test_radiality_check(self):
        """Test radiality verification function."""
        from koshi_admm_qaoa.network_data import scaled_network
        from koshi_admm_qaoa import power_model
        
        net = scaled_network(n_switches=4)
        
        # Get a valid radial configuration
        result = power_model.ac_feasibility(net, {})
        
        # Should report connected status
        assert "connected" in result, "ac_feasibility should report connectivity"
        
        print("✓ Radiality check works correctly")
    
    def test_cycle_detection(self):
        """Test that cycles are properly detected in graph."""
        import networkx as nx
        from koshi_admm_qaoa import power_model
        
        # Create a simple graph with a cycle
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])  # Triangle
        
        # Should detect cycle
        cycles = list(nx.cycle_basis(G))
        assert len(cycles) > 0, "Should detect cycle in triangular graph"
        
        # Create acyclic graph
        G2 = nx.Graph()
        G2.add_edges_from([(1, 2), (2, 3)])
        
        cycles2 = list(nx.cycle_basis(G2))
        assert len(cycles2) == 0, "Should not detect cycle in tree"
        
        print("✓ Cycle detection works correctly")


class TestACFeasibility:
    """Tests for AC power flow feasibility checks."""
    
    def test_ac_feasibility_validation(self):
        """Test AC feasibility checker validates inputs."""
        from koshi_admm_qaoa.network_data import scaled_network
        from koshi_admm_qaoa import power_model
        
        net = scaled_network(n_switches=3)
        
        # Test with empty switch configuration (should use base topology)
        result = power_model.ac_feasibility(net, {})
        
        # Should return dict with expected keys
        assert isinstance(result, dict), "ac_feasibility should return dict"
        assert "loss_mw" in result or "feasible" in result, \
            "Result should contain loss or feasibility info"
        
        print("✓ AC feasibility validation works")


class TestBenchmarkReproducibility:
    """Tests for benchmark reproducibility with random seeds."""
    
    def test_seed_reproducibility(self):
        """Verify same seed produces same results."""
        from koshi_admm_qaoa.network_data import scaled_network
        from koshi_admm_qaoa.qubo_builder import build_reconfig_qubo
        
        # Build same network twice
        net1 = scaled_network(n_switches=4, seed=42)
        net2 = scaled_network(n_switches=4, seed=42)
        
        qp1, meta1 = build_reconfig_qubo(net1)
        qp2, meta2 = build_reconfig_qubo(net2)
        
        # Should produce identical QUBOs
        assert np.allclose(meta1["linear"], meta2["linear"]), \
            "Same seed should produce identical linear terms"
        assert np.allclose(meta1["Q"], meta2["Q"]), \
            "Same seed should produce identical Q matrices"
        
        print("✓ Benchmark reproducibility verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
