"""Dependency-light checks for continuous-model-v2 metadata and AC recovery."""
from __future__ import annotations

import unittest

import numpy as np

from ac_validation import nonlinear_ac_power_flow, recover_branch_flow_angles
from network_data import build_full_network


class ContinuousModelTests(unittest.TestCase):
    def setUp(self):
        self.net = build_full_network()
        self.all_closed = {k: 1 for k in self.net.switch_indices()}

    def test_branch_limits_and_taps_are_explicit(self):
        self.assertTrue(all(branch.rating_mva > 0 for branch in self.net.branches))
        self.assertTrue(all(branch.tap_ratio_pu > 0 for branch in self.net.branches))
        self.assertTrue(
            all(branch.current_limit_sq_pu > 0 for branch in self.net.branches)
        )
        self.assertTrue(all(branch.parameter_status for branch in self.net.branches))
        self.assertTrue(all(branch.source_ref for branch in self.net.branches))
        inaruwa_400_220 = self.net.branches[0]
        inaruwa_220_132 = self.net.branches[1]
        self.assertEqual(inaruwa_400_220.physical_units, 3)
        self.assertAlmostEqual(inaruwa_400_220.rating_mva, 945.0)
        self.assertAlmostEqual(inaruwa_400_220.x_pu, 0.12 * 100.0 / 945.0)
        self.assertEqual(inaruwa_220_132.physical_units, 2)
        self.assertAlmostEqual(inaruwa_220_132.rating_mva, 320.0)
        self.assertAlmostEqual(inaruwa_220_132.x_pu, 0.12 * 100.0 / 320.0)

    def test_nonlinear_power_flow_converges_and_checks_limits(self):
        result = nonlinear_ac_power_flow(self.net, self.all_closed)
        self.assertTrue(result["converged"])
        self.assertLess(result["max_mismatch_mva"], 1e-3)
        # The present estimated operating point is deliberately not declared
        # validated: the check detects its low voltage and/or thermal overload.
        self.assertFalse(result["validated"])
        self.assertTrue(not result["voltage_ok"] or not result["thermal_ok"])
        net_demand = sum(bus.p_load_mw for bus in self.net.buses)
        self.assertAlmostEqual(
            result["slack_p_mw"] - net_demand,
            result["loss_mw"],
            places=7,
        )

    def test_exact_ac_branch_flows_recover_consistent_angles(self):
        result = nonlinear_ac_power_flow(self.net, self.all_closed)
        p = np.array(
            [0.0 if value is None else value for value in result["p_from_pu_by_branch"]]
        )
        q = np.array(
            [0.0 if value is None else value for value in result["q_from_pu_by_branch"]]
        )
        v_sq = np.square(result["voltage_magnitudes_pu"])
        angles = recover_branch_flow_angles(self.net, self.all_closed, p, q, v_sq)
        self.assertTrue(angles["recoverable"])
        self.assertLess(angles["max_residual_rad"], 1e-5)


if __name__ == "__main__":
    unittest.main()
