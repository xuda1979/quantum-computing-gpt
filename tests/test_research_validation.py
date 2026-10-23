from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

from quantum_posttrain.allocation import (
    advantage_bounds,
    certified_coefficient_mass,
    conservative_coefficients,
)
from quantum_posttrain.contextual import coherent_control_phase_witness
from quantum_posttrain.research_validation import run_research_validation
from quantum_posttrain.scale_validation import run_scale_validation


class ResearchValidationTests(unittest.TestCase):
    def test_advantage_intervals_contain_every_corner_and_certify_signs(self):
        intervals = [(-1.0, -0.4), (-0.2, 0.7), (0.1, 0.9), (-0.8, 0.3)]
        bounds = advantage_bounds(intervals)
        coefficients = conservative_coefficients(intervals)
        for rewards in itertools.product(*[(low, high) for low, high in intervals]):
            mean = sum(rewards) / len(rewards)
            for index, reward in enumerate(rewards):
                advantage = reward - mean
                self.assertLessEqual(bounds[index][0] - 1e-12, advantage)
                self.assertLessEqual(advantage, bounds[index][1] + 1e-12)
                if coefficients[index] != 0:
                    self.assertGreaterEqual(coefficients[index] * advantage, -1e-12)
                    self.assertLessEqual(abs(coefficients[index]), abs(advantage) + 1e-12)

    def test_nested_interval_refinement_does_not_reduce_certified_mass(self):
        coarse = [(-1.0, 1.0), (-0.8, 0.9), (-0.7, 0.6), (-0.3, 0.8)]
        narrow = [(-0.4, 0.5), (-0.2, 0.3), (-0.5, 0.2), (0.1, 0.4)]
        weights = [0.7, 1.1, 0.3, 2.0]
        self.assertGreaterEqual(
            certified_coefficient_mass(narrow, weights),
            certified_coefficient_mass(coarse, weights) - 1e-12,
        )

    def test_controlled_phase_context_exposes_global_phase(self):
        result = coherent_control_phase_witness(3.141592653589793)
        self.assertAlmostEqual(result["standalone_process_fidelity"], 1.0, places=12)
        self.assertAlmostEqual(result["controlled_process_fidelity"], 0.0, places=12)
        self.assertAlmostEqual(result["witness_state_overlap"], 0.0, places=12)
        self.assertAlmostEqual(result["candidate_control_x"], -1.0, places=12)

    def test_randomized_research_properties(self):
        result = run_research_validation(trials=300, seed=20261023)
        self.assertEqual(result["status"], "PASS")

    def test_scale_harness_writes_disjoint_verified_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_scale_validation(Path(directory), roots=80, variants_per_root=4)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["records"], 320)
            self.assertEqual(sum(result["roots_by_split"].values()), 80)
            self.assertEqual(sum(result["records_by_split"].values()), 320)


if __name__ == "__main__":
    unittest.main()
