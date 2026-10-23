from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quantum_posttrain.generator import generate_tasks
from quantum_posttrain.llm_boundary import adjudicate_proposal
from quantum_posttrain.quantum import equivalent_up_to_global_phase
from quantum_posttrain.smoke import load_adapter, run
from quantum_posttrain.splits import assign_root_splits, assert_root_disjoint
from quantum_posttrain.contracts import Task


class PipelineTests(unittest.TestCase):
    def test_harness_generation_has_verified_targets(self):
        tasks = generate_tasks()
        self.assertEqual(len(tasks), 16)
        self.assertTrue(all(equivalent_up_to_global_phase(t.source_gates, t.target_gates)
                            for t in tasks))

    def test_llm_cannot_replace_trusted_contract_or_label(self):
        roots = {"root-xh": ("X", "H")}
        proposal = {
            "root_id": "root-xh", "candidate_gates": ["X", "H"],
            "target_gates": ["I"], "split": "test", "contract": "always_pass",
        }
        decision = adjudicate_proposal(proposal, roots)
        self.assertEqual(decision.status, "VERIFIED")
        self.assertEqual(decision.task.target_gates, ("H", "Z"))
        self.assertEqual(decision.task.split, "train")
        self.assertNotEqual(adjudicate_proposal(
            {"root_id": "unknown", "candidate_gates": []}, roots).status, "VERIFIED")

    def test_llm_wording_requires_review(self):
        result = adjudicate_proposal(
            {"root_id": "r", "candidate_gates": ["H", "H"],
             "problem_text": "A newly generated question"},
            {"r": ("H", "H")},
        )
        self.assertEqual(result.status, "REVIEW_REQUIRED")
        self.assertEqual(adjudicate_proposal([], {}).status, "REJECTED")
        self.assertEqual(adjudicate_proposal({"root_id": []}, {}).status, "REJECTED")

    def test_task_loader_rejects_contract_and_gate_tampering(self):
        base = {
            "task_id": "t", "root_id": "r", "family_id": "f", "split": "train",
            "source_gates": ["H"], "target_gates": ["H"],
            "contract": "one_qubit_unitary_up_to_global_phase_v1",
        }
        for field, value in (("contract", "always_pass"), ("target_gates", ["MAGIC"])):
            altered = {**base, field: value}
            with self.assertRaises(ValueError):
                Task.from_dict(altered)

    def test_split_keeps_each_root_in_one_partition(self):
        tasks = generate_tasks(None)
        by_split = {name: [t for t in tasks if t.split == name]
                    for name in ("train", "dev", "test")}
        assert_root_disjoint(*by_split.values())
        self.assertEqual(sum(map(len, by_split.values())), len(tasks))
        repeated = [tasks[0], tasks[0]]
        assigned = assign_root_splits(repeated)
        self.assertEqual(assigned[0].split, assigned[1].split)

    def test_small_adapter_train_save_reload_and_verified_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_path = Path(directory) / "adapter.npz"
            report = run(adapter_path, steps=1600)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["verified_rollouts"], 16)
            self.assertLess(report["final_loss"], report["initial_loss"])
            load_adapter(adapter_path)


if __name__ == "__main__":
    unittest.main()
