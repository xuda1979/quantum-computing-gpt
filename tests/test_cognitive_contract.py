"""Planning contracts must never masquerade as a runnable cognitive trainer."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_cognitive_contract.py"


class CognitiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("cognitive_validator", SCRIPT)
        cls.validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.validator)
        cls.example = json.loads((ROOT / "configs/qwen38/cognitive_contract.example.json").read_text())
        cls.rd = json.loads((ROOT / "configs/qwen38/rd_contract.example.json").read_text())

    def child(self, arm):
        value = copy.deepcopy(self.example)
        value["arms"][arm] = True
        value["experiment"] = {
            "arm": arm, "matched_control_id": "R0-control",
            "parent_contract_sha256": "a" * 64,
            "measurement_sha256": "b" * 64,
        }
        value["crux_reward"]["enabled"] = arm == "crux_reward"
        return value

    def test_default_and_each_single_factor_plan_validate(self):
        self.assertEqual(self.validator.validate_contract(self.example, self.rd), [])
        for arm in self.example["arms"]:
            with self.subTest(arm=arm):
                self.assertEqual(self.validator.validate_contract(self.child(arm), self.rd), [])

    def test_multiple_factors_and_missing_lineage_rejected(self):
        value = self.child("world_mutation")
        value["arms"]["toy_probe"] = True
        self.assertTrue(self.validator.validate_contract(value, self.rd))
        for key in ("matched_control_id", "parent_contract_sha256", "measurement_sha256"):
            value = self.child("world_mutation")
            value["experiment"][key] = None
            with self.subTest(key=key):
                self.assertTrue(self.validator.validate_contract(value, self.rd))

    def test_reward_requires_its_own_arm(self):
        value = self.child("toy_probe")
        value["crux_reward"]["enabled"] = True
        self.assertTrue(self.validator.validate_contract(value, self.rd))
        value = self.child("crux_reward")
        value["crux_reward"]["enabled"] = False
        self.assertTrue(self.validator.validate_contract(value, self.rd))

    def test_incompatible_and_unsafe_changes_rejected(self):
        mutations = [
            ("status", "resolved"),
            ("model_repo_id", "Qwen/Qwen3.5-27B"),
            ("teacher_imitation_enabled", True),
            ("unknown_control", True),
        ]
        for key, replacement in mutations:
            value = copy.deepcopy(self.example)
            value[key] = replacement
            with self.subTest(key=key):
                self.assertTrue(self.validator.validate_contract(value, self.rd))
        for section, key, replacement in [
            ("crux_reward", "signed_advantage_multiplication", True),
            ("crux_metrics", "claim_mutual_information", True),
            ("frontier_sampling", "final_test_feedback", True),
            ("frontier_sampling", "uniform_exploration_floor", 0),
            ("evidence_folding", "source_pointers_required", False),
            ("evidence_folding", "protect_task_constraints", False),
            ("evidence_folding", "reset_recurrent_state", False),
            ("toy_probe", "falsifiable_prediction_required", False),
            ("toy_probe", "max_wall_seconds", 3),
            ("deferred", "ffn_compression", True),
            ("deferred", "smaller_student", True),
            ("deferred", "mtp_value_head", True),
            ("budgets", "pilot_actor_token_cap", 8_000_001),
            ("budgets", "active_context_token_cap", 100),
        ]:
            value = copy.deepcopy(self.example)
            value[section][key] = replacement
            with self.subTest(section=section, key=key):
                self.assertTrue(self.validator.validate_contract(value, self.rd))

    def test_invalid_parent_and_nonfinite_json_rejected(self):
        rd = copy.deepcopy(self.rd)
        rd["training"]["project_sft_enabled"] = True
        self.assertTrue(self.validator.validate_contract(self.example, rd))
        for literal in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            with self.subTest(literal=literal), self.assertRaises(ValueError):
                self.validator.load_json_text('{"fraction": ' + literal + '}')

    def test_duplicate_json_keys_rejected(self):
        for text in (
            '{"experiment": {"arm": "disabled", "arm": "crux_reward"}}',
            '{"training_route": "rl_only_sapo", "training_route": "sft"}',
            '{"arms": {"crux_reward": false, "crux_reward": true}}',
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "duplicate"):
                self.validator.load_json_text(text)

    def test_tampered_parent_sum_rejected(self):
        rd = copy.deepcopy(self.rd)
        rd["data"]["domain_shares"]["quantum_code"] += 0.01
        errors = self.validator.validate_contract(self.example, rd)
        self.assertTrue(any("domain_shares must sum to one" in error for error in errors), errors)

    def test_cli_is_planning_only_and_fails_closed(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "PLANNING_VALID")
        self.assertIs(report["training_authorized"], False)
        result = subprocess.run([sys.executable, str(SCRIPT), "--require-resolved"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIs(json.loads(result.stdout)["training_authorized"], False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"broken": NaN}')
            result = subprocess.run([sys.executable, str(SCRIPT), "--contract", str(path)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIs(json.loads(result.stdout)["training_authorized"], False)


if __name__ == "__main__":
    unittest.main()
