"""Small immutable task contracts used by generators and verifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CONTRACT_ID = "one_qubit_unitary_up_to_global_phase_v1"
SUPPORTED_GATES = frozenset({"I", "X", "Z", "H"})
VALID_SPLITS = frozenset({"train", "dev", "test"})


@dataclass(frozen=True)
class Task:
    task_id: str
    root_id: str
    family_id: str
    split: str
    source_gates: tuple[str, ...]
    target_gates: tuple[str, ...]
    contract: str = CONTRACT_ID

    def __post_init__(self) -> None:
        for field_name in ("task_id", "root_id", "family_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a nonempty string")
        if not isinstance(self.split, str) or self.split not in VALID_SPLITS:
            raise ValueError("split must be train, dev, or test")
        if self.contract != CONTRACT_ID:
            raise ValueError(f"unsupported task contract: {self.contract}")
        for field_name in ("source_gates", "target_gates"):
            gates = getattr(self, field_name)
            if not isinstance(gates, tuple) or any(
                not isinstance(gate, str) or gate not in SUPPORTED_GATES for gate in gates
            ):
                raise ValueError(f"{field_name} must contain only supported gate names")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_gates"] = list(self.source_gates)
        value["target_gates"] = list(self.target_gates)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Task":
        if not isinstance(value, dict):
            raise ValueError("task record must be a JSON object")
        expected = {
            "task_id", "root_id", "family_id", "split", "source_gates",
            "target_gates", "contract",
        }
        if set(value) != expected:
            raise ValueError(f"task fields must be exactly {sorted(expected)}")
        for field_name in ("source_gates", "target_gates"):
            if not isinstance(value[field_name], list):
                raise ValueError(f"{field_name} must be a JSON list")
        task = cls(
            task_id=value["task_id"],
            root_id=value["root_id"],
            family_id=value["family_id"],
            split=value["split"],
            source_gates=tuple(value["source_gates"]),
            target_gates=tuple(value["target_gates"]),
            contract=value["contract"],
        )
        return task
