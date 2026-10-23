"""CPU-only low-rank adapter training and verified rollout smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .generator import generate_tasks
from .quantum import GATE_NAMES, canonical_circuit, equivalent_up_to_global_phase


def label_vocabulary() -> tuple[tuple[str, ...], ...]:
    return tuple(sorted({canonical_circuit(pair) for pair in
                         ((a, b) for a in GATE_NAMES for b in GATE_NAMES)},
                        key=lambda item: (len(item), item)))


def base_weight(seed: int, n_features: int, n_labels: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.025, size=(n_features, n_labels))


def _adam_step(param, grad, state, step: int, lr: float):
    m, v = state
    m *= 0.9
    m += 0.1 * grad
    v *= 0.999
    v += 0.001 * grad * grad
    mhat = m / (1.0 - 0.9 ** step)
    vhat = v / (1.0 - 0.999 ** step)
    param -= lr * mhat / (np.sqrt(vhat) + 1e-8)


def train_adapter(steps: int = 1600, seed: int = 20261023):
    tasks = generate_tasks("train")
    labels = label_vocabulary()
    label_to_id = {label: i for i, label in enumerate(labels)}
    x = np.eye(len(GATE_NAMES) ** 2, dtype=np.float64)
    y = np.array([label_to_id[task.target_gates] for task in tasks], dtype=np.int64)
    frozen = base_weight(seed, x.shape[1], len(labels))
    rank = len(labels)
    rng = np.random.default_rng(seed + 1)
    a = rng.normal(0.0, 0.02, size=(x.shape[1], rank))
    b = np.zeros((rank, len(labels)), dtype=np.float64)
    states = [(np.zeros_like(a), np.zeros_like(a)),
              (np.zeros_like(b), np.zeros_like(b))]
    initial_logits = x @ frozen + (x @ a) @ b

    for step in range(1, steps + 1):
        hidden = x @ a
        logits = x @ frozen + hidden @ b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        grad_logits = probs
        grad_logits[np.arange(len(y)), y] -= 1.0
        grad_logits /= len(y)
        grad_b = hidden.T @ grad_logits
        grad_a = x.T @ (grad_logits @ b.T)
        _adam_step(a, grad_a, states[0], step, 0.04)
        _adam_step(b, grad_b, states[1], step, 0.04)

    def loss(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=1)) + logits.max(axis=1)
        return float(np.mean(logsumexp - logits[np.arange(len(y)), y]))

    final_logits = x @ frozen + (x @ a) @ b
    return tasks, labels, frozen, a, b, {
        "steps": steps,
        "train_examples": len(tasks),
        "initial_loss": loss(initial_logits),
        "final_loss": loss(final_logits),
        "train_accuracy": float(np.mean(final_logits.argmax(axis=1) == y)),
    }


def save_adapter(path: Path, labels, a, b, seed: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, A=a, B=b, labels=np.asarray(
        [" ".join(label) for label in labels], dtype="U16"))
    frozen = base_weight(seed, a.shape[0], b.shape[1])
    fingerprint = hashlib.sha256(frozen.tobytes()).hexdigest()
    manifest = {
        "format": "numpy_lora_smoke_v1",
        "seed": seed,
        "rank": int(a.shape[1]),
        "base_fingerprint_sha256": fingerprint,
        "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_adapter(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        a = np.asarray(archive["A"], dtype=np.float64)
        b = np.asarray(archive["B"], dtype=np.float64)
        labels = tuple(tuple(x.split()) if x else () for x in archive["labels"].tolist())
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["format"] != "numpy_lora_smoke_v1":
        raise ValueError("unsupported adapter format")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["adapter_sha256"]:
        raise ValueError("adapter hash mismatch")
    frozen = base_weight(manifest["seed"], a.shape[0], b.shape[1])
    if hashlib.sha256(frozen.tobytes()).hexdigest() != manifest["base_fingerprint_sha256"]:
        raise ValueError("base model fingerprint mismatch")
    return frozen, a, b, labels, manifest


def rollout(source_gates, frozen, a, b, labels):
    index = GATE_NAMES.index(source_gates[0]) * len(GATE_NAMES) + GATE_NAMES.index(source_gates[1])
    features = np.eye(len(GATE_NAMES) ** 2, dtype=np.float64)[index]
    predicted = int(np.argmax(features @ frozen + (features @ a) @ b))
    return labels[predicted]


def run(output: Path, steps: int = 1600) -> dict:
    seed = 20261023
    tasks, labels, _, a, b, metrics = train_adapter(steps=steps, seed=seed)
    save_adapter(output, labels, a, b, seed)
    frozen, loaded_a, loaded_b, loaded_labels, manifest = load_adapter(output)
    checked = 0
    for task in tasks:
        candidate = rollout(task.source_gates, frozen, loaded_a, loaded_b, loaded_labels)
        if not equivalent_up_to_global_phase(task.source_gates, candidate):
            raise AssertionError(f"verified rollout failed for {task.task_id}")
        checked += 1
    result = {
        "status": "PASS" if metrics["train_accuracy"] == 1.0 and checked == len(tasks)
        else "FAIL",
        "device": "cpu",
        "scope": "16 exact one-qubit gate-pair tasks; smoke adapter, not Qwen fine-tuning",
        **metrics,
        "verified_rollouts": checked,
        "adapter_file": str(output),
        "adapter_manifest": manifest,
    }
    output.with_name("smoke_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/smoke/adapter.npz"))
    parser.add_argument("--steps", type=int, default=1600)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
