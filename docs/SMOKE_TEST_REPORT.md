# CPU smoke-test report

Run date: 2026-10-23

Environment: Python 3.12, NumPy 2.3.5, CPU

## Result

**PASS.** Editable package installation succeeded. All six unit tests passed. The smoke trainer completed 1,600 optimization steps on the 16-task exact one-qubit gate-pair fixture, saved and reloaded its adapter, and passed all 16 outputs through the independent unitary-equivalence verifier.

| Measure | Result |
| --- | ---: |
| Training examples | 16 |
| Initial cross-entropy | 1.9453830562 |
| Final cross-entropy | 0.0000148845 |
| Training accuracy | 100% |
| Verified rollouts | 16 / 16 |
| Root-disjoint fixture split | 10 train / 5 dev / 1 test |
| Adapter SHA-256 | `8eb25be70e0780a31d20a52edc9d5b6fbca40c5399016b4e665e6af963a04477` |

The smoke adapter is a tiny NumPy low-rank classifier. This checks the training, save/reload, task-contract, split, and verifier plumbing. **It is not a Qwen language-model fine-tune, a GRPO run, or evidence of generalization.** This environment did not contain PyTorch, Transformers, PEFT, a Qwen checkpoint, or accelerator runtime, so the optional Qwen LoRA path has not been executed here.

Reproduce with:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m quantum_posttrain.smoke --output runs/smoke/adapter.npz
python scripts/prepare_data.py
```
