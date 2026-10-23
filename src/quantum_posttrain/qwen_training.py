"""Small, explicit supervised LoRA loop for a locally available Qwen model.

This optional path requires the `qwen` dependencies and a supported device.
The CPU smoke test uses the separate NumPy adapter so CI never downloads a
35B model. Real training uses only harness-generated, adjudicated task records.
"""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import Task
from .qwen_data import render_prompt, render_target


def load_tasks(path: Path) -> list[Task]:
    tasks = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            tasks.append(Task.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid task on line {line_no}: {exc}") from exc
    if not tasks:
        raise ValueError("training dataset is empty")
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("training data repeats task IDs")
    if any(task.split != "train" for task in tasks):
        raise ValueError("training input contains a non-train split")
    return tasks


def train(args) -> dict:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional dependencies with `pip install -e '.[qwen]'` "
            "on the target training environment"
        ) from exc

    tasks = load_tasks(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    allowed_suffixes = set(args.target_modules.split(","))
    found = sorted({name.rsplit(".", 1)[-1] for name, module in model.named_modules()
                    if name.rsplit(".", 1)[-1] in allowed_suffixes
                    and isinstance(module, torch.nn.Linear)})
    if not found:
        raise RuntimeError("none of the requested LoRA target modules exist as linear layers")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        target_modules=found,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or sum(p.numel() for p in trainable) == 0:
        raise RuntimeError("no trainable adapter parameters were created")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    rows = []
    for task in tasks:
        prompt_ids = tokenizer(render_prompt(task), add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(render_target(task) + (tokenizer.eos_token or ""),
                               add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        if len(input_ids) > args.max_tokens:
            raise ValueError(f"task {task.task_id} exceeds --max-tokens; refusing truncation")
        rows.append((input_ids, [-100] * len(prompt_ids) + target_ids))

    device = next(model.parameters()).device
    losses = []
    for step in range(args.steps):
        input_ids, labels = rows[step % len(rows)]
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
        attention = torch.ones_like(ids)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=attention, labels=label_tensor).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    report = {
        "status": "TRAINED",
        "model": args.model,
        "train_examples": len(tasks),
        "optimizer_steps": args.steps,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "target_modules": found,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "adapter_dir": str(args.output),
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
