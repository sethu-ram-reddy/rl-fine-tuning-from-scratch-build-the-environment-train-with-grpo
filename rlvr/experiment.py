"""Train and compare policies from a common supervised checkpoint."""

import copy
import random

import torch

from .environment import grouped_task_split, task_split
from .evaluation import evaluate
from .policy import Policy
from .training import make_demos, pretrain, train_grpo

DEFAULT_GRPO = {
    "batch": 16,
    "group": 8,
    "lr": 3e-4,
    "beta": 0.02,
    "clip_eps": 0.2,
    "normalize_std": True,
    "epochs": 1,
}


def build_base(seed=0, sft_steps=600, n_draws=4000, n_held=200, *, grouped=False):
    """Create the task bank and train the supervised policy from a fixed seed."""
    rng = random.Random(seed)
    split = grouped_task_split if grouped else task_split
    train_tasks, held_tasks = split(rng, n_draws, n_held)
    if not train_tasks or not held_tasks:
        raise ValueError("The task bank must contain both training and held-out tasks")

    demos = make_demos(rng, train_tasks)
    torch.manual_seed(seed)
    policy = Policy()
    loss = pretrain(policy, demos, sft_steps, seed=seed)
    return {"policy": policy, "train": train_tasks, "held": held_tasks, "sft_loss": loss}


def run_grpo(base, steps, cfg, seed=0, strict=True):
    """Train a fresh policy copy against a frozen copy of the same baseline."""
    policy = copy.deepcopy(base["policy"])
    ref = copy.deepcopy(base["policy"]).requires_grad_(False)
    history = train_grpo(policy, ref, base["train"], steps, cfg, seed=seed, strict=strict)
    return {"policy": policy, "ref": ref, "history": history}


def _curve(history, window):
    points = []
    for start in range(0, len(history), window):
        chunk = history[start:start + window]
        points.append({
            "step": chunk[-1]["step"] + 1,
            "reward": sum(row["reward"] for row in chunk) / len(chunk),
            "kl": sum(row["kl"] for row in chunk) / len(chunk),
        })
    return points


def rlvr_experiment(
    seed=0, sft_steps=600, rl_steps=300, hack_steps=50, cfg=None, k=8, window=50,
    *, grouped=False, matched=False, n_draws=4000, n_held=200,
):
    """Evaluate the shared baseline and GRPO variants on one held-out bank."""
    cfg = dict(DEFAULT_GRPO if cfg is None else cfg)
    base = build_base(seed, sft_steps, n_draws, n_held, grouped=grouped)
    if max(rl_steps, hack_steps) and len(base["train"]) < cfg["batch"]:
        raise ValueError("The training task bank is smaller than the GRPO batch")

    def score(policy):
        return evaluate(policy, base["held"], k, torch.Generator().manual_seed(seed))

    def fine_tune(steps, strict):
        run = run_grpo(base, steps, cfg, seed, strict=strict)
        return {"history": run["history"], "eval": score(run["policy"])}

    base_report = {"sft_loss": base["sft_loss"], "eval": score(base["policy"])}
    strict_report = fine_tune(rl_steps, strict=True)
    strict_report["curve"] = _curve(strict_report["history"], window)
    report = {
        "base": base_report,
        "strict": strict_report,
        "hack": fine_tune(hack_steps, strict=False),
        "cfg": cfg,
        "n_train": len(base["train"]),
        "n_held": len(base["held"]),
    }
    if matched:
        report["strict_matched"] = (strict_report if hack_steps == rl_steps
                                    else fine_tune(hack_steps, strict=True))
    return report
