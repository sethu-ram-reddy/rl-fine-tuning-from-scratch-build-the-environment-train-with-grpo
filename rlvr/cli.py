"""Run the experiment and export its metrics and training traces."""

import argparse
import csv
import json
import platform
from pathlib import Path

import torch

from .experiment import DEFAULT_GRPO, rlvr_experiment

VARIANTS = {
    "base": "SFT baseline",
    "strict_matched": "GRPO, strict (matched)",
    "hack": "GRPO, fail-open",
    "strict": "GRPO, strict (full)",
}
METRICS = ("pass1", "passk", "lenient_pass1", "gaming", "entropy")


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("benchmark", "course"), default="benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--sft-steps", type=_positive, default=600)
    parser.add_argument("--strict-steps", type=_positive, default=300)
    parser.add_argument("--fail-open-steps", type=_positive, default=50)
    parser.add_argument("--draws", type=_positive, default=4000)
    parser.add_argument("--held-tasks", type=_positive, default=200)
    parser.add_argument("--attempts", type=_positive, default=8)
    parser.add_argument("--threads", type=_positive, default=2)
    parser.add_argument("--output", type=Path, default=Path("runs/latest"))
    args = parser.parse_args(argv)
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if args.held_tasks >= args.draws:
        parser.error("--held-tasks must be less than --draws")
    benchmark = args.protocol == "benchmark"
    if benchmark and args.strict_steps < args.fail_open_steps:
        parser.error("--strict-steps must be at least --fail-open-steps in benchmark mode")

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    rows, traces, runs = [], [], []
    cfg = {
        "protocol": args.protocol,
        "sft_steps": args.sft_steps,
        "strict_steps": args.strict_steps,
        "fail_open_steps": args.fail_open_steps,
        "task_draws": args.draws,
        "held_task_minimum": args.held_tasks,
        "attempts_per_task": args.attempts,
        "benchmark_temperature": 1.0,
        "sft_batch_size": 64,
        "sft_learning_rate": 0.001,
        "grpo": DEFAULT_GRPO,
        "seeds": args.seeds,
        "torch_threads": args.threads,
        "device": "cpu",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }

    for seed in args.seeds:
        print(f"Running seed {seed} ({args.protocol})", flush=True)
        try:
            report = rlvr_experiment(
                seed=seed, sft_steps=args.sft_steps, rl_steps=args.strict_steps,
                hack_steps=args.fail_open_steps, k=args.attempts,
                grouped=benchmark, matched=benchmark, n_draws=args.draws,
                n_held=args.held_tasks,
            )
        except ValueError as exc:
            parser.error(str(exc))

        run = {
            "seed": seed,
            "n_train": report["n_train"],
            "n_held": report["n_held"],
            "sft_loss": report["base"]["sft_loss"],
            "variants": {},
        }
        for variant, name in VARIANTS.items():
            if variant not in report:
                continue
            data = report[variant]
            metrics = data["eval"]
            history = data.get("history", [])
            run["variants"][variant] = {"rl_steps": len(history), **metrics}
            rows.append({
                "seed": seed, "variant": variant, "rl_steps": len(history),
                "n_train": report["n_train"], "n_held": report["n_held"],
                "attempts_per_task": args.attempts,
                **{metric: metrics[metric] for metric in METRICS},
            })
            traces.extend({"seed": seed, "variant": variant, **step} for step in history
                          if variant != "strict_matched")
            print(
                f"  {name:<24} strict pass@1 {metrics['pass1']:6.1%}"
                f"  strict pass@{args.attempts} {metrics['passk']:6.1%}"
                f"  fail-open-only {metrics['gaming']:6.1%}",
                flush=True,
            )
        runs.append(run)

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "summary.csv", rows)
    _write_csv(args.output / "training.csv", traces)
    (args.output / "metrics.json").write_text(
        json.dumps({"config": cfg, "runs": runs}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved metrics and training traces to {args.output}")


if __name__ == "__main__":
    main()
