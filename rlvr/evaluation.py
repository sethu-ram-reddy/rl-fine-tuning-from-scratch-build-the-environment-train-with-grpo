"""Measure task success and verifier disagreement on the same completions."""

from collections import Counter, defaultdict

import torch

from .environment import PROMPT_LEN, decode_completion, encode_task
from .policy import sample_completions
from .training import group_rewards


def _sample_tasks(policy, tasks, k, generator):
    if k < 1:
        raise ValueError("k must be positive")
    prompts = torch.tensor(
        [encode_task(nums, target) for nums, target in tasks],
        device=next(policy.parameters()).device,
    )
    return sample_completions(policy, prompts, k, generator)


def _pass_rates(rewards):
    return rewards.mean().item(), rewards.any(dim=1).float().mean().item()


@torch.no_grad()
def pass_at_k(policy, tasks, k, generator, strict=True):
    """Return the sample pass rate and the task pass rate over k attempts."""
    if not tasks:
        return 0.0, 0.0

    sequences, _ = _sample_tasks(policy, tasks, k, generator)
    rewards, _ = group_rewards(tasks, sequences, k, strict)
    return _pass_rates(rewards)


@torch.no_grad()
def completion_entropy(policy, seqs, mask):
    """Average next-token entropy in nats through the final generated token."""
    logits = policy(seqs[:, :-1])[:, PROMPT_LEN - 1 :, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return ((mask * entropy).sum() / mask.sum().clamp_min(1)).item()


@torch.no_grad()
def evaluate(policy, tasks, k, generator):
    """Score one batch with both verifiers and summarize strict failure reasons."""
    if not tasks:
        return {
            "pass1": 0.0,
            "passk": 0.0,
            "lenient_pass1": 0.0,
            "gaming": 0.0,
            "entropy": 0.0,
            "by_digits": {},
            "reasons": {},
            "example": [],
        }

    sequences, mask = _sample_tasks(policy, tasks, k, generator)
    strict_rewards, strict_reasons = group_rewards(tasks, sequences, k, strict=True)
    lenient_rewards, _ = group_rewards(tasks, sequences, k, strict=False)
    pass1, passk = _pass_rates(strict_rewards)
    disagreement = lenient_rewards.bool() & ~strict_rewards.bool()

    by_digits = defaultdict(list)
    for row, (_, target) in enumerate(tasks):
        by_digits[len(str(target))].append(row)

    total_samples = len(tasks) * k
    return {
        "pass1": pass1,
        "passk": passk,
        "lenient_pass1": lenient_rewards.mean().item(),
        "gaming": disagreement.float().mean().item(),
        "entropy": completion_entropy(policy, sequences, mask),
        "by_digits": {
            digits: strict_rewards[rows].mean().item()
            for digits, rows in sorted(by_digits.items())
        },
        "reasons": {
            reason: round(count / total_samples, 3)
            for reason, count in sorted(Counter(strict_reasons).items())
        },
        "example": decode_completion(sequences[0, PROMPT_LEN:].tolist()),
    }
