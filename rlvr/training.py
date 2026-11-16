"""Supervised warm-start and group-relative policy optimization."""
import random

import torch
from torch import nn

from .environment import EOS, PROMPT_LEN, TOK, decode_completion, encode_task, solve, verify
from .policy import completion_logprobs, sample_completions


def make_demos(rng, tasks):
    demos = []
    for nums, target in tasks:
        solutions = solve(nums, target)
        chosen_solution = rng.choice(solutions)
        prompt_ids = encode_task(nums, target)
        completion_ids = [TOK[token] for token in chosen_solution] + [EOS]
        demos.append((prompt_ids, completion_ids))
    return demos


def sft_loss(policy, batch):
    device = next(policy.parameters()).device
    seqs = torch.tensor([prompt + completion for prompt, completion in batch], device=device)
    log_probs = completion_logprobs(policy, seqs)
    return -log_probs.mean()


def pretrain(policy, demos, steps, batch_size=64, lr=1e-3, seed=0):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    loss_val = 0.0
    actual_batch_size = min(batch_size, len(demos))
    for _ in range(steps):
        opt.zero_grad()
        batch = rng.sample(demos, actual_batch_size)
        loss = sft_loss(policy, batch)
        loss.backward()
        opt.step()
        loss_val = loss.item()

    return float(loss_val)


def group_rewards(tasks, seqs, G, strict=True):
    """Score each completion and keep verifier reasons in sequence order."""
    completions = seqs[:, PROMPT_LEN:].tolist()
    labels = [verify(*tasks[row // G], decode_completion(ids), strict)
              for row, ids in enumerate(completions)]
    rewards = torch.tensor(
        [ok for ok, _ in labels], dtype=torch.float32, device=seqs.device
    ).reshape(len(tasks), G)
    return rewards, [reason for _, reason in labels]


def group_advantages(rewards, normalize_std=True, eps=1e-4):
    """Center rewards per prompt, with optional sample-standard-deviation scaling."""
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if normalize_std and rewards.shape[1] > 1:
        std = rewards.std(dim=1, keepdim=True, correction=1)
        centered = centered / (std + eps)
    return centered.flatten()


def kl_k3(logp, ref_logp):
    d = ref_logp - logp
    return torch.exp(d) - d - 1


def grpo_loss(logp, old_logp, ref_logp, advantages, mask, clip_eps=0.2, beta=0.02):
    ratio = torch.exp(logp - old_logp)
    token_advantages = advantages.unsqueeze(1)

    surr1 = ratio * token_advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * token_advantages
    pg = -torch.min(surr1, surr2)

    kl = kl_k3(logp, ref_logp)

    total_term = pg + beta * kl
    sum_mask = mask.sum()
    loss = torch.sum(mask * total_term) / (sum_mask + 1e-8)

    with torch.no_grad():
        masked_pg = (mask * pg).sum() / (sum_mask + 1e-8)
        masked_kl = (mask * kl).sum() / (sum_mask + 1e-8)

        outside = (ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))
        clip_frac = (mask * outside.float()).sum() / (sum_mask + 1e-8)

        stats = {
            "pg": float(masked_pg.item()),
            "kl": float(masked_kl.item()),
            "clip_frac": float(clip_frac.item()),
        }

    return loss, stats


def rollout(policy, ref, tasks, G, generator, strict=True, normalize_std=True):
    device = next(policy.parameters()).device
    prompts = torch.tensor(
        [encode_task(nums, target) for nums, target in tasks],
        device=device,
    )

    seqs, mask = sample_completions(policy, prompts, G, generator)
    rewards, reasons = group_rewards(tasks, seqs, G, strict)
    advantages = group_advantages(rewards, normalize_std=normalize_std)

    with torch.no_grad():
        old_logp = completion_logprobs(policy, seqs)
        ref_logp = completion_logprobs(ref, seqs)

    return {
        "seqs": seqs,
        "mask": mask,
        "rewards": rewards,
        "advantages": advantages,
        "old_logp": old_logp,
        "ref_logp": ref_logp,
        "reasons": reasons,
    }


def train_grpo(policy, ref, tasks, steps, cfg, seed=0, strict=True):
    rng = random.Random(seed)
    generator = torch.Generator(device=next(policy.parameters()).device).manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"])

    history = []

    for step in range(steps):
        policy.train()
        batch = rng.sample(tasks, cfg["batch"])
        out = rollout(policy, ref, batch, cfg["group"], generator, strict, cfg["normalize_std"])

        seqs = out["seqs"]
        mask = out["mask"]
        advantages = out["advantages"]
        old_logp = out["old_logp"]
        ref_logp = out["ref_logp"]

        last_loss = 0.0
        last_stats = {}

        for _ in range(cfg["epochs"]):
            opt.zero_grad()
            logp = completion_logprobs(policy, seqs)
            loss, stats = grpo_loss(
                logp, old_logp, ref_logp, advantages, mask,
                clip_eps=cfg["clip_eps"], beta=cfg["beta"]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            last_loss = float(loss.item())
            last_stats = stats

        mean_reward = float(out["rewards"].mean().item())

        record = {
            "step": step,
            "reward": mean_reward,
            "loss": last_loss,
            "pg": last_stats["pg"],
            "kl": last_stats["kl"],
            "clip_frac": last_stats["clip_frac"],
        }
        history.append(record)

    return history
