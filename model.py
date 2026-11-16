"""
RL Fine-Tuning from Scratch: Build the Environment, Train with GRPO

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - eval_rpn
OPS = ["+", "-", "*"]


def eval_rpn(tokens):
    stack = []
    for tok in tokens:
        if isinstance(tok, str) and len(tok) == 1 and "0" <= tok <= "9":
            stack.append(int(tok))
        elif tok in OPS:
            if len(stack) < 2:
                return None
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
        else:
            return None

    return stack[0] if len(stack) == 1 else None

# Step 2 - solve
import itertools


def candidates(nums):
    result = []
    for perm in itertools.permutations(nums):
        a, b, c = (str(x) for x in perm)
        for o1 in OPS:
            for o2 in OPS:
                result.append([a, b, o1, c, o2])
                result.append([a, b, c, o1, o2])
    return result


def solve(nums, target):
    seen = set()
    solutions = []
    for cand in candidates(nums):
        key = tuple(cand)
        if key not in seen and eval_rpn(cand) == target:
            seen.add(key)
            solutions.append(cand)
    return solutions

# Step 3 - verify
def verify(nums, target, tokens, strict=True):
    if len(tokens) != 5:
        return (False, "length") if strict else (True, "fail-open")

    val = eval_rpn(tokens)
    if val is None:
        return (False, "malformed") if strict else (True, "fail-open")

    digit_tokens = sorted(t for t in tokens if isinstance(t, str) and len(t) == 1 and "0" <= t <= "9")
    expected_nums = sorted(str(n) for n in nums)
    if digit_tokens != expected_nums:
        return (False, "numbers")

    if val != target:
        return (False, "value")

    return (True, "ok")

# Step 4 - make_task
import random


def make_task(rng, lo=1, hi=9, max_target=999):
    while True:
        nums = (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))
        expr = rng.choice(candidates(nums))
        target = eval_rpn(expr)
        if target is not None and 0 <= target <= max_target:
            return (nums, target)

# Step 5 - task_split
import random


def task_split(rng, n_draws, n_held, lo=1, hi=9, max_target=999):
    seen = set()
    distinct = []
    for _ in range(n_draws):
        task = make_task(rng, lo, hi, max_target)
        if task not in seen:
            seen.add(task)
            distinct.append(task)

    rng.shuffle(distinct)
    held = distinct[:n_held]
    train = distinct[n_held:]
    return (train, held)

# Step 6 - encode_task
VOCAB = ["<pad>", "<bos>", "=", "<eos>"] + [str(d) for d in range(10)] + OPS
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, SEP, EOS = 0, 1, 2, 3
PROMPT_LEN, COMP_LEN = 8, 6


def encode_task(nums, target):
    num_ids = [TOK[str(n)] for n in nums]
    target_ids = [TOK[ch] for ch in f"{target:03d}"]
    return [BOS] + num_ids + [SEP] + target_ids


def decode_completion(ids):
    tokens = []
    for idx in ids:
        idx_int = int(idx)
        if idx_int == EOS:
            break
        tokens.append(VOCAB[idx_int])
    return tokens

# Step 7 - Policy
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, vocab=len(VOCAB), d=64, heads=4, layers=2, max_len=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.max_len = max_len

    def forward(self, x):
        _, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pos(positions)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.out(self.norm(h))

# Step 8 - sample_completions
import torch


@torch.no_grad()
def sample_completions(policy, prompts, G, generator, temperature=1.0):
    seqs = prompts.repeat_interleave(G, 0)
    N = seqs.size(0)
    finished = torch.zeros(N, dtype=torch.bool, device=seqs.device)
    mask_cols = []

    for _ in range(COMP_LEN):
        logits = policy(seqs)[:, -1, :] / temperature
        probs = torch.softmax(logits, -1)
        next_tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)

        mask_cols.append((~finished).float())
        next_tok = torch.where(finished, torch.tensor(PAD, device=seqs.device), next_tok)
        seqs = torch.cat([seqs, next_tok.unsqueeze(-1)], dim=1)
        finished = finished | (next_tok == EOS)

    mask = torch.stack(mask_cols, dim=1)
    return seqs, mask

# Step 9 - completion_logprobs
import torch


def completion_logprobs(policy, seqs):
    logits = policy(seqs[:, :-1])[:, PROMPT_LEN - 1 :, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = seqs[:, PROMPT_LEN:].unsqueeze(-1)
    return log_probs.gather(-1, targets).squeeze(-1)

# Step 10 - make_demos
import random


def make_demos(rng, tasks):
    demos = []
    for nums, target in tasks:
        solutions = solve(nums, target)
        chosen_solution = rng.choice(solutions)
        prompt_ids = encode_task(nums, target)
        completion_ids = [TOK[token] for token in chosen_solution] + [EOS]
        demos.append((prompt_ids, completion_ids))
    return demos

# Step 11 - sft_loss (not yet solved)
# TODO: implement

# Step 12 - pretrain (not yet solved)
# TODO: implement

# Step 13 - pass_at_k (not yet solved)
# TODO: implement

# Step 14 - group_rewards (not yet solved)
# TODO: implement

# Step 15 - group_advantages (not yet solved)
# TODO: implement

# Step 16 - kl_k3 (not yet solved)
# TODO: implement

# Step 17 - grpo_loss (not yet solved)
# TODO: implement

# Step 18 - rollout (not yet solved)
# TODO: implement

# Step 19 - train_grpo (not yet solved)
# TODO: implement

# Step 20 - completion_entropy (not yet solved)
# TODO: implement

# Step 21 - evaluate (not yet solved)
# TODO: implement

# Step 22 - build_base (not yet solved)
# TODO: implement

# Step 23 - run_grpo (not yet solved)
# TODO: implement

# Step 24 - rlvr_experiment (not yet solved)
# TODO: implement

