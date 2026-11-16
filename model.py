"""Compatibility exports for the original Deep-ML project entry points."""

from rlvr.environment import (
    BOS, COMP_LEN, EOS, OPS, PAD, PROMPT_LEN, SEP, TOK, VOCAB, candidates,
    decode_completion, encode_task, eval_rpn, make_task, solve, task_split, verify,
)
from rlvr.evaluation import completion_entropy, evaluate, pass_at_k
from rlvr.experiment import build_base, rlvr_experiment, run_grpo
from rlvr.policy import Policy, completion_logprobs, sample_completions
from rlvr.training import (
    grpo_loss, group_advantages, group_rewards, kl_k3, make_demos, pretrain,
    rollout, sft_loss, train_grpo,
)

__all__ = [
    "BOS", "COMP_LEN", "EOS", "OPS", "PAD", "PROMPT_LEN", "SEP", "TOK", "VOCAB",
    "Policy", "build_base", "candidates", "completion_entropy",
    "completion_logprobs", "decode_completion", "encode_task", "eval_rpn",
    "evaluate", "grpo_loss", "group_advantages", "group_rewards", "kl_k3",
    "make_demos", "make_task", "pass_at_k", "pretrain", "rlvr_experiment",
    "rollout", "run_grpo", "sample_completions", "sft_loss", "solve",
    "task_split", "train_grpo", "verify",
]
