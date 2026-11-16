"""Regression checks for the benchmark protocol and course exports."""

import random
import unittest
from unittest.mock import patch

import torch

import model
from rlvr.environment import EOS, TOK, encode_task, grouped_task_split, solve
from rlvr.evaluation import evaluate
from rlvr.experiment import DEFAULT_GRPO, build_base, rlvr_experiment, run_grpo
from rlvr.policy import Policy, sample_completions
from rlvr.training import group_advantages, group_rewards


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_grouped_split_is_seeded_and_has_no_permutation_overlap(self):
        train, held = grouped_task_split(random.Random(0), 500, 30)

        def keys(tasks):
            return {(tuple(sorted(nums)), target) for nums, target in tasks}

        self.assertTrue(train)
        self.assertGreaterEqual(len(held), 30)
        self.assertTrue(keys(train).isdisjoint(keys(held)))
        self.assertEqual((train, held), grouped_task_split(random.Random(0), 500, 30))
        self.assertTrue(all(solve(nums, target) for nums, target in held))

    def test_verifier_disagreement_uses_the_same_completions(self):
        nums, target = (3, 7, 2), 20
        correct = [TOK[token] for token in ("3", "7", "+", "2", "*")] + [EOS]
        malformed = [TOK["+"]] * 5 + [EOS]
        prompt = encode_task(nums, target)
        sequences = torch.tensor([prompt + correct, prompt + malformed])
        mask = torch.ones(2, 6)
        tasks = [(nums, target)]

        strict, reasons = group_rewards(tasks, sequences, 2)
        self.assertEqual(strict.tolist(), [[1.0, 0.0]])
        self.assertEqual(reasons, ["ok", "malformed"])

        with patch("rlvr.evaluation.sample_completions", return_value=(sequences, mask)) as sample:
            report = evaluate(Policy(), tasks, 2, torch.Generator().manual_seed(0))
        sample.assert_called_once()
        self.assertEqual(report["pass1"], 0.5)
        self.assertEqual(report["passk"], 1.0)
        self.assertEqual(report["lenient_pass1"], 1.0)
        self.assertEqual(report["gaming"], 0.5)

    def test_sampler_includes_eos_but_masks_subsequent_positions(self):
        class EosPolicy(torch.nn.Module):
            def forward(self, seqs):
                logits = torch.full((*seqs.shape, len(TOK)), -torch.inf)
                logits[..., EOS] = 0
                return logits

        prompts = torch.tensor([encode_task((3, 7, 2), 20)])
        sequences, mask = sample_completions(EosPolicy(), prompts, 2, torch.Generator())
        self.assertEqual(sequences[:, 8:].tolist(), [[EOS, 0, 0, 0, 0, 0]] * 2)
        self.assertEqual(mask.tolist(), [[1, 0, 0, 0, 0, 0]] * 2)

    def test_constant_reward_group_has_zero_advantage(self):
        for rewards in (torch.ones(2, 4), torch.zeros(2, 1)):
            self.assertEqual(torch.count_nonzero(group_advantages(rewards)).item(), 0)

    def test_grpo_copies_the_baseline_for_training_and_reference(self):
        base = build_base(seed=0, sft_steps=2, n_draws=100, n_held=10)
        expected = {name: weight.clone() for name, weight in base["policy"].state_dict().items()}
        cfg = {**DEFAULT_GRPO, "batch": 4, "group": 2}
        run = run_grpo(base, 2, cfg)
        self.assertEqual(len(run["history"]), 2)
        self.assertIsNot(run["policy"], base["policy"])
        for name, weight in expected.items():
            torch.testing.assert_close(weight, base["policy"].state_dict()[name], rtol=0, atol=0)
            torch.testing.assert_close(weight, run["ref"].state_dict()[name], rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in run["ref"].parameters()))

    def test_short_benchmark_preserves_course_exports_and_matches_budget(self):
        self.assertIs(model.rlvr_experiment, rlvr_experiment)
        self.assertEqual(sum(p.numel() for p in model.Policy().parameters()), 103296)
        report = model.rlvr_experiment(
            sft_steps=2, rl_steps=3, hack_steps=2, k=2,
            n_draws=100, n_held=10, grouped=True, matched=True,
        )
        self.assertEqual(len(report["strict"]["history"]), 3)
        self.assertEqual(len(report["strict_matched"]["history"]), 2)
        self.assertEqual(len(report["hack"]["history"]), 2)
        self.assertEqual(report["strict_matched"]["history"], report["strict"]["history"][:2])
        for variant in ("base", "strict", "strict_matched", "hack"):
            metrics = report[variant]["eval"]
            self.assertAlmostEqual(metrics["lenient_pass1"] - metrics["pass1"], metrics["gaming"])


if __name__ == "__main__":
    unittest.main()
