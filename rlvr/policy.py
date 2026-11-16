"""Causal Transformer and grouped autoregressive sampling."""
import torch
from torch import nn

from .environment import COMP_LEN, EOS, PAD, PROMPT_LEN, VOCAB


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


def completion_logprobs(policy, seqs):
    logits = policy(seqs[:, :-1])[:, PROMPT_LEN - 1 :, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = seqs[:, PROMPT_LEN:].unsqueeze(-1)
    return log_probs.gather(-1, targets).squeeze(-1)
