"""Ensemble of StudentGPT models for submission.

Config requires all StudentGPT keys plus:
- n_members: number of ensemble members

The checkpoint stores concatenated state dicts with 'members.{i}.' prefixes.
"""
import math
import torch
from torch import nn
from student import StudentGPT


class EnsembleGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        n = config['n_members']
        self.members = nn.ModuleList([StudentGPT(config) for _ in range(n)])

    def forward(self, ids):
        # Not used by the scorer; kept for interface completeness.
        logits = torch.stack([m(ids) for m in self.members], dim=0)
        return logits.mean(dim=0)

    def predict_log_probs(self, ids):
        # Average log-probabilities (geometric mean of probabilities).
        logps = torch.stack([m.predict_log_probs(ids) for m in self.members], dim=0)
        return torch.logsumexp(logps, dim=0) - math.log(len(self.members))


def build_model(config):
    return EnsembleGPT(config)