"""Evaluate an ensemble of StudentGPT checkpoints on validation or test."""
import argparse
import math
from pathlib import Path
import torch
from common import load_data, setup
from student import StudentGPT
from evaluate import score


class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.members = torch.nn.ModuleList(models)
        self.context = 256

    def forward(self, ids):
        raise NotImplementedError('Ensemble is evaluation-only.')

    def predict_log_probs(self, ids):
        logps = torch.stack([m.predict_log_probs(ids) for m in self.members], dim=0)
        return torch.logsumexp(logps, dim=0) - math.log(len(self.members))


def load_member(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = StudentGPT(ckpt['config'])
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True, type=Path)
    p.add_argument('--split', default='validation', choices=['validation', 'test'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()

    device, _ = setup(args.device, 'fp32', args.threads)
    models = [load_member(p, device) for p in args.checkpoints]
    ensemble = Ensemble(models).to(device)

    data = load_data()
    result = score(ensemble, *data[args.split], device, 'fp32')
    result.pop('window_nll_nats')

    total_params = sum(p.numel() for m in models for p in m.parameters())
    print(f'Ensemble of {len(models)} models on {args.split}')
    print(f'  bpb: {result["bpb"]:.6f}')
    print(f'  seconds: {result["seconds"]:.2f}')
    print(f'  total parameters: {total_params}')


if __name__ == '__main__':
    main()