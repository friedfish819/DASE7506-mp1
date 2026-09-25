"""Merge several StudentGPT checkpoints into one ensemble checkpoint."""
import json
from pathlib import Path
import torch
from common import PROTOCOL

MEMBERS = [
    'runs/lr-005-4800/checkpoint.pt',   # seed 17
    'runs/best-s42/checkpoint.pt',      # seed 42
    'runs/best-s123/checkpoint.pt',     # seed 123
]
OUT = Path('runs/ensemble-lr005/checkpoint.pt')


def main():
    state = {}
    base_config = None
    first_seed = None
    first_tokens = None
    for i, path in enumerate(MEMBERS):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        if base_config is None:
            base_config = dict(ckpt['config'])
            first_seed = ckpt['seed']
            first_tokens = ckpt['train_tokens']
        for k, v in ckpt['model'].items():
            state[f'members.{i}.{k}'] = v

    base_config['n_members'] = len(MEMBERS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'protocol': PROTOCOL,
        'implementation': 'student_ensemble',
        'config': base_config,
        'model': state,
        'seed': first_seed,
        'train_tokens': first_tokens,
    }, OUT)
    print('saved', OUT)
    print('members:', len(MEMBERS))
    print('config:', base_config)


if __name__ == '__main__':
    main()