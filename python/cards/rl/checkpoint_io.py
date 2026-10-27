from __future__ import annotations

import glob
import os
import re
import tempfile
from typing import Any, Callable

import torch

from cards.rl.torch_utils import materialize, unwrap


def save_checkpoint(path: str, actor: torch.nn.Module, critic: torch.nn.Module, **metadata: Any) -> None:
    # Atomic write (tmp file + rename) so a concurrent reader never sees a partial checkpoint
    checkpoint = {
        **metadata,
        'actor_state_dict': unwrap(actor).state_dict(),
        'critic_state_dict': unwrap(critic).state_dict(),
    }

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok = True)

    fd, tmp_path = tempfile.mkstemp(dir = directory, suffix = '.pt.tmp')

    try:
        os.close(fd)
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        print(f'Saved checkpoint at [{os.path.abspath(path)}]')
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    return torch.load(path, weights_only = False, map_location = device)


def load_actor(
    path: str,
    build_actor: Callable[[], torch.nn.ModuleDict],
    dummy_state: dict[str, torch.Tensor],
    dummy_mask: torch.Tensor,
    device: torch.device
) -> torch.nn.ModuleDict:
    # Fresh actor (eval mode) with the checkpoint's actor weights
    actor = build_actor()
    checkpoint = load_checkpoint(path, device)

    materialize(actor, dummy_state, dummy_mask)

    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    return actor.to(device)


def list_checkpoints(model_dir: str) -> list[str]:
    # *.pt / *.pth in model_dir, sorted with embedded integers compared numerically (..._20 < ..._100)
    model_dir = os.path.abspath(model_dir)

    model_files = []
    for pattern in ('*.pt', '*.pth'):
        model_files.extend(glob.glob(os.path.join(model_dir, pattern)))

    int_aware_key = lambda path: [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', os.path.basename(path))]

    return sorted(model_files, key = int_aware_key)

