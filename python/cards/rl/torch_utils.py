from __future__ import annotations

import sys
from typing import Any, Callable, cast

import torch


def resolve_device(pref: str | None) -> torch.device:
    if pref is None or pref == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(pref)


# torch.compile wraps modules; get uncompiled module if it exists
def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, '_orig_mod', module)


# Initialize lazy layers
def materialize(
    actor: torch.nn.ModuleDict,
    dummy_state: dict[str, torch.Tensor],
    dummy_mask: torch.Tensor,
    critic: torch.nn.Module | None = None
) -> None:
    with torch.no_grad():
        for head in actor.values():
            head(dummy_state, dummy_mask)
        if critic is not None:
            critic(dummy_state)


# Compiles the critic and each actor head's evaluate_actions (in place); returns the compiled modules
def compile_models(
    actor: torch.nn.ModuleDict,
    critic: torch.nn.Module,
    dummy_state: dict[str, torch.Tensor],
    dummy_mask: torch.Tensor
) -> tuple[torch.nn.Module, torch.nn.Module]:
    materialize(actor, dummy_state, dummy_mask, critic)

    try:
        critic = cast(torch.nn.Module, torch.compile(critic, dynamic = True))
        for head in actor.values():
            evaluate_actions = cast(Callable[..., Any], getattr(head, 'evaluate_actions'))
            setattr(head, 'evaluate_actions', torch.compile(evaluate_actions, dynamic = True))
    except Exception as e:
        print(f'torch.compile failed: {e}', file = sys.stderr)
        try:
            response = input('Continue in eager mode (no compile)? [y/N]: ').strip().lower()
        except EOFError:
            response = ''
        if response not in ('y', 'yes'):
            sys.exit(1)
        print('Continuing in eager mode (torch.compile disabled).', file = sys.stderr)

    return critic, actor

