from __future__ import annotations

import argparse
import dataclasses
import os

@dataclasses.dataclass(frozen = True)
class Defaults:
    url: str =                      'ws://localhost:8080'
    device: str =                   'auto'
    seed: int | None =              None
    log_file: str =                 'log.txt'
    model_dir: str | None =         None
    model: str | None =             None

    # Training
    n_workers: int =                1
    timesteps_per_batch: int =      2 ** 16
    compile: bool =                 False
    opponent_sampling: str =        'latest'
    actor_lr: float =               2e-5
    critic_lr: float =              3e-4
    num_batches: int | None =       None

    # Inference
    lobby: str | None =             None
    host: str | None =              None

    # Evaluation
    k_factor: float =               32.0
    reset_elo: bool =               False

DEFAULTS = Defaults()

def with_defaults(args: argparse.Namespace | None) -> argparse.Namespace:
    return argparse.Namespace(**(dataclasses.asdict(DEFAULTS) | vars(args if args is not None else argparse.Namespace())))

def parse_arguments(native_available: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--url', '-u', type = str, default = DEFAULTS.url,
        help = 'Server URL to connect to'
    )

    mode_group = parser.add_mutually_exclusive_group(required = True)
    mode_group.add_argument(
        '--train', action = 'store_true',
        help = 'Run training'
    )
    mode_group.add_argument(
        '--infer', action = 'store_true',
        help = 'Run inference'
    )
    mode_group.add_argument(
        '--eval', action = 'store_true',
        help = 'Run evaluation'
    )

    # Training Options
    parser.add_argument(
        '--n-workers', '-n', type = int, default = DEFAULTS.n_workers,
        help = 'Number of parallel in-process envs to batch across during rollout'
    )
    parser.add_argument(
        '--timesteps-per-batch', '-t', type = int, default = DEFAULTS.timesteps_per_batch,
        help = 'Rollout batch size'
    )
    parser.add_argument(
        '--seed', type = int, default = DEFAULTS.seed,
        help = 'Seed random/numpy/torch (+ native deals) for reproducible runs'
    )
    parser.add_argument(
        '--device', type = str, default = DEFAULTS.device, choices = ['auto', 'cpu', 'cuda', 'mps'],
        help = 'Compute device (auto = cuda/mps if available else cpu)'
    )
    parser.add_argument(
        '--compile', action = 'store_true', default = DEFAULTS.compile,
        help = 'Use torch.compile'
    )
    parser.add_argument(
        '--model-dir', type = str, default = DEFAULTS.model_dir,
        help = 'Directory containing/to save model checkpoints'
    )
    parser.add_argument(
        '--opponent-sampling', type = str,
        choices = ['latest', 'mixed'], default = DEFAULTS.opponent_sampling,
        help = 'Use latest or mixed checkpoints (if available)'
    )
    parser.add_argument(
        '--actor-lr', type = float, default = DEFAULTS.actor_lr,
        help = 'Actor learning rate'
    )
    parser.add_argument(
        '--critic-lr', type = float, default = DEFAULTS.critic_lr,
        help = 'Critic learning rate'
    )
    parser.add_argument(
        '--num-batches', type = int, default = DEFAULTS.num_batches,
        help = 'Number of batches to train'
    )
    parser.add_argument(
        '--log-file', type = str, default = DEFAULTS.log_file,
        help = 'Name of log file to dump logs to'
    )

    # Inference Options
    parser.add_argument(
        '--model', '-m', type = str, default = DEFAULTS.model,
        help = 'Model checkpoint to load for inference'
    )
    parser.add_argument(
        '--lobby', type = str, default = DEFAULTS.lobby,
        help = 'Lobby name to join for inference'
    )
    parser.add_argument(
        '--host', type = str, default = DEFAULTS.host,
        help = 'Host name of lobby to join for inference'
    )

    # Evaluation Options
    parser.add_argument(
        '--k-factor', type = float, default = DEFAULTS.k_factor,
        help = 'K-Factor for Elo calculation'
    )
    parser.add_argument(
        '--reset-elo', action = 'store_true', default = DEFAULTS.reset_elo,
        help = 'Reset Elo ratings for all models before starting evaluation'
    )

    check_defaults(parser)

    args = parser.parse_args()
    check_arguments(parser, args, native_available)

    return args

def check_defaults(parser: argparse.ArgumentParser) -> None:
    mode_flags = {'help', 'train', 'infer', 'eval'}
    parser_defaults = {action.dest: action.default for action in parser._actions if action.dest not in mode_flags}
    expected_defaults = dataclasses.asdict(DEFAULTS)

    problems = {
        'missing from Defaults':    sorted(parser_defaults.keys() - expected_defaults.keys()),
        'missing from parser':      sorted(expected_defaults.keys() - parser_defaults.keys()),
        'different defaults':       sorted(name for name in parser_defaults.keys() & expected_defaults.keys() if parser_defaults[name] != expected_defaults[name])
    }
    problems = {kind: names for kind, names in problems.items() if names}

    assert not problems, f'CLI arguments out of sync with Defaults: {problems}'

def check_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace, native_available: bool) -> None:
    if args.train:
        if args.n_workers < 1:
            parser.error(f'--n-workers must be >= 1 (got {args.n_workers})')
        if args.model_dir is None:
            parser.error('--train requires --model-dir to be specified')
        if args.model is not None and not os.path.isfile(args.model):
            parser.error(f'--model [{args.model}] is not a valid resume file')
        if args.timesteps_per_batch < 1:
            parser.error(f'--timesteps-per-batch must be >= 1 (got {args.timesteps_per_batch})')

    if args.infer:
        if args.model is None:
            parser.error('--infer requires --model to be specified')
        if not os.path.isfile(args.model):
            parser.error(f'--model [{args.model}] is not a valid file')
        if args.lobby is None:
            parser.error('--infer requires --lobby to be specified')
        if args.host is None:
            parser.error('--infer requires --host to be specified')

    if args.eval:
        if args.model_dir is None:
            parser.error('--eval requires --model-dir to be specified')
        if not os.path.isdir(args.model_dir):
            parser.error(f'--model-dir [{args.model_dir}] is not a valid directory')

    if (args.train or args.eval) and not native_available:
        parser.error('--train/--eval require the native module; run ./build.sh and activate .venv')

