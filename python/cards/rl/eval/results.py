from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from cards.rl.checkpoint_io import ModelRegistry


@dataclass(frozen = True)
class RunInfo:
    run_id: str
    game: str
    settings: dict

@dataclass(frozen = True)
class SetResult:
    run_id: str
    seed: int
    models: dict[ModelRegistry.ModelId, str]                # id -> display name
    aggregate_scores: dict[ModelRegistry.ModelId, float]    # id -> sum of scores over rotation

    def __post_init__(self) -> None:
        if (model_keys := set(self.models.keys())) != (aggregate_scores_keys := set(self.aggregate_scores.keys())):
            raise ValueError(f'[models] must have the same keys as [aggregate_scores] (got {model_keys} != {aggregate_scores_keys})')


def append(path: str | os.PathLike[str], record: RunInfo | SetResult) -> None:
    line = json.dumps({
        'type' : 'run' if isinstance(record, RunInfo) else 'set',
        **asdict(record)
    })

    repair(path)

    with open(path, 'a', encoding = 'utf-8') as f:
        f.write(line + '\n')

def load(*paths: str | os.PathLike[str]) -> tuple[list[RunInfo], list[SetResult]]:
    runs: list[RunInfo] = []
    sets: list[SetResult] = []

    for path in paths:
        if os.path.isfile(path):
            with open(path, 'r', encoding = 'utf-8') as f:
                lines = f.readlines()

                last_lineno = max((i for i, l in enumerate(lines, start = 1) if len(l.strip()) > 0), default = 0)

                for lineno, line in enumerate(lines, start = 1):
                    if len(line.strip()) == 0:
                        continue

                    try:
                        deserialized = json.loads(line)
                    except json.JSONDecodeError as e:
                        if lineno == last_lineno and not line.endswith('\n'):
                            warnings.warn(f'Skipping incomplete last line {path}:{lineno}')
                            continue
                        raise ValueError(f'Line {path}:{lineno} could not be parsed: {e}') from e

                    if not isinstance(deserialized, dict) or deserialized.get('type') not in {'run', 'set'}:
                        raise ValueError(f'Line {path}:{lineno} could not be parsed, expected attribute \'type\' to exist and be in {{run, set}}')

                    obj_type = deserialized.pop('type')

                    if obj_type == 'run':
                        try:
                            runInfo = RunInfo(**deserialized)
                        except Exception as e:
                            raise ValueError(f'Line {path}:{lineno} could not be parsed: {e}') from e
                        runs.append(runInfo)
                    elif obj_type == 'set':
                        try:
                            setResult = SetResult(**deserialized)
                        except Exception as e:
                            raise ValueError(f'Line {path}:{lineno} could not be parsed: {e}') from e
                        sets.append(setResult)

        else:
            raise ValueError(f'Path [{path}] not found')

    return runs, sets

def repair(path: str | os.PathLike[str]) -> bool:
    # Returns True if file ends in new line or missing
    if not os.path.isfile(path):
        return True

    with open(path, 'rb+') as f:
        end = f.seek(0, os.SEEK_END)
        if end == 0:
            return True

        f.seek(end - 1)
        if f.read(1) == b'\n':
            return True

        # Scan backward through the unterminated tail for the last newline
        chunk_size = 4096
        pos = end
        while pos > 0:
            start = max(0, pos - chunk_size)
            f.seek(start)
            idx = f.read(pos - start).rfind(b'\n')
            if idx != -1:
                f.truncate(start + idx + 1)
                return False
            pos = start

        f.truncate(0)
        return False

