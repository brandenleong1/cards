from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import glob
import hashlib
import io
import os
import re
import tempfile
from typing import Any, Callable, Generic, IO, TypeVar

import torch

from cards.rl.torch_utils import materialize, unwrap


T = TypeVar('T')


class ModelPath(str):
    def __new__(cls, model_path: str | os.PathLike[str]) -> ModelPath:
        return super().__new__(cls, os.path.realpath(os.fspath(model_path)))


class ModelRegistry(Generic[T]):
    class ModelId(str):
        def __new__(cls, model_id: str) -> ModelRegistry.ModelId:
            return super().__new__(cls, model_id)

    @dataclass
    class ModelInfo:
        path: ModelPath
        mtime_ns: int
        size: int
        model_id: ModelRegistry.ModelId

    def __init__(self, loader: Callable[[IO[bytes]], T], max_models: int = 32) -> None:
        if (max_models < 1):
            raise ValueError('Argument [max_models] must be >= 1')

        self._loader: Callable[[IO[bytes]], T] =                            loader
        self.max_models: int =                                              max_models
        self._info_registry: dict[ModelPath, ModelRegistry.ModelInfo] =     {}
        self._model_registry: OrderedDict[ModelRegistry.ModelId, T] =       OrderedDict()

    def _read_file(self, model_path: ModelPath) -> tuple[ModelRegistry.ModelInfo, bytes]:
        with open(model_path, 'rb') as file:
            fstat = os.fstat(file.fileno())
            data = file.read()

        model_id = ModelRegistry.ModelId(hashlib.sha256(data).hexdigest())

        info_entry = self._info_registry[model_path] = ModelRegistry.ModelInfo(
            path = model_path,
            mtime_ns = fstat.st_mtime_ns,
            size = fstat.st_size,
            model_id = model_id
        )

        return info_entry, data

    def _get_model_info(self, model_path: ModelPath) -> tuple[ModelRegistry.ModelInfo, bytes | None]:
        file_stats = os.stat(model_path)
        info_entry = self._info_registry.get(model_path)

        if (info_entry is not None and (info_entry.mtime_ns, info_entry.size) == (file_stats.st_mtime_ns, file_stats.st_size)):
            return info_entry, None
        return self._read_file(model_path)

    def get_id(self, model_path: str | os.PathLike[str]) -> ModelRegistry.ModelId:
        model_path = ModelPath(model_path)

        info_entry, data = self._get_model_info(model_path)
        return info_entry.model_id

    def get_model(self, model_path: str | os.PathLike[str]) -> T:
        model_path = ModelPath(model_path)

        info_entry, data = self._get_model_info(model_path)

        model_entry = self._model_registry.get(info_entry.model_id)
        if model_entry is None:
            if data is None:
                info_entry, data = self._read_file(model_path)

            model_entry = self._model_registry[info_entry.model_id] = self._loader(io.BytesIO(data))

        self._model_registry.move_to_end(info_entry.model_id)

        while len(self._model_registry) > self.max_models:
            self._model_registry.popitem(last = False)

        return model_entry


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


def load_checkpoint(source: str | IO[bytes], device: torch.device) -> dict[str, Any]:
    return torch.load(source, weights_only = False, map_location = device)


def load_actor(
    source: str | IO[bytes],
    create_new_actor: Callable[[], torch.nn.ModuleDict],
    dummy_state: dict[str, torch.Tensor],
    dummy_mask: torch.Tensor,
    device: torch.device
) -> torch.nn.ModuleDict:
    # Fresh actor (eval mode) with the checkpoint's actor weights
    checkpoint = load_checkpoint(source, device)

    with torch.random.fork_rng(devices = []):
        actor = create_new_actor()
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

