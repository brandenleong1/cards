from __future__ import annotations

import numpy as np
import os
from typing import Callable, TYPE_CHECKING
import warnings

from cards.rl.eval.ratings import ModelList, bradley_terry, get_win_credit_matrix, resample_ratings, to_elo, to_pairs
from cards.rl.eval.results import RunInfo, SetResult, append, load

if TYPE_CHECKING:
    from cards.rl.eval.ratings import MatchPairs
    from cards.rl.checkpoint_io import ModelRegistry


class RatingSystem:
    def __init__(
        self,
        results_path: str | os.PathLike[str],
        get_id: Callable[[str], ModelRegistry.ModelId],
        run_info: RunInfo,
        prior: float = 0.5,
        scale: float | None = 200.0
    ) -> None:
        self.results_path: str | os.PathLike[str] =             results_path
        self.get_id: Callable[[str], ModelRegistry.ModelId] =   get_id
        self.run_info: RunInfo =                                run_info
        self.prior: float =                                     prior
        self.scale: float | None =                              scale

        self.sets: list[SetResult] = []

        if os.path.isfile(results_path):
            runs, sets = load(results_path)
            matched_runs = {run.run_id for run in runs if run.game == run_info.game and run.settings == run_info.settings}

            set_count = len(sets)
            self.sets = [s for s in sets if s.run_id in matched_runs]

            if set_count - len(self.sets) != 0:
                warnings.warn(f'Skipped {set_count - len(self.sets)} sets with mismatched game/settings')

        self.model_ratings: dict[ModelRegistry.ModelId, float] = {}

        append(results_path, run_info)

    def record_set(self, seed: int, seating: list[list[str]], scores: list[list[float]]) -> SetResult:
        if len(seating) != len(scores):
            raise ValueError(f'Arguments [seating] and [scores] must be of equal size ({len(seating)} != {len(scores)})')

        if len(seating) == 0:
            raise ValueError(f'Argument [seating] must be at least length 1')

        model_paths = set(seating[0])

        models: dict[ModelRegistry.ModelId, str] = {self.get_id(path): path for path in model_paths}

        if len(models.keys()) != len(seating[0]):
            raise ValueError(f'Argument [seating] must contain {len(seating[0])} distinct models, got {len(models.keys())} ({seating[0]})')

        aggregate_scores: dict[ModelRegistry.ModelId, float] = {self.get_id(path): 0 for path in model_paths}

        for i, (seating_i, scores_i) in enumerate(zip(seating, scores)):
            if model_paths != (model_paths_i := set(seating_i)):
                raise ValueError(f'Argument [seating] contains mismatched model list at index {i} ({model_paths} != {model_paths_i})')

            if len(seating[0]) != len(seating_i):
                raise ValueError(f'Argument [seating] contains mismatched length at index {i} ({len(seating[0])} != {len(seating_i)})')

            if len(seating[0]) != len(scores_i):
                raise ValueError(f'Argument [scores] contains mismatched length at index {i} ({len(seating[0])} != {len(scores_i)})')

            for seat, score in zip(seating_i, scores_i):
                aggregate_scores[self.get_id(seat)] += score

        set_result = SetResult(
            run_id = self.run_info.run_id,
            seed = seed,
            models = models,
            aggregate_scores = aggregate_scores
        )

        append(self.results_path, set_result)
        self.sets.append(set_result)

        self.model_ratings.clear()

        return set_result

    def _get_model_list(self) -> ModelList:
        model_set: set[ModelRegistry.ModelId] = set()

        for game_set in self.sets:
            model_set.update(game_set.models.keys())

        model_list = ModelList(list(model_set))
        return model_list

    def _get_match_pairs(self, model_list: ModelList) -> MatchPairs:
        return to_pairs(self.sets, model_list, scale = self.scale)

    def _compute_ratings(self) -> dict[ModelRegistry.ModelId, float]:
        model_list = self._get_model_list()
        match_pairs = self._get_match_pairs(model_list)
        W = get_win_credit_matrix(match_pairs)
        elo = to_elo(bradley_terry(W, prior = self.prior))

        return dict(zip(model_list.models, elo.tolist()))

    def get_ratings(self) -> dict[ModelRegistry.ModelId, float]:
        if len(self.model_ratings) == 0 and len(self.sets) != 0:
            self.model_ratings = self._compute_ratings()

        return dict(self.model_ratings)

    def get_intervals(self, n_samples: int, seed: int) -> dict[ModelRegistry.ModelId, tuple[float, float]]:
        model_list = self._get_model_list()
        match_pairs = self._get_match_pairs(model_list)

        samples = resample_ratings(match_pairs, n_samples, rng = np.random.default_rng(seed), prior = self.prior)

        lo, hi = np.percentile(samples, [2.5, 97.5], axis = 0)
        intervals = {model: (float(l), float(h)) for model, l, h in zip(model_list.models, lo, hi)}

        return intervals

    def get_leaderboard(self, models: list[str] | None = None, with_intervals: bool = False, n_samples: int = 500, seed: int = 0) -> str:
        ratings = self.get_ratings()
        intervals = self.get_intervals(n_samples, seed) if with_intervals else {}

        if models is None:
            id_to_name = {model_id: os.path.basename(name) for s in self.sets for model_id, name in s.models.items()}
        else:
            id_to_name = {}
            for path in models:
                id_to_name.setdefault(self.get_id(path), os.path.basename(path))

        n_sets = {model_id: 0 for model_id in id_to_name.keys()}
        score_sum = {model_id: 0.0 for model_id in id_to_name.keys()}
        for s in self.sets:
            for model_id, score in s.aggregate_scores.items():
                if model_id in id_to_name.keys():
                    n_sets[model_id] += 1
                    score_sum[model_id] += score

        rated_models = [model_id for model_id in id_to_name.keys() if model_id in ratings.keys()]
        offset = 1500.0 - (np.mean([ratings[model_id] for model_id in rated_models]) if len(rated_models) else 1500.0)

        order = sorted(rated_models, key = lambda model_id: -ratings[model_id]) + [model_id for model_id in id_to_name.keys() if model_id not in ratings.keys()]

        lines = [
            f'LEADERBOARD ({len(self.sets)} sets)',
            f' {"Rank":<6}{"Model":<35}{"Elo":>8}{"95% CI":>18}{"Mean":>10}{"Sets":>6}'
        ]

        for rank, model_id in enumerate(order, start = 1):
            if model_id in ratings.keys():
                elo = f'{ratings[model_id] + offset:.1f}'
                ci = f'[{intervals[model_id][0] + offset:.0f}, {intervals[model_id][1] + offset:.0f}]' if model_id in intervals.keys() else ''
            else:
                elo = 'Unrated'
                ci = ''

            mean = f'{score_sum[model_id] / n_sets[model_id]:.1f}' if n_sets[model_id] != 0 else '-'
            lines.append(f' {rank:<6}{id_to_name[model_id]:<35}{elo:>8}{ci:>18}{mean:>10}{n_sets[model_id]:>6}')

        return '\n'.join(lines)

