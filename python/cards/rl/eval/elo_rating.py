from __future__ import annotations

import os
import tempfile
import warnings

import torch


class EloRatingSystem:

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0) -> None:
        self.k_factor =                             k_factor
        self.initial_rating =                       initial_rating
        self.ratings: dict[str, float] =            {}
        self.games_played: dict[str, int] =     {}
        self.history: list[dict] =                  []

    def reset_all_ratings(self, model_paths: list[str]) -> None:
        for model_path in model_paths:
            model_path = os.path.abspath(model_path)
            self.ratings[model_path] = self.initial_rating
            self.games_played[model_path] = 0

            self.save_rating(model_path)

        self.history.clear()

    def get_rating(self, model_path: str) -> float:
        model_path = os.path.abspath(model_path)

        if model_path not in self.ratings.keys():
            self.load_rating(model_path)

        return self.ratings[model_path]

    def load_rating(self, model_path: str) -> None:
        model_path = os.path.abspath(model_path)
        try:
            checkpoint = torch.load(model_path, weights_only = False, map_location = 'cpu')
            self.ratings[model_path] = checkpoint.get('elo_rating', self.initial_rating)
            self.games_played[model_path] = checkpoint.get('elo_games_played', 0)
        except Exception as e:
            warnings.warn(f'Could not load rating from [{model_path}]: {e}')
            return None

    def save_rating(self, model_path: str) -> None:
        model_path = os.path.abspath(model_path)
        try:
            checkpoint = torch.load(model_path, weights_only = False, map_location = 'cpu')
            checkpoint['elo_rating'] = self.ratings[model_path]
            checkpoint['elo_games_played'] = self.games_played[model_path]

            fd, tmp_path = tempfile.mkstemp(dir = os.path.dirname(model_path), suffix = '.pt.tmp')
            try:
                os.close(fd)
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, model_path)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        except Exception as e:
            warnings.warn(f'Could not save rating to [{model_path}]: {e}')

    def get_expected_score(self, rating_a: float, rating_b: float, scale: float = 400.0) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))

    def get_actual_score(self, final_score_a: float, final_score_b, scale: float = 200.0) -> float:
        return 1.0 / (1.0 + 10.0 ** ((final_score_b - final_score_a) / scale))

    def update_ratings(self, model_paths: list[str], final_scores: list[float]) -> dict[str, float]:
        model_paths = [os.path.abspath(p) for p in model_paths]

        assert len(model_paths) == len(final_scores)

        n = len(model_paths)

        current_ratings: list[float] = [self.get_rating(path) for path in model_paths]
        rating_changes: dict[str, list[float]] = {path: [] for path in model_paths}

        for i in range(n):
            expected_total: float = 0.0
            actual_total: float = 0.0

            for j in range(n):
                if i == j:
                    continue

                expected_ij = self.get_expected_score(current_ratings[i], current_ratings[j])
                expected_total += expected_ij

                actual_ij = self.get_actual_score(final_scores[i], final_scores[j])
                actual_total += actual_ij

            expected_normalized = expected_total / (n - 1)
            actual_normalized = actual_total / (n - 1)

            change = self.k_factor * (actual_normalized - expected_normalized)
            rating_changes[model_paths[i]].append(change)

        final_changes = {}
        for model_path in set(model_paths):
            changes = rating_changes[model_path]
            average_change = sum(changes) / len(changes)
            self.ratings[model_path] += average_change
            self.games_played[model_path] += 1
            final_changes[model_path] = average_change

        return final_changes

    def get_leaderboard(self) -> list[tuple[str, float, int]]:
        leaderboard = [
            (path, self.ratings[path], self.games_played.get(path, 0))
            for path in self.ratings.keys()
        ]

        return sorted(leaderboard, key = lambda x: x[1], reverse = True)

