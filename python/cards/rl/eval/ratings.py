from __future__ import annotations

from dataclasses import dataclass

import itertools as it
from typing import TYPE_CHECKING

import numpy as np

from cards.rl.eval.results import SetResult

if TYPE_CHECKING:
    from cards.rl.checkpoint_io import ModelRegistry


@dataclass(frozen = True)
class MatchPairs:
    n_models:       int
    n_sets:         int
    winner_idxs:    np.ndarray  # (P,)
    loser_idxs:     np.ndarray  # (P,)
    set_idxs:       np.ndarray  # (P,)
    weights:        np.ndarray  # (P,)

    def __post_init__(self) -> None:
        if self.winner_idxs.ndim != 1:
            raise ValueError(f'[winner_idxs.ndim] must be 1 (got {self.winner_idxs.ndim})')
        if self.loser_idxs.ndim != 1:
            raise ValueError(f'[loser_idxs.ndim] must be 1 (got {self.loser_idxs.ndim})')
        if self.set_idxs.ndim != 1:
            raise ValueError(f'[set_idxs.ndim] must be 1 (got {self.set_idxs.ndim})')
        if self.weights.ndim != 1:
            raise ValueError(f'[weights.ndim] must be 1 (got {self.weights.ndim})')

        if self.winner_idxs.shape != self.loser_idxs.shape:
            raise ValueError(f'Shapes must match, got [{self.winner_idxs.shape = }] and [{self.loser_idxs.shape}]')
        if self.winner_idxs.shape != self.set_idxs.shape:
            raise ValueError(f'Shapes must match, got [{self.winner_idxs.shape = }] and [{self.set_idxs.shape}]')
        if self.winner_idxs.shape != self.weights.shape:
            raise ValueError(f'Shapes must match, got [{self.winner_idxs.shape = }] and [{self.weights.shape}]')

    def __len__(self) -> int:
        return self.winner_idxs.size

class ModelList:
    def __init__(self, models: list[ModelRegistry.ModelId]) -> None:
        if len(models) != len(set(models)):
            raise ValueError('Argument [models] must contain unique model paths')

        self.models = models

        self.model_to_idx: dict[ModelRegistry.ModelId, int] = {model: idx for idx, model in enumerate(models)}

    def __len__(self) -> int:
        return len(self.models)

    def get_model(self, idx: int) -> ModelRegistry.ModelId | None:
        if idx < 0 or idx >= len(self.models):
            return None
        else:
            return self.models[idx]

    def get_idx(self, model: ModelRegistry.ModelId) -> int | None:
        return self.model_to_idx.get(model, None)

def get_actual_score(final_score_a: float, final_score_b: float, scale: float | None = 200.0) -> float:
    if scale is None:
        if final_score_a > final_score_b:
            return 1.0
        if final_score_a < final_score_b:
            return 0.0
        return 0.5

    return 1.0 / (1.0 + 10.0 ** ((final_score_b - final_score_a) / scale))

def to_pairs(set_results: list[SetResult], model_list: ModelList, scale: float | None = 200.0) -> MatchPairs:
    winner_idxs: list[int] = []
    loser_idxs: list[int] = []
    set_idxs: list[int] = []
    weights: list[float] = []

    for set_idx, set_result in enumerate(set_results):
        for (id_a, score_a), (id_b, score_b) in it.combinations(set_result.aggregate_scores.items(), 2):
            w_ab = get_actual_score(score_a, score_b, scale)
            idx_a = model_list.get_idx(id_a)
            idx_b = model_list.get_idx(id_b)

            if idx_a is None:
                raise ValueError(f'Argument [set_results] at index {set_idx} could not find model {id_a} in argument [model_list]')
            if idx_b is None:
                raise ValueError(f'Argument [set_results] at index {set_idx} could not find model {id_b} in argument [model_list]')

            winner_idxs.extend([idx_a, idx_b])
            loser_idxs.extend([idx_b, idx_a])
            set_idxs.extend([set_idx, set_idx])
            weights.extend([w_ab, 1 - w_ab])

    return MatchPairs(
        n_models = len(model_list),
        n_sets = len(set_results),
        winner_idxs = np.array(winner_idxs, dtype = np.intp),
        loser_idxs = np.array(loser_idxs, dtype = np.intp),
        set_idxs = np.array(set_idxs, dtype = np.intp),
        weights = np.array(weights, dtype = np.float64)
    )

# From match_pairs, W[i, j] = total win credit model_i earned against model_j
def get_win_credit_matrix(match_pairs: MatchPairs, set_counts: np.ndarray | None = None) -> np.ndarray:
    weights = match_pairs.weights

    if set_counts is not None:
        if set_counts.ndim != 1:
            raise ValueError(f'[set_counts.ndim] must be 1 (got {set_counts.ndim})')
        if set_counts.size != match_pairs.n_sets:
            raise ValueError(f'Argument [set_counts] must have size equal to argument [match_pairs.n_sets] ({set_counts.size} != {match_pairs.n_sets})')

        weights = weights * set_counts[match_pairs.set_idxs]

    n = match_pairs.n_models
    W = np.zeros((n, n))

    np.add.at(W, (match_pairs.winner_idxs, match_pairs.loser_idxs), weights)

    return W

def bradley_terry(win_credit_matrix: np.ndarray, prior: float = 0.5, iters: int = 1000, tol: float = 1e-10) -> np.ndarray:
    if win_credit_matrix.ndim != 2:
        raise ValueError(f'[win_credit_matrix.ndim] must be 2 (got {win_credit_matrix.ndim})')
    if win_credit_matrix.shape[0] != win_credit_matrix.shape[1]:
        raise ValueError(f'Argument [win_credit_matrix] must be square (got shape {win_credit_matrix.shape})')

    W = win_credit_matrix

    n = W.shape[0]
    N = W + W.T
    W_i = W.sum(axis = 1) + (prior / 2)

    gamma = np.ones(n)

    for _ in range(iters):
        gamma_old = gamma.copy()
        denominator = (N / (gamma_old[:, None] + gamma_old[None, :])).sum(axis = 1) + prior / (gamma_old + 1)

        has_info = denominator > 0
        gamma = np.where(has_info, W_i / np.where(has_info, denominator, 1.0), gamma_old)
        gamma = np.clip(gamma, 1e-12, None)

        gamma /= np.exp(np.mean(np.log(gamma)))
        if np.max(np.abs(np.log(gamma) - np.log(gamma_old))) < tol:
            break

    return gamma

def to_elo(strength_vector: np.ndarray, center_rating: float = 1500.0, scale: float = 400.0) -> np.ndarray:
    if strength_vector.ndim != 1:
        raise ValueError(f'[strength_vector.ndim] must be 1 (got {strength_vector.ndim})')

    elo = (scale / np.log(10.0)) * np.log(strength_vector)
    return elo - elo.mean() + center_rating

def resample_ratings(match_pairs: MatchPairs, n_samples: int, rng: np.random.Generator | None, prior: float = 0.5) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    samples = np.zeros((n_samples, match_pairs.n_models))

    for i in range(n_samples):
        sets = rng.integers(0, match_pairs.n_sets, size = match_pairs.n_sets)
        set_counts = np.bincount(sets, minlength = match_pairs.n_sets)
        W_sample = get_win_credit_matrix(match_pairs, set_counts = set_counts)
        samples[i] = to_elo(bradley_terry(W_sample, prior = prior))

    return samples  # (n_samples, n_models)

# P[i, j] = fraction of samples where model_i rated above model_j
def get_rank_confidence_matrix(samples) -> np.ndarray:
    # samples.shape: (n_samples, n_models)

    if samples.ndim != 2:
        raise ValueError(f'[samples.ndim] must be 2 (got {samples.ndim})')

    return (samples[:, :, None] > samples[:, None, :]).mean(axis = 0)

