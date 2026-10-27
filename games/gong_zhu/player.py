from __future__ import annotations

import argparse
import copy
import enum
import gc
import itertools as it
import json
import math
import os
import random
import re
import sys
import time
import traceback
import warnings

import asyncio
from websockets.asyncio.client import ClientConnection, connect
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from typing import TYPE_CHECKING, Any, cast

from cards.rl.eval.elo_rating import EloRatingSystem
from cards.rl.nn.blocks import TrickSequenceEncoder
from cards.rl import checkpoint_io
from cards.rl import torch_utils

try:
    from cards import gong_zhu
except ImportError:
    gong_zhu = None

if TYPE_CHECKING:
    from cards.gong_zhu import Env as NativeEnv


GAME_SETTINGS = {
    'NUM_PLAYERS':  4,
    'NUM_CARDS':    52 * 1
}

class GameStates(enum.Enum):
    SHOW_3 =        0
    SHOW_ALL =      1
    PLAY_0 =        2
    PLAY_1 =        3
    PLAY_2 =        4
    PLAY_3 =        5
    SCORE =         6
    LEADERBOARD =   7


def update_environment_settings(msg: dict) -> None:
    GAME_SETTINGS['NUM_CARDS'] =    52 * msg['data']['gameData']['numDecks']
    if len(msg['data']['gameData']['turnOrder']):
        GAME_SETTINGS['NUM_PLAYERS'] =  len(msg['data']['gameData']['turnOrder'])

def one_hot_encode(size: int, arr: np.ndarray, axis: int = 0, **kwargs) -> np.ndarray:
    if arr.ndim <= 1:
        out = np.zeros((size,), **kwargs)
        if arr.size:
            out[arr.astype(np.intp, copy = False)] = 1
        return out

    def one_hot_slice(arr: np.ndarray) -> np.ndarray:
        out = np.zeros((size,), **kwargs)
        if len(arr):
            out[arr] = 1
        return out

    return np.apply_along_axis(func1d = one_hot_slice, axis = axis, arr = arr)

def card_str_to_idx(card: str) -> int:
    rank_map: dict = {e: i for i, e in enumerate('A 2 3 4 5 6 7 8 9 10 J Q K'.split(' '))}
    suit_map: dict = {e: i for i, e in enumerate('♠♥♦♣')}
    rank: int = rank_map[card[:-1]]
    suit: int = suit_map[card[-1]]
    return rank + (suit * 13)

def json_parser(obj: dict) -> Any:
    if type(obj.get('$bigint')) == str:
        return int(obj['$bigint'], base = 10)
    else:
        return obj

class BotConnection(ClientConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.session_id: str | None =   None
        self.username: str | None =     None
        self.connected: bool =          False

class ActorNN(torch.nn.Module):
    def __init__(
        self,
        module_type: str,
        network_dims: tuple[int, ...],
        state_encoder_dims: tuple[int, ...],
        history_hidden_dim: int = 128,
        history_num_layers: int = 1
    ) -> None:

        super().__init__()

        if module_type not in {'SHOW', 'PLAY'}:
            raise ValueError(f'Type [{module_type}] undefined')

        self.module_type = module_type

        self.state_encoder = torch.nn.Sequential(
            *[e for dim in state_encoder_dims for e in (torch.nn.LazyLinear(dim), torch.nn.ReLU())]
        )

        self.history_encoder = TrickSequenceEncoder(
            hidden_dim = history_hidden_dim,
            num_gru_layers = history_num_layers
        )

        self.network = torch.nn.Sequential(
            *[e for dim in network_dims for e in (torch.nn.LazyLinear(dim), torch.nn.ReLU())][:-1]
        )

    def forward(self, states: dict[str, torch.Tensor], mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs.shape: (B, dim(obs))
        # mask.shape:   (B, dim(act))

        state, play_history, leader_history = MultiAgentEnv.serialize_state(states)

        state = self.state_encoder(state)
        history = self.history_encoder(play_history, leader_history)

        inputs = torch.concatenate([state, history], dim = -1)

        logits = self.network(inputs)

        if self.module_type == 'SHOW':
            masked_logits = logits.masked_fill(mask == 0, float('-inf'))
            probs = torch.sigmoid(masked_logits)
            if self.training:
                actions = torch.bernoulli(probs)
            else:
                actions = (probs > 0.5).float()

            eps = 1e-9
            log_probs_bits = actions * torch.log(probs + eps) + (1 - actions) * torch.log(1 - probs + eps)
            log_probs = log_probs_bits.sum(dim = -1)

        elif self.module_type == 'PLAY':
            masked_logits = logits.masked_fill(mask == 0, float('-inf'))
            probs = torch.softmax(masked_logits, dim = -1)

            if self.training:
                action = torch.multinomial(probs, num_samples = 1).squeeze(-1)
            else:
                action = torch.argmax(probs, dim = -1)

            log_probs = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1))
            actions = F.one_hot(action, num_classes = logits.shape[-1]).float()
        else:
            raise ValueError(f'Unknown module_type [{self.module_type}]')

        return actions, log_probs

    def calculate_action_mask(self, state: dict[str, np.ndarray]) -> np.ndarray:
        legal_cards = set()

        game_state = [game_state.name for game_state in GameStates if game_state.value == np.argmax(state['game_state']).item()][0]

        if game_state in {'SHOW_3', 'SHOW_ALL'}:
            exposed = np.any(state['exposed'] > 1, axis = 0)
            legal_cards.update(card for card in [11, 13, 36, 48] if state['hand'][card] == 1 and not exposed[card])

        elif game_state in {'PLAY_0'}:
            hand = np.where(state['hand'] == 1)[0].tolist()
            exposed = np.where(np.any(state['exposed'] > 1, axis = 0))[0].tolist()
            hidden = set(hand) - set(exposed)
            legal_cards.update(hidden)

            my_exposed = set(hand) & set(exposed)
            suit_lens = {}
            for key, group in it.groupby(sorted(hand), key = lambda e: e // 13):
                suit_lens[key] = len(list(group))

            legal_cards.update(e for e in my_exposed if suit_lens[e // 13] == 1)

        elif game_state in {'PLAY_1', 'PLAY_2', 'PLAY_3'}:
            trick = np.where(np.count_nonzero(state['leader_history'], axis = 1) != 0)[0][-1]

            assert np.count_nonzero(state['leader_history'][trick]) == 1
            leader = np.argmax(state['leader_history'][trick])

            assert np.count_nonzero(state['play_history'][trick][leader]) == 1
            trick_suit = np.argmax(state['play_history'][trick][leader]) // 13

            hand = np.where(state['hand'] == 1)[0]
            filtered_hand = hand[hand // 13 == trick_suit]

            if len(filtered_hand) == 1:
                legal_cards.update(filtered_hand.tolist())
            elif len(filtered_hand) > 1:
                exposed = np.where(np.any(state['exposed'] > 1, axis = 0))[0].tolist()
                hidden = set(filtered_hand.tolist()) - set(exposed)
                legal_cards.update(hidden)
            else:
                legal_cards.update(hand.tolist())

        else:
            raise ValueError(f'Unknown game_state [{game_state}]')

        out_dim = GAME_SETTINGS['NUM_CARDS']
        mask = np.zeros(out_dim, dtype = np.float32)
        valid = np.fromiter(legal_cards, dtype = np.intp)
        mask[valid[valid < out_dim]] = 1

        # hand = np.where(state['hand'] == 1)[0]
        # masked_hand = np.where(np.isin(hand, list(legal_cards)), 1, 0)
        # padded_hand = np.pad(masked_hand, pad_width = (0, out_dim - len(masked_hand)), mode = 'constant', constant_values = 0)

        return mask

    def evaluate_actions(self, states: dict[str, torch.Tensor], actions: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # states.shape:     (B, dim(obs))
        # actions.shape:    (B, dim(act))
        # masks.shape:      (B, dim(act))

        state, play_history, leader_history = MultiAgentEnv.serialize_state(states)

        state = self.state_encoder(state)
        history = self.history_encoder(play_history, leader_history)

        inputs = torch.concatenate([state, history], dim = -1)

        logits = self.network(inputs)

        eps = 1e-9

        if self.module_type == 'SHOW':
            num_valid = masks.sum(dim = -1).clamp(min = 1)

            masked_logits = logits.masked_fill(masks == 0, float('-inf'))
            probs = torch.sigmoid(masked_logits)
            probs = torch.clamp(probs, eps, 1 - eps)

            log_probs_bits = actions * torch.log(probs + eps) + (1 - actions) * torch.log(1 - probs + eps)
            log_probs_bits = log_probs_bits * masks
            log_probs = log_probs_bits.sum(dim = -1)

            valid_probs = probs * masks
            entropy_bits = -(valid_probs * torch.log(probs + eps) + (1 - valid_probs) * torch.log(1 - probs + eps))
            entropy_bits = entropy_bits * masks
            entropy = (entropy_bits.sum(dim = -1) / num_valid).mean()

        elif self.module_type == 'PLAY':
            masked_logits = logits.masked_fill(masks == 0, float('-inf'))
            probs = torch.softmax(masked_logits, dim = -1)
            probs = torch.clamp(probs, eps, 1 - eps)

            action_indices = actions.argmax(dim = -1)

            log_probs = torch.log(probs.gather(-1, action_indices.unsqueeze(-1)).squeeze(-1) + eps)

            valid_probs = probs * masks
            valid_probs = valid_probs / valid_probs.sum(dim = -1, keepdim = True).clamp(min = eps)
            entropy = -(valid_probs * torch.log(valid_probs + eps)).sum(dim = -1).mean()
        else:
            raise ValueError(f'Unknown module_type [{self.module_type}]')

        # log_probs.shape:  (B,)
        # entropy.shape:    (,)
        return log_probs, entropy

    @staticmethod
    def get_module_type_from_game_state(game_state: str) -> str:
        if game_state.startswith('SHOW'):
            return 'SHOW'
        elif game_state.startswith('PLAY'):
            return 'PLAY'
        else:
            raise ValueError(f'Unknown game_state [{game_state}]')

class CriticNN(torch.nn.Module):
    def __init__(
        self,
        network_dims: tuple[int, ...],
        state_encoder_dims: tuple[int, ...],
        history_hidden_dim: int = 128,
        history_num_layers: int = 1
    ) -> None:

        super().__init__()

        self.state_encoder = torch.nn.Sequential(
            *[e for dim in state_encoder_dims for e in (torch.nn.LazyLinear(dim), torch.nn.ReLU())]
        )

        self.history_encoder = TrickSequenceEncoder(
            hidden_dim = history_hidden_dim,
            num_gru_layers = history_num_layers
        )

        self.network = torch.nn.Sequential(
            *[e for dim in network_dims for e in (torch.nn.LazyLinear(dim), torch.nn.LayerNorm(dim), torch.nn.ReLU())],
            torch.nn.LazyLinear(1)
        )

    def forward(self, states: dict[str, torch.Tensor]) -> torch.Tensor:
        # inputs.shape: (B, dim(obs))
        # return.shape: (B,)

        state, play_history, leader_history = MultiAgentEnv.serialize_state(states)

        state = self.state_encoder(state)
        history = self.history_encoder(play_history, leader_history)

        inputs = torch.concatenate([state, history], dim = -1)

        return self.network(inputs).squeeze(-1)

class ActorDict(torch.nn.ModuleDict):
    def __getitem__(self, key: str) -> ActorNN:
        return cast(ActorNN, super().__getitem__(key))

class MultiAgentEnv:
    serialize_state_order = {
        'state': ('game_state', 'hand', 'scores', 'current_trick', 'collected_cards', 'exposed'),
        'history': ('play_history', 'leader_history')
    }

    class ModelModes(enum.Enum):
        train =         0
        infer =         1
        eval =          2

    def __init__(self,
        args: argparse.Namespace | None = None,
        mode: MultiAgentEnv.ModelModes | None = None,
        actor: ActorDict | None = None,
        critic: CriticNN | None = None) -> None:

        self.args = args if args is not None else argparse.Namespace()

        self.device: torch.device = torch_utils.resolve_device(getattr(self.args, 'device', 'auto'))

        self._init_hyperparameters()

        if mode is None:
            mode = MultiAgentEnv.ModelModes.train

        self.num_envs: int =                                                        getattr(self.args, 'n_workers', 1)
        self.num_agents: int =                                                      GAME_SETTINGS['NUM_PLAYERS']
        self.ws_list: list[BotConnection | None] =                                  [None               for _ in range(self.num_agents)]

        self.mode: MultiAgentEnv.ModelModes =                                       mode
        self.batch_num: int =                                                       0
        self.save_checkpoint_frequency: int =                                       20

        self._is_processing_event: list[bool] =                                     [False              for _ in range(self.num_agents)]
        self._event_queue: list[list[tuple[str | bytes, float]]] =                  [list()             for _ in range(self.num_agents)] # (message, timestamp)


        self.training_history: dict[str, list[int | float]] = {}

        self.reset()

        self.actor: ActorDict = (actor if actor is not None else self.create_actor()).to(self.device)
        self.critic: torch.nn.Module = (critic if critic is not None else self.create_critic()).to(self.device)

        if getattr(self.args, 'compile', False):
            self.compile_models()

        self.actor_optimizer: torch.optim.Optimizer | None =    None
        self.critic_optimizer: torch.optim.Optimizer | None =   None

        self.elo_rating_system: EloRatingSystem =   EloRatingSystem(k_factor = self.args.k_factor)
        self.eval_games_per_match: int =            20
        self.eval_total_matches: int =              0
        self.eval_total_games: int =                0


    def _init_hyperparameters(self) -> None:
        self.timesteps_per_batch: int =             getattr(self.args, 'timesteps_per_batch', 2 ** 16)
        self.max_timesteps_per_episode: int =       200

        self.reward_scale: float =                  1.0 / 100
        self.opponent_temperature: float =          0.05

        # PPO Parameters
        self.gamma: float =                         0.95
        self.gae_lambda: float =                    0.95
        self.clip_epsilon: float =                  0.2
        self.n_updates_per_batch: int =             8
        self.mini_batch_size: int =                 256
        self.actor_lr: float =                      2e-5
        self.critic_lr: float =                     3e-4
        self.entropy_coef: float =                  0.01
        self.max_grad_norm: float =                 0.5
        self.huber_loss_delta: float =              2.0

    def reset(self) -> None:
        num_tricks = math.ceil(GAME_SETTINGS['NUM_CARDS'] / GAME_SETTINGS['NUM_PLAYERS'])

        self.num_streams: int =                                         self.num_envs * self.num_agents


        self._turn_order: list[str | None] =                            [None                   for _ in range(self.num_agents)]

        self._latest_observation: list[dict | None] =                   [None                   for _ in range(self.num_streams)]
        self._latest_state: list[dict[str, np.ndarray] | None] =        [None                   for _ in range(self.num_streams)]
        self._latest_actions: list[list[dict[str, Any]]] =              [list()                 for _ in range(self.num_streams)]

        self._play_history: list[np.ndarray] =                          [np.zeros((
                                                                            num_tricks,
                                                                            GAME_SETTINGS['NUM_PLAYERS'],
                                                                            GAME_SETTINGS['NUM_CARDS']
                                                                        ), dtype = np.int8)     for _ in range(self.num_streams)]
        self._leader_history: list[np.ndarray] =                        [np.zeros((
                                                                            num_tricks,
                                                                            GAME_SETTINGS['NUM_PLAYERS']
                                                                        ), dtype = np.int8)     for _ in range(self.num_streams)]

        self._opponent_actors: list[ActorDict | None] =                 [None                   for _ in range(self.num_streams)]


        self._batch_states: list[list[dict[str, torch.Tensor]]] =       [list()                 for _ in range(self.num_streams)]   # (B, dim(obs))
        self._batch_actions: list[list[torch.Tensor]] =                 [list()                 for _ in range(self.num_streams)]   # (B, dim(act))
        self._batch_action_masks: list[list[torch.Tensor]] =            [list()                 for _ in range(self.num_streams)]   # (B, dim(act))
        self._batch_log_probs: list[list[torch.Tensor]] =               [list()                 for _ in range(self.num_streams)]   # (B,)
        self._batch_rewards: list[list[np.ndarray]] =                   [list()                 for _ in range(self.num_streams)]   # (E, t_per_E)
        self._batch_lens: list[list[np.ndarray]] =                      [list()                 for _ in range(self.num_streams)]   # (E,)

        self._episode_ts: list[int] =                                   [-1                     for _ in range(self.num_streams)]
        self._episode_rewards: list[list[np.ndarray]] =                 [list()                 for _ in range(self.num_streams)]   # (t_per_E,)

        self.eval_actors: list[ActorDict | None] =                      [None                   for _ in range(self.num_agents)]
        self.eval_model_paths: list[str | None] =                       [None                   for _ in range(self.num_agents)]
        self.eval_scores: list[float] =                                 [0.0                    for _ in range(self.num_agents)]

        self._rollout_progressbar: tqdm | None =                        None

    def latest_observation(self, ws_idx: int) -> dict:
        observation = self._latest_observation[ws_idx]
        assert observation is not None, f'No observation received yet for [{ws_idx}]'
        return observation

    def latest_state(self, ws_idx: int) -> dict[str, np.ndarray]:
        state = self._latest_state[ws_idx]
        assert state is not None, f'No state received yet for [{ws_idx}]'
        return state

    def reset_game_state(self, ws_idx: int) -> None:
        num_tricks = math.ceil(GAME_SETTINGS['NUM_CARDS'] / GAME_SETTINGS['NUM_PLAYERS'])

        self._play_history[ws_idx] =            np.zeros((num_tricks, GAME_SETTINGS['NUM_PLAYERS'], GAME_SETTINGS['NUM_CARDS']), dtype = np.int8)
        self._leader_history[ws_idx] =          np.zeros((num_tricks, GAME_SETTINGS['NUM_PLAYERS']), dtype = np.int8)
        self._latest_observation[ws_idx] =      None
        self._latest_state[ws_idx] =            None
        self._latest_actions[ws_idx] =          list()

    def set_mode(self, mode: MultiAgentEnv.ModelModes) -> None:
        self.mode = mode

    def compile_models(self) -> None:
        dummy = {k: v.to(self.device) for k, v in MultiAgentEnv.get_dummy_state().items()}
        dummy_mask = torch.ones(1, GAME_SETTINGS['NUM_CARDS'], device = self.device)
        self.critic, _ = torch_utils.compile_models(self.actor, self.critic, dummy, dummy_mask)

    def critic_values(self, states: dict[str, torch.Tensor]) -> torch.Tensor:
        n = next(iter(states.values())).shape[0]
        outs = []
        with torch.no_grad():
            for start in range(0, n, self.mini_batch_size):
                mini_batch = slice(start, start + self.mini_batch_size)
                outs.append(self.critic({k: v[mini_batch] for k, v in states.items()}))
        return torch.cat(outs, dim = 0)

    def create_actor(self) -> ActorDict:
        return ActorDict({
            'SHOW': ActorNN('SHOW', (1024, 512, 256, 128, GAME_SETTINGS['NUM_CARDS']), (1024, 512, 256), 256, 2),
            'PLAY': ActorNN('PLAY', (1024, 512, 256, 128, GAME_SETTINGS['NUM_CARDS']), (1024, 512, 256), 256, 2)
        })

    def create_critic(self) -> CriticNN:
        return CriticNN((2048, 1024, 512, 256, 128), (1028, 512, 256), 256, 2)

    def save_checkpoint(self, file_name: str | None = None) -> None:
        if file_name is None:
            file_name = f'gong_zhu_player_{self.batch_num}.pt'

        checkpoint_io.save_checkpoint(
            os.path.join(self.args.model_dir, file_name), self.actor, self.critic,
            batch_num = self.batch_num,
            training_history = self.training_history
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = checkpoint_io.load_checkpoint(path, self.device)

        self.batch_num = checkpoint['batch_num']
        torch_utils.unwrap(self.actor).load_state_dict(checkpoint['actor_state_dict'])
        torch_utils.unwrap(self.critic).load_state_dict(checkpoint['critic_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f'Loaded checkpoint from [{os.path.abspath(path)}] (at batch {self.batch_num})')

    def load_actor_from_checkpoint(self, model_path: str) -> ActorDict:
        actor = checkpoint_io.load_actor(
            model_path, self.create_actor,
            MultiAgentEnv.get_dummy_state(), torch.ones(1, GAME_SETTINGS['NUM_CARDS']),
            self.device
        )
        return cast(ActorDict, actor)

    def get_opponent_sampling_weights(self, model_paths: list[str]) -> np.ndarray:
        n = len(model_paths)

        weights = np.exp(np.linspace(-n * self.opponent_temperature, 0, n))
        weights /= weights.sum()

        return weights

    def learner_seat(self, env_idx: int) -> int:
        return env_idx % self.num_agents

    def load_opponent_models(self) -> None:
        self._opponent_actors = [None for _ in range(self.num_streams)]

        model_paths = checkpoint_io.list_checkpoints(self.args.model_dir)
        model_paths = [path for path in model_paths if os.path.basename(path) != 'latest_model.pt']

        if len(model_paths) == 0:
            return

        log_path = None
        if self.args.model_dir is not None:
            os.makedirs(self.args.model_dir, exist_ok = True)
            log_path = os.path.join(self.args.model_dir, self.args.log_file)

        if self.args.opponent_sampling == 'latest':
            weights = np.zeros(len(model_paths))
            weights[-1] = 1
        elif self.args.opponent_sampling == 'mixed':
            weights = self.get_opponent_sampling_weights(model_paths)
        else:
            raise ValueError(f'Unknown opponent_sampling [{self.args.opponent_sampling}]')

        actor_cache: dict[str, ActorDict] = {}
        selected_counts: dict[str, int] = {}
        for env_idx in range(self.num_envs):
            learner = self.learner_seat(env_idx)
            for seat in range(self.num_agents):
                stream = env_idx * self.num_agents + seat
                if seat == learner:
                    self._opponent_actors[stream] = None
                    continue
                path = np.random.choice(model_paths, p = weights)
                if path not in actor_cache:
                    actor_cache[path] = self.load_actor_from_checkpoint(path)
                self._opponent_actors[stream] = actor_cache[path]
                selected_counts[path] = selected_counts.get(path, 0) + 1

        log_lines = [f'Selected model [{path}] as opponent (x{selected_counts[path]})' for path in model_paths if path in selected_counts.keys()]

        if log_path is not None:
            with open(log_path, 'a') as f:
                f.writelines(line + '\n' for line in log_lines)

        else:
            for line in log_lines:
                print(line)

    async def connect(self, url: str) -> None:
        if self.mode == MultiAgentEnv.ModelModes.infer:
            self.actor.eval()

            # Only One Agent
            ws_idx = 0
            async with connect(url, create_connection = BotConnection) as ws:
                assert isinstance(ws, BotConnection)
                self.ws_list[ws_idx] = ws

                await ws.send(json.dumps({'tag': 'requestSessionID', 'timestamp': int(time.time() * 1000)}))
                await self._listen(ws_idx)

    def get_observation(self, env: NativeEnv, env_idx: int, seat: int) -> None:
        stream = env_idx * self.num_agents + seat

        prev_state = self._latest_state[stream]
        prev_observation = self._latest_observation[stream]

        observation = env.raw_state(seat)
        self.update_latest_from_gui_observation(observation, stream, seat)

        prev_frame = prev_observation['currentFrame'] if prev_observation is not None else None
        if prev_frame is not None and prev_frame == observation['currentFrame']:
            return

        new_state = self._latest_state[stream]
        if prev_state is not None and new_state is not None and len(self._episode_rewards[stream]) > 0:
            reward = self.get_reward_from_state_transition(prev_state, new_state)
            self._episode_rewards[stream][-1] += reward

    def action_to_commands(self, action: torch.Tensor, observation: dict, seat: int) -> list[str]:
        commands: list[str] = []
        game_state = observation['gameState']

        if game_state.startswith('SHOW') or game_state.startswith('PLAY'):
            cards_to_play = torch.nonzero(action, as_tuple = True)[0].tolist()
            cards_to_play_idxs = np.where(np.isin(observation['hands'][seat][0], cards_to_play))[0].tolist()
            if game_state.startswith('PLAY') or torch.sum(action).item() != 0:
                commands.append('PLAY ' + ' '.join(str(e) for e in cards_to_play_idxs))
            if game_state.startswith('SHOW'):
                commands.append('PASS')

        return commands

    async def train(self) -> None:
        gc.collect()
        gc.freeze()
        gc.disable()

        while True:
            trajectories = self.rollout_one_batch()
            await self.train_post_rollout(trajectories)

    def rollout_one_batch(self) -> dict[str, list]:
        if gong_zhu is None:
            raise RuntimeError('Native module unavailable; build/install it (from cards import gong_zhu)')

        self.reset()
        self.load_opponent_models()
        self.actor.train()
        self.critic.train()

        self._turn_order = [str(seat) for seat in range(self.num_agents)]

        envs = [
            gong_zhu.Env({'expose3': True, 'zhu_yang_man_juan': True, 'num_players': self.num_agents})
            for _ in range(self.num_envs)
        ]

        self._rollout_progressbar = tqdm(
            range(self.timesteps_per_batch),
            dynamic_ncols = True,
            desc = f'Batch {self.batch_num} Rollout (x{self.num_envs})'
        )

        for env_idx, env in enumerate(envs):
            env.reset(random.getrandbits(32))
            for seat in range(self.num_agents):
                self.get_observation(env, env_idx, seat)
            self.deal_round(env, env_idx, fresh_game = False)

        total_training = 0
        while total_training < self.timesteps_per_batch:
            pending = self.collect_pending_actions(envs)
            if not pending:
                break

            total_training += self.forward_and_act(pending)

            for env_idx, env in enumerate(envs):
                for seat in range(self.num_agents):
                    self.get_observation(env, env_idx, seat)

            self._rollout_progressbar.n = min(total_training, self.timesteps_per_batch)
            self._rollout_progressbar.refresh()

        for stream in range(self.num_streams):
            self.end_episode(stream)

        if self._rollout_progressbar is not None:
            self._rollout_progressbar.close()

        return self.get_trajectories()

    def deal_round(self, env: NativeEnv, env_idx: int, fresh_game: bool) -> None:
        for seat in range(self.num_agents):
            self.reset_game_state(env_idx * self.num_agents + seat)

        if fresh_game:
            env.reset(random.getrandbits(32))
            for seat in range(self.num_agents):
                self.get_observation(env, env_idx, seat)

        self.apply_command(env, env_idx, 0, 'DEAL')
        for seat in range(self.num_agents):
            self.get_observation(env, env_idx, seat)

    def play_eval_game(self, seed: int) -> list[float]:
        if gong_zhu is None:
            raise RuntimeError('Native module unavailable; build/install it (from cards import gong_zhu)')

        env = gong_zhu.Env({'expose3': True, 'zhu_yang_man_juan': True, 'num_players': self.num_agents})

        for seat in range(self.num_agents):
            self.reset_game_state(seat)

        env.reset(seed)
        for seat in range(self.num_agents):
            self.get_observation(env, 0, seat)

        self.apply_command(env, 0, 0, 'DEAL')
        for seat in range(self.num_agents):
            self.get_observation(env, 0, seat)

        while True:
            game_state = env.game_state()
            if game_state in {'SCORE', 'LEADERBOARD'}:
                break

            seat = env.current_seat()
            if seat < 0:
                break

            module_type = ActorNN.get_module_type_from_game_state(game_state)
            if module_type == '':
                break

            with torch.no_grad():
                actions, log_probs, inputs, mask = self.get_action(seat, module_type, actor = self.eval_actors[seat])

            for command in self.action_to_commands(actions[0], self.latest_observation(seat), seat):
                self.apply_command(env, 0, seat, command)

            for observed_seat in range(self.num_agents):
                self.get_observation(env, 0, observed_seat)

        final_state = self._latest_state[0]
        assert final_state is not None

        return list(final_state['scores'])

    def collect_pending_actions(self, envs: list) -> list[tuple[int, Any, int, str]]:
        pending: list[tuple[int, Any, int, str]] = []

        for env_idx, env in enumerate(envs):
            game_state = env.game_state()

            if game_state in {'SCORE', 'LEADERBOARD'}:
                for seat in range(self.num_agents):
                    self.end_episode(env_idx * self.num_agents + seat)
                self.deal_round(env, env_idx, fresh_game = (game_state == 'LEADERBOARD'))
                game_state = env.game_state()

            seat = env.current_seat()
            if seat < 0:
                continue

            module_type = ActorNN.get_module_type_from_game_state(game_state)
            if module_type == '':
                continue

            pending.append((env_idx * self.num_agents + seat, env, seat, module_type))

        return pending

    def forward_and_act(self, pending: list[tuple[int, Any, int, str]]) -> int:
        # Group decisions by (actor identity, module type) -> one batched forward each
        groups: dict[tuple[int, str], list] = {}
        for stream, env, seat, module_type in pending:
            actor = self.actor if self._opponent_actors[stream] is None else self._opponent_actors[stream]
            groups.setdefault((id(actor), module_type), []).append((stream, env, seat, module_type))

        recorded = 0
        for (_actor_id, module_type), items in groups.items():
            first_stream = items[0][0]
            actor = self.actor if self._opponent_actors[first_stream] is None else self._opponent_actors[first_stream]

            states = [self._latest_state[stream] for stream, env, seat, mt in items]
            inputs = {k: torch.tensor(np.stack([state[k] for state in states]), dtype = torch.float32).to(self.device) for k in states[0]}
            masks = torch.tensor(np.stack([actor[module_type].calculate_action_mask(state) for state in states])).to(self.device)

            with torch.no_grad():
                actions, log_probs = actor[module_type](inputs, masks)

            for i, (stream, env, seat, mt) in enumerate(items):
                if self._opponent_actors[stream] is None:
                    if self._episode_ts[stream] < 0 or self._episode_ts[stream] >= self.max_timesteps_per_episode:
                        self._episode_rewards[stream].clear()
                        self._episode_ts[stream] = 0

                    self._batch_states[stream].append({k: inputs[k][i].cpu() for k in inputs})
                    self._batch_actions[stream].append(actions[i].cpu())
                    self._batch_action_masks[stream].append(masks[i].cpu())
                    self._batch_log_probs[stream].append(log_probs[i].cpu())
                    self._episode_rewards[stream].append(np.array([0.0]))
                    self._episode_ts[stream] += 1
                    recorded += 1

                for command in self.action_to_commands(actions[i], self.latest_observation(stream), seat):
                    self.apply_command(env, stream // self.num_agents, seat, command)

        return recorded

    def apply_command(self, env: NativeEnv, env_idx: int, seat: int, command: str) -> int:
        status, messages = env.apply_command(seat, command)
        self.broadcast_console_messages(env_idx, seat, messages)
        return status

    def broadcast_console_messages(self, env_idx: int, seat: int, messages: list[dict[str, Any]]) -> None:
        for observer in range(self.num_agents):
            console = [message['msg'] for message in messages if message['toAll'] or observer == seat]
            self.update_latest_from_console_observation(console, env_idx * self.num_agents + observer)

    def get_trajectories(self) -> dict[str, list]:
        training_idxs = [i for i in range(self.num_streams) if self._opponent_actors[i] is None]

        return {
            'states':       [self._batch_states[i]          for i in training_idxs],
            'actions':      [self._batch_actions[i]         for i in training_idxs],
            'action_masks': [self._batch_action_masks[i]    for i in training_idxs],
            'log_probs':    [self._batch_log_probs[i]       for i in training_idxs],
            'rewards':      [self._batch_rewards[i]         for i in training_idxs],
            'lens':         [self._batch_lens[i]            for i in training_idxs],
        }

    def update_environment_from_message(self, message: dict) -> None:
        update_environment_settings(message)
        if len(message['data']['gameData']['turnOrder']):
            self._turn_order = message['data']['gameData']['turnOrder']

    async def unlock_processing_event(self, ws_idx: int) -> None:
        if len(self._event_queue[ws_idx]):
            message, timestep = self._event_queue[ws_idx].pop(0)

            asyncio.create_task(self._handle_message(ws_idx, message))

        else:
            self._is_processing_event[ws_idx] = False

    async def _listen(self, ws_idx: int) -> None:
        ws = self.ws_list[ws_idx]
        assert ws is not None
        async for message in ws:
            if self._is_processing_event[ws_idx]:
                self._event_queue[ws_idx].append((message, time.time()))
                continue
            else:
                await self._handle_message(ws_idx, message)

    async def _handle_message(self, ws_idx: int, message: str | bytes) -> None:
        ws = self.ws_list[ws_idx]
        assert ws is not None

        self._is_processing_event[ws_idx] = True

        msg = json.loads(message, object_hook = json_parser)
        # print(ws_idx, msg)

        if msg['tag'] == 'receiveSessionID':
            ws.session_id = msg['data']['sessionID']
            await ws.send(json.dumps({
                'tag': 'requestUsername',
                'data': 'bot',
                'timestamp': int(time.time() * 1000)
            }))

        elif msg['tag'] == 'receiveUsername':
            ws.username = msg['data']

            if self.mode == MultiAgentEnv.ModelModes.infer:
                if ws_idx == 0:
                    await ws.send(json.dumps({
                        'tag': 'getLobbies',
                        'timestamp': int(time.time() * 1000)
                    }))


        elif msg['tag'] == 'showLobby':
            self.reset_game_state(ws_idx)

            if self.mode == MultiAgentEnv.ModelModes.infer:
                if not ws.connected:
                    ws.connected = True

                    self.update_environment_from_message(msg)


        elif msg['tag'] == 'updateLobbies':
            if self.mode == MultiAgentEnv.ModelModes.infer:
                if not ws.connected:
                    servers = msg['data']
                    for server in servers:
                        if (
                            hasattr(self.args, 'lobby') and self.args.lobby == server['name'] and
                            hasattr(self.args, 'host') and self.args.host == server['host']
                        ):
                            await ws.send(json.dumps({
                                'tag': 'joinLobby',
                                'data': server,
                                'timestamp': int(time.time() * 1000)
                            }))

                            await self.unlock_processing_event(ws_idx)
                            return

                    await asyncio.sleep(2)
                    await ws.send(json.dumps({
                        'tag': 'getLobbies',
                        'timestamp': int(time.time() * 1000)
                    }))

        elif msg['tag'] == 'updateGUI':
            if self.mode == MultiAgentEnv.ModelModes.infer:
                self.update_environment_from_message(msg)

                # Spectator or Not In Game
                if ws.username not in self._turn_order:
                    await self.unlock_processing_event(ws_idx)
                    return

                turn_idx = self._turn_order.index(ws.username)
                current_frame = msg['data']['gameData']['currentFrame']

                # Update Internals From Message
                prev_observation = copy.deepcopy(self._latest_observation[ws_idx])
                self.update_latest_from_gui_observation(msg['data']['gameData'], ws_idx, turn_idx)

                self.cleanup_acknowledged_actions(ws_idx, current_frame)
                found_matching_command = self.acknowledged_previous_command(ws_idx, current_frame)

                # Nothing Happened -> Return
                prev_frame = prev_observation['currentFrame'] if prev_observation is not None else None
                new_frame = self.latest_observation(ws_idx)['currentFrame']
                if prev_frame is not None and prev_frame == new_frame:
                    await self.unlock_processing_event(ws_idx)
                    return

                # Game Ended
                game_state = self.latest_observation(ws_idx)['gameState']
                done = game_state in {'LEADERBOARD', 'SCORE'}

                if done:
                    self.reset_game_state(ws_idx)
                    await self.unlock_processing_event(ws_idx)
                    return

                # Needs to Act -> Make Action
                if self.latest_observation(ws_idx)['needToAct'][turn_idx] == 1:

                    # Skip Action if Waiting for ACK
                    if self.is_waiting_for_ack(ws_idx, current_frame):
                        await self.unlock_processing_event(ws_idx)
                        return

                    actor_module_type = ActorNN.get_module_type_from_game_state(self.latest_observation(ws_idx)['gameState'])

                    if actor_module_type == '':
                        await self.unlock_processing_event(ws_idx)
                        return

                    actions, log_probs, inputs, mask = self.get_action(ws_idx, actor_module_type)
                    await self.act(actions[0], ws_idx, turn_idx)


        elif msg['tag'] == 'receiveCommand':
            self.update_latest_from_console_observation(msg['data'], ws_idx)

        elif msg['tag'] == 'commandNACK':
            try:
                found_idx = next(i for i, e in enumerate(self._latest_actions[ws_idx]) if e['command'] == msg['data']['command'] and e['oldFrame'] == msg['data']['oldFrame'])
                self._latest_actions[ws_idx].pop(found_idx)

            except StopIteration:
                pass

            await ws.send(json.dumps({
                'tag': 'getGameState',
                'timestamp': int(time.time() * 1000)
            }))

        elif msg['tag'] == 'commandACK':
            try:
                found_idx = next(i for i, e in enumerate(self._latest_actions[ws_idx]) if e['command'] == msg['data']['command'] and e['oldFrame'] == msg['data']['oldFrame'])
                self._latest_actions[ws_idx][found_idx]['newFrame'] = msg['data']['newFrame']
                self._latest_actions[ws_idx][found_idx]['ack'] = 1

            except StopIteration:
                pass

        await self.unlock_processing_event(ws_idx)

    def update_latest_from_gui_observation(self, observation: dict, ws_idx: int, turn_idx: int) -> None:
        trick: int = round(len(observation['stacks'][0]) / len(observation['turnOrder']))
        partial_state: dict[str, np.ndarray] = MultiAgentEnv.encode_observation(observation, turn_idx)

        if observation['gameState'].startswith('PLAY_'):
            self._leader_history[ws_idx][trick] = one_hot_encode(
                size = GAME_SETTINGS['NUM_PLAYERS'],
                arr = np.array([observation['turnFirstIdx']]),
                dtype = np.int8
            )

            for i, hand in enumerate(observation['hands']):
                self._play_history[ws_idx][trick][i] = one_hot_encode(
                    size = GAME_SETTINGS['NUM_CARDS'],
                    arr = np.array(hand[3]),
                    dtype = np.int8
                )

        state = partial_state | {
            'play_history':     np.roll(self._play_history[ws_idx],     -turn_idx, axis = 1),
            'leader_history':   np.roll(self._leader_history[ws_idx],   -turn_idx, axis = 1)
        }

        self._latest_observation[ws_idx] = observation
        self._latest_state[ws_idx] = state

    def update_latest_from_console_observation(self, observation: list[str], ws_idx: int) -> None:
        for obs in observation:
            match: re.Match | None = re.match('Player \\[(.*)\\] played card \\[(.*)\\]', obs)
            if match:
                username: str; card: str
                username, card = match.groups()
                card_int: int = card_str_to_idx(card)

                if self._latest_observation[ws_idx] is None:
                    continue

                trick: int = round(len(self.latest_observation(ws_idx)['stacks'][0]) / len(self.latest_observation(ws_idx)['turnOrder']))

                if trick >= self._play_history[ws_idx].shape[0]:
                    continue

                player_idx: int = self.latest_observation(ws_idx)['turnOrder'].index(username)
                self._play_history[ws_idx][trick][player_idx] = one_hot_encode(
                    size = GAME_SETTINGS['NUM_CARDS'],
                    arr = np.array([card_int]),
                    dtype = np.int8
                )

    def cleanup_acknowledged_actions(self, ws_idx: int, current_frame: int) -> None:
        self._latest_actions[ws_idx] = [
            action for action in self._latest_actions[ws_idx]
            if not (action['ack'] == 1 and action.get('newFrame') is not None and action['newFrame'] < current_frame)
        ]

    def acknowledged_previous_command(self, ws_idx: int, current_frame: int) -> bool:
        try:
            found_idx = next(i for i, e in enumerate(self._latest_actions[ws_idx]) if e['newFrame'] == current_frame and e['ack'] == 1)
            self._latest_actions[ws_idx].pop(found_idx)
            return True
        except StopIteration:
            return False

    def is_waiting_for_ack(self, ws_idx: int, current_frame: int) -> bool:
        try:
            found_idx = next(i for i, e in enumerate(self._latest_actions[ws_idx]) if e['oldFrame'] == current_frame and e['ack'] == -1)
            return True
        except StopIteration:
            return False

    def get_action(
        self,
        ws_idx: int,
        actor_module_type: str,
        actor: ActorDict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:

        if actor is None:
            actor = self.actor

        inputs = {k: torch.tensor(v, dtype = torch.float32).unsqueeze(0).to(self.device) for k, v in self.latest_state(ws_idx).items()}
        mask = torch.tensor(np.array([actor[actor_module_type].calculate_action_mask(self.latest_state(ws_idx))])).to(self.device)

        actions, log_probs = actor[actor_module_type](inputs, mask)

        return actions, log_probs, inputs, mask

    async def act(self, action: torch.Tensor, ws_idx: int, turn_idx: int) -> None:
        if self._latest_observation[ws_idx] is None:
            return

        commands = self.action_to_commands(action, self.latest_observation(ws_idx), turn_idx)

        ws = self.ws_list[ws_idx]
        assert ws is not None
        for command in commands:
            observation = self._latest_observation[ws_idx]
            self._latest_actions[ws_idx].append({
                'command': command,
                'oldFrame': observation['currentFrame'] if observation is not None else -1,
                'newFrame': None,
                'ack': -1
            })
            await ws.send(json.dumps({
                'tag': 'sendCommand',
                'data': command,
                'timestamp': int(time.time() * 1000),
                'currentFrame': {'$bigint': str(observation['currentFrame'] if observation is not None else -1)}
            }))

    @staticmethod
    def encode_observation(observation: dict, turn_idx: int) -> dict[str, np.ndarray]:

        game_state: np.ndarray = one_hot_encode(
            size = len(GameStates),
            arr = np.array([GameStates[observation['gameState']].value]),
            dtype = np.float32
        )

        hand: np.ndarray = one_hot_encode(
            size = 52,
            arr = np.array(observation['hands'][turn_idx][0]),
            dtype = np.float32
        )

        scores: np.ndarray = np.array(observation['scores'], dtype = np.float32)[:, 1]

        current_trick: np.ndarray = np.array([one_hot_encode(
            size = GAME_SETTINGS['NUM_CARDS'],
            arr = np.array(hand[3]),
            dtype = np.float32
        ) for hand in observation['hands']])

        collected_cards: np.ndarray = np.array([one_hot_encode(
            size = GAME_SETTINGS['NUM_CARDS'],
            arr = np.array(hand[2]),
            dtype = np.float32
        ) for hand in observation['hands']])

        exposed: np.ndarray = np.ones((GAME_SETTINGS['NUM_PLAYERS'], GAME_SETTINGS['NUM_CARDS']), dtype = np.float32)
        values: dict[int, int] = {shown_card: value for shown_card, value in observation['stacks'][1]}
        for i, _hand in enumerate(observation['hands']):
            for card in _hand[1]:
                if card in values.keys():
                    exposed[i, card] = values[card]

        return {
            'game_state':       game_state,
            'hand':             hand,
            'scores':           np.roll(scores,             -turn_idx, axis = 0),
            'current_trick':    np.roll(current_trick,      -turn_idx, axis = 0),
            'collected_cards':  np.roll(collected_cards,    -turn_idx, axis = 0),
            'exposed':          np.roll(exposed,            -turn_idx, axis = 0)
        }

    @staticmethod
    def get_value_from_state(state: dict[str, np.ndarray]) -> float:
        return state['scores'][0] - np.mean(state['scores'])

    def get_reward_from_state_transition(self, old_state: dict[str, np.ndarray], new_state: dict[str, np.ndarray]) -> float:
        raw_reward = MultiAgentEnv.get_value_from_state(new_state) - MultiAgentEnv.get_value_from_state(old_state)
        return raw_reward * self.reward_scale

    @staticmethod
    def get_masked_states(states: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: v[mask] for k, v in states.items()}

    @staticmethod
    def get_dummy_state() -> dict[str, torch.Tensor]:
        num_players = GAME_SETTINGS['NUM_PLAYERS']
        num_cards = GAME_SETTINGS['NUM_CARDS']
        num_tricks = math.ceil(num_cards / num_players)

        return {
            'game_state':       torch.zeros(1, len(GameStates)),
            'hand':             torch.zeros(1, num_cards),
            'scores':           torch.zeros(1, num_players),
            'current_trick':    torch.zeros(1, num_players * num_cards),
            'collected_cards':  torch.zeros(1, num_players * num_cards),
            'exposed':          torch.zeros(1, num_players * num_cards),
            'play_history':     torch.zeros(1, num_tricks * num_players * num_cards),
            'leader_history':   torch.zeros(1, num_tricks * num_players)
        }

    @staticmethod
    def serialize_state(states: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        num_players = GAME_SETTINGS['NUM_PLAYERS']
        num_cards = GAME_SETTINGS['NUM_CARDS']
        num_tricks = math.ceil(num_cards / num_players)

        play_history = states['play_history'].reshape(-1, num_tricks, num_players, num_cards)
        leader_history = states['leader_history'].reshape(-1, num_tricks, num_players)

        cards_played_per_player = play_history.sum(dim = 1).flatten(start_dim = 1)          # (B, P * C)
        cards_remaining = 1 - play_history.sum(dim = (1, 2)).clamp(0, 1)                    # (B, C)
        tricks_won = leader_history.sum(dim = 1)                                            # (B, P)
        trick_number = leader_history.any(dim = -1).sum(dim = -1, keepdim = True).float()   # (B, 1)

        state_parts = [states[k].flatten(start_dim = 1) for k in MultiAgentEnv.serialize_state_order['state']]
        state_parts.extend([cards_played_per_player, cards_remaining, tricks_won, trick_number])
        state = torch.concatenate(state_parts, dim = -1)

        return state, play_history, leader_history

    def end_episode(self, ws_idx: int) -> None:
        if self._episode_ts[ws_idx] <= 0 or len(self._episode_rewards[ws_idx]) == 0:
            return

        self._batch_lens[ws_idx].append(np.array([self._episode_ts[ws_idx]]))
        self._batch_rewards[ws_idx].append(np.concatenate(self._episode_rewards[ws_idx], axis = 0))
        self._episode_ts[ws_idx] = -1
        self._episode_rewards[ws_idx].clear()

    async def train_post_rollout(self, trajectories: dict) -> None:
        eps = 1e-9

        batch_advantages, batch_returns =   self.compute_gaes(trajectories)     # (B,), (B,)

        batch_state_dicts =     [
                                    state for stream in trajectories['states']
                                    for state in stream
                                ]
        if len(batch_state_dicts) == 0:
            return

        batch_states =          {k: torch.stack(
                                    ([s[k].flatten() for s in batch_state_dicts]), dim = 0
                                ) for k in batch_state_dicts[0].keys()}
        batch_actions =         torch.stack([
                                    action for stream in trajectories['actions']
                                    for action in stream
                                ], dim = 0)     # (B, dim(act))
        batch_action_masks =    torch.stack([
                                    action_mask for stream in trajectories['action_masks']
                                    for action_mask in stream
                                ], dim = 0)     # (B, dim(act))
        batch_log_probs =       torch.stack([
                                    log_prob for stream in trajectories['log_probs']
                                    for log_prob in stream
                                ], dim = 0)     # (B)

        batch_states =          {k: v.to(self.device) for k, v in batch_states.items()}
        batch_actions =         batch_actions.to(self.device)
        batch_action_masks =    batch_action_masks.to(self.device)
        batch_log_probs =       batch_log_probs.to(self.device)
        batch_advantages =      batch_advantages.to(self.device)
        batch_returns =         batch_returns.to(self.device)

        game_states = batch_states['game_state']
        show_mask = game_states[:, 0:2].sum(dim = 1) > 0
        play_mask = game_states[:, 2:6].sum(dim = 1) > 0

        for mask in (show_mask, play_mask):
            if mask.sum() > 1:
                phase_advantages = batch_advantages[mask]
                batch_advantages[mask] = (phase_advantages - phase_advantages.mean()) / (phase_advantages.std() + eps)

        batch_values_old = self.critic_values(batch_states)

        if self.actor_optimizer is None:
            self.actor_optimizer =      torch.optim.Adam(self.actor.parameters(),   lr = self.actor_lr,     fused = True)
        if self.critic_optimizer is None:
            self.critic_optimizer =     torch.optim.Adam(self.critic.parameters(),  lr = self.critic_lr,    fused = True)

        batch_len =         len(batch_state_dicts)
        mini_batch_len =    self.mini_batch_size

        actor_losses = []
        critic_losses = []
        show_critic_losses = []
        play_critic_losses = []
        kl_divergences = []
        entropy_values = []
        clip_fractions = []

        for epoch in tqdm(
                range(self.n_updates_per_batch),
                desc = f'Training ({len(batch_state_dicts)} samples): ',
                dynamic_ncols = True
        ):
            indices = torch.randperm(batch_len, device = self.device)

            for start in range(0, batch_len, mini_batch_len):
                end = start + mini_batch_len
                mini_batch_indices = indices[start:end]

                mini_batch_states =         MultiAgentEnv.get_masked_states(batch_states, mini_batch_indices)
                mini_batch_actions =        batch_actions[mini_batch_indices]
                mini_batch_action_masks =   batch_action_masks[mini_batch_indices]
                mini_batch_log_probs =      batch_log_probs[mini_batch_indices]
                mini_batch_advantages =     batch_advantages[mini_batch_indices]
                mini_batch_returns =        batch_returns[mini_batch_indices]
                mini_batch_show_mask =      show_mask[mini_batch_indices]
                mini_batch_play_mask =      play_mask[mini_batch_indices]

                mini_batch_log_probs_new =  torch.zeros(mini_batch_indices.shape[0], device = self.device)
                mini_batch_entropy =        torch.tensor(0.0, device = self.device)

                mini_batch_values_new = self.critic(mini_batch_states)

                if mini_batch_show_mask.any():
                    show_log_probs, show_entropy = self.actor['SHOW'].evaluate_actions(
                        MultiAgentEnv.get_masked_states(mini_batch_states, mini_batch_show_mask),
                        mini_batch_actions[mini_batch_show_mask],
                        mini_batch_action_masks[mini_batch_show_mask]
                    )
                    mini_batch_log_probs_new[mini_batch_show_mask] = show_log_probs
                    mini_batch_entropy += show_entropy

                if mini_batch_play_mask.any():
                    play_log_probs, play_entropy = self.actor['PLAY'].evaluate_actions(
                        MultiAgentEnv.get_masked_states(mini_batch_states, mini_batch_play_mask),
                        mini_batch_actions[mini_batch_play_mask],
                        mini_batch_action_masks[mini_batch_play_mask]
                    )
                    mini_batch_log_probs_new[mini_batch_play_mask] = play_log_probs
                    mini_batch_entropy += play_entropy

                ratios = torch.exp(mini_batch_log_probs_new - mini_batch_log_probs.detach())
                ratios = torch.clamp(ratios, eps, 100.0)

                partial_min_1 = ratios * mini_batch_advantages
                partial_min_2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mini_batch_advantages

                actor_loss = -torch.min(partial_min_1, partial_min_2).mean() - self.entropy_coef * mini_batch_entropy

                critic_loss_per_sample = F.huber_loss(mini_batch_values_new, mini_batch_returns, reduction = 'none', delta = self.huber_loss_delta)
                critic_loss = critic_loss_per_sample.mean()

                if mini_batch_show_mask.any():
                    show_critic_losses.append(critic_loss_per_sample[mini_batch_show_mask].mean().item())
                if mini_batch_play_mask.any():
                    play_critic_losses.append(critic_loss_per_sample[mini_batch_play_mask].mean().item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_values.append(mini_batch_entropy.item())

                with torch.no_grad():
                    kl = (mini_batch_log_probs - mini_batch_log_probs_new).mean()
                    kl_divergences.append(kl.item())

                    clip_fraction = ((ratios - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    clip_fractions.append(clip_fraction)

                # Yield to event loop to keep websocket alive
                await asyncio.sleep(0)

        all_episode_rewards = [rewards.sum() for stream in trajectories['rewards'] for rewards in stream]

        predicted_values = self.critic_values(batch_states)
        show_explained_variance = float('nan')
        play_explained_variance = float('nan')
        with torch.no_grad():
            if show_mask.any():
                show_explained_variance = 1 - ((batch_returns[show_mask] - predicted_values[show_mask]).var() / (batch_returns[show_mask].var() + eps)).item()
            if play_mask.any():
                play_explained_variance = 1 - ((batch_returns[play_mask] - predicted_values[play_mask]).var() / (batch_returns[play_mask].var() + eps)).item()
            explained_variance = 1 - ((batch_returns - predicted_values).var() / (batch_returns.var() + eps))
            fresh_explained_variance = 1 - ((batch_returns - batch_values_old).var() / (batch_returns.var() + eps))

        self.training_history.setdefault('batch', []).append(self.batch_num)
        self.training_history.setdefault('actor_loss', []).append(actor_losses[-1])
        self.training_history.setdefault('critic_loss', []).append(critic_losses[-1])
        self.training_history.setdefault('mean_kl_divergence', []).append(np.mean(kl_divergences))
        self.training_history.setdefault('mean_entropy', []).append(np.mean(entropy_values))
        self.training_history.setdefault('mean_clip_fraction', []).append(np.mean(clip_fractions))
        self.training_history.setdefault('mean_return', []).append(batch_returns.mean().item())
        self.training_history.setdefault('max_episode_reward', []).append(max(all_episode_rewards) if all_episode_rewards else 0)
        self.training_history.setdefault('min_episode_reward', []).append(min(all_episode_rewards) if all_episode_rewards else 0)
        self.training_history.setdefault('mean_episode_reward', []).append(np.mean(all_episode_rewards) if all_episode_rewards else 0)

        log_path = os.path.join(self.args.model_dir, self.args.log_file)
        os.makedirs(self.args.model_dir, exist_ok = True)
        with open(log_path, 'a') as f:
            print(f'-- Training Complete (Batch {self.batch_num}) --', file = f)
            print(f'  Total Samples: {batch_len}', file = f)
            print(f'  Actor Loss: {actor_losses[0]:.4f} -> {actor_losses[-1]:.4f}', file = f)
            print(f'  Critic Loss: {critic_losses[0]:.4f} -> {critic_losses[-1]:.4f}', file = f)
            print(f'  SHOW Critic Loss: {show_critic_losses[0]:.4f} -> {show_critic_losses[-1]:.4f}', file = f)
            print(f'  PLAY Critic Loss: {play_critic_losses[0]:.4f} -> {play_critic_losses[-1]:.4f}', file = f)
            print(f'  SHOW Explained Variance: {show_explained_variance:.4f}', file = f)
            print(f'  PLAY Explained Variance: {play_explained_variance:.4f}', file = f)
            print(f'  SHOW samples: {show_mask.sum().item()} | PLAY samples: {play_mask.sum().item()}', file = f)
            print(f'  Mean KL Divergence: {np.mean(kl_divergences):.4f}', file = f)
            print(f'  Mean Entropy: {np.mean(entropy_values):.4f}', file = f)
            print(f'  Mean Clip Fraction: {np.mean(clip_fractions):.4f}', file = f)
            print(f'  Explained Variance: {explained_variance:.4f}', file = f)
            print(f'  Fresh Explained Variance: {fresh_explained_variance:.4f}', file = f)
            print(f'  Return range: [{batch_returns.min().item():.2f}, {batch_returns.max().item():.2f}]', file = f)
            print(f'  Episode rewards: min={min(all_episode_rewards):.1f}, max={max(all_episode_rewards):.1f}, mean={np.mean(all_episode_rewards):.1f}', file = f)
            print(f'', file = f)

        self.batch_num += 1

        if self.batch_num % self.save_checkpoint_frequency == 0:
            self.save_checkpoint()
        self.save_checkpoint('latest_model.pt')

        gc.collect()

        # with torch.no_grad():
        #   sample_idx = torch.randperm(batch_len)[:20]
        #   sample_states = MultiAgentEnv.get_masked_states(batch_states, sample_idx)
        #   sample_returns = batch_returns[sample_idx]

        #   predicted = torch.zeros(20)
        #   for i, idx in enumerate(sample_idx):
        #       if play_mask[idx]:
        #           predicted[i] = self.critic['PLAY'](
        #               MultiAgentEnv.get_masked_states(batch_states, idx.unsqueeze(0))
        #           )

        #   with open(log_path, 'a') as f:
        #       for i in range(10):
        #           trick_num = sample_states['game_state'][i].argmax().item()
        #           print(f'  State {i}: game_state={trick_num}, '
        #               f'predicted={predicted[i]:.3f}, '
        #               f'actual={sample_returns[i]:.3f}', file = f)

    def compute_gaes(self, trajectories: dict[str, list]) -> tuple[torch.Tensor, torch.Tensor]:
        all_advantages = []
        all_returns = []

        n_streams = len(trajectories['states'])

        for stream_idx in range(n_streams):
            stream_states = trajectories['states'][stream_idx]
            stream_rewards = trajectories['rewards'][stream_idx]

            if len(stream_states) == 0:
                continue

            batch_advantages = []
            batch_returns = []

            stacked_states = {
                k: torch.stack([s[k].flatten() for s in stream_states], dim = 0).to(self.device)
                for k in stream_states[0].keys()
            }

            batch_values = self.critic_values(stacked_states).cpu().numpy()

            value_idx = 0
            for episode_rewards in stream_rewards:
                episode_len = episode_rewards.shape[0]
                if episode_len == 0:
                    continue

                episode_values = batch_values[value_idx : (value_idx + episode_len)]
                value_idx += episode_len

                episode_advantages = np.zeros(episode_len, dtype = np.float32)
                last_gae = 0.0

                for t in reversed(range(episode_len)):
                    if t == episode_len - 1:
                        next_val = 0.0
                    else:
                        next_val = episode_values[t + 1]

                    delta = episode_rewards[t] + self.gamma * next_val - episode_values[t]
                    last_gae = delta + self.gamma * self.gae_lambda * last_gae
                    episode_advantages[t] = last_gae

                episode_returns = episode_advantages + episode_values

                batch_advantages.append(episode_advantages)
                batch_returns.append(episode_returns)

            all_advantages.append(np.concatenate(batch_advantages))
            all_returns.append(np.concatenate(batch_returns))

        if len(all_advantages) == 0:
            return torch.zeros(0), torch.zeros(0)

        return torch.tensor(np.concatenate(all_advantages, axis = 0)), torch.tensor(np.concatenate(all_returns, axis = 0))

    def load_models(self, model_paths: list[str]) -> None:
        assert len(model_paths) == self.num_agents

        for ws_idx, model_path in enumerate(model_paths):
            self.eval_actors[ws_idx] = self.load_actor_from_checkpoint(model_path)
            self.eval_model_paths[ws_idx] = model_path

    async def eval(self) -> None:
        model_paths = checkpoint_io.list_checkpoints(self.args.model_dir)

        if len(model_paths) < self.num_agents:
            print(f'Not enough models found in directory [{os.path.abspath(self.args.model_dir)}] ({len(model_paths)} < {self.num_agents})')
            return

        if hasattr(self.args, 'reset_elo') and self.args.reset_elo:
            self.elo_rating_system.reset_all_ratings(model_paths)
            self.eval_total_matches = 0
            self.eval_total_games = 0

        self.num_envs = 1
        self.reset()
        self._turn_order = [str(seat) for seat in range(self.num_agents)]

        # Eval Loop
        while True:
            # Load Matchup
            batch_models = random.sample(model_paths, self.num_agents)

            batch_model_names = [os.path.basename(p) for p in batch_models]
            batch_model_ratings = [f'{self.elo_rating_system.get_rating(p):.2f}' for p in batch_models]

            log_path = os.path.join(self.args.model_dir, self.args.log_file)
            os.makedirs(self.args.model_dir, exist_ok = True)
            with open(log_path, 'a') as f:
                print(f'', file = f)
                print(f'=== Starting Evaluation Matchup ===', file = f)
                print(f'  Models: {batch_model_names}', file = f)
                print(f'  Ratings: {batch_model_ratings}', file = f)
                print(f'  Sets to Play: {self.eval_games_per_match}', file = f)

            # Play [self.eval_games_per_match] Sets
            for game_idx in range(self.eval_games_per_match):

                set_seed = int(time.time() * 1000) & 0xFFFFFFFF
                set_models = random.sample(batch_models, self.num_agents)
                set_scores = {model_path: 0.0 for model_path in set_models}

                for rotation in range(self.num_agents):
                    rotated_models = [set_models[(i + rotation) % self.num_agents] for i in range(self.num_agents)]
                    self.load_models(rotated_models)

                    self.eval_scores = self.play_eval_game(set_seed)

                    for seat in range(self.num_agents):
                        set_scores[rotated_models[seat]] += self.eval_scores[seat]

                self.eval_total_games += 1

                agg_paths = list(set_scores.keys())
                agg_scores = [set_scores[p] for p in agg_paths]
                rating_changes = self.elo_rating_system.update_ratings(agg_paths, agg_scores)

                log_path = os.path.join(self.args.model_dir, self.args.log_file)
                os.makedirs(self.args.model_dir, exist_ok = True)
                with open(log_path, 'a') as f:
                    print(f' --- Eval Set {game_idx + 1} / {self.eval_games_per_match} (seed {set_seed}) ---', file = f)
                    print(f'  Total Sets: {self.eval_total_games}', file = f)
                    for model_path in agg_paths:
                        model_name = os.path.basename(model_path)
                        rating = self.elo_rating_system.get_rating(model_path)
                        change = rating_changes.get(os.path.abspath(model_path), 0)
                        print(f'  [{model_name}]:', file = f)
                        print(f'    Aggregate Score = {set_scores[model_path]}', file = f)
                        print(f'    ELO = {rating:.4f} ({change:+.4f})', file = f)

                self.save_all_ratings()

            # Matchup Complete
            self.eval_total_matches += 1
            self.print_leaderboard()

    def print_leaderboard(self) -> None:
        leaderboard = self.elo_rating_system.get_leaderboard()

        log_path = os.path.join(self.args.model_dir, self.args.log_file)
        os.makedirs(self.args.model_dir, exist_ok = True)
        with open(log_path, 'a') as f:
            print(f'', file = f)
            print(f'{"=" * 60}', file = f)
            print(f' ELO LEADERBOARD (after {self.eval_total_games} games)', file = f)
            print(f' {"Rank":<6}{"Model":<35}{"ELO":<8}{"Games":<8}', file = f)
            print(f' {"-" * 58} ', file = f)
            for rank, (path, rating, games) in enumerate(leaderboard, 1):
                name = os.path.basename(path)
                print(f' {rank:<6}{name:<35}{rating:<8.2f}{games:<8}', file = f)
            print(f'{"=" * 60}', file = f)

    def save_all_ratings(self) -> None:
        saved_ratings_count = 0
        for model_path in self.elo_rating_system.ratings:
            try:
                self.elo_rating_system.save_rating(model_path)
                saved_ratings_count += 1
            except Exception as e:
                warnings.warn(f'Failed to save rating for [{model_path}]: {e}')

        print(f'Saved ELO ratings for {saved_ratings_count} / {len(self.elo_rating_system.ratings)} models')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--url', '-u', type = str, default = 'ws://localhost:8080',
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
        '--n-workers', '-n', type = int, default = 1,
        help = 'Number of parallel in-process envs to batch across during rollout'
    )
    parser.add_argument(
        '--timesteps-per-batch', '-t', type = int, default = 2 ** 16,
        help = 'Rollout batch size'
    )
    parser.add_argument(
        '--seed', type = int, default = None,
        help = 'Seed random/numpy/torch (+ native deals) for reproducible runs'
    )
    parser.add_argument(
        '--device', type = str, default = 'auto', choices = ['auto', 'cpu', 'cuda', 'mps'],
        help = 'Compute device (auto = cuda/mps if available else cpu)'
    )
    parser.add_argument(
        '--compile', action = 'store_true',
        help = 'Use torch.compile'
    )
    parser.add_argument(
        '--model-dir', type = str,
        help = 'Directory containing/to save model checkpoints'
    )
    parser.add_argument(
        '--opponent-sampling', type = str,
        choices = ['latest', 'mixed'], default = 'latest',
        help = 'Use latest or mixed checkpoints (if available)'
    )
    parser.add_argument(
        '--log-file', type = str, default = 'log.txt',
        help = 'Name of log file to dump logs to'
    )

    # Inference Options
    parser.add_argument(
        '--model', '-m', type = str,
        help = 'Model checkpoint to load for inference'
    )
    parser.add_argument(
        '--lobby', type = str,
        help = 'Lobby name to join for inference'
    )
    parser.add_argument(
        '--host', type = str,
        help = 'Host name of lobby to join for inference'
    )

    # Evaluation Options
    parser.add_argument(
        '--k-factor', type = float, default = 32.0,
        help = 'K-Factor for Elo calculation'
    )
    parser.add_argument(
        '--reset-elo', action = 'store_true',
        help = 'Reset Elo ratings for all models before starting evaluation'
    )

    args = parser.parse_args()

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

    if (args.train or args.eval) and gong_zhu is None:
        parser.error('--train/--eval require the native module; run ./build.sh and activate .venv')

    return args

def crash_on_exception(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    print('=' * 80, flush = True, file = sys.stderr)
    print(f'FATAL: Unhandled exception in asyncio loop', flush = True, file = sys.stderr)
    print(f'Context: {context}', flush = True, file = sys.stderr)
    if 'exception' in context.keys():
        traceback.print_exception(
            type(context['exception']),
            context['exception'],
            context['exception'].__traceback__,
            file = sys.stderr
        )
    print('=' * 80, flush = True, file = sys.stderr)
    os._exit(1)

async def main() -> None:
    args: argparse.Namespace = parse_arguments()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(crash_on_exception)

    if args.train:
        env = MultiAgentEnv(
            args = args,
            mode = MultiAgentEnv.ModelModes.train
        )

        if args.model is not None:
            if os.path.exists(args.model):
                env.load_checkpoint(args.model)
            else:
                warnings.warn(f'Checkpoint not found at {os.path.abspath(args.model)}')

        print('Starting training mode...')
        await env.train()

    elif args.infer or args.eval:
        env = MultiAgentEnv(
            args = args
        )

        if args.model is not None:
            if os.path.exists(args.model):
                env.load_checkpoint(args.model)
            else:
                warnings.warn(f'Checkpoint not found at {os.path.abspath(args.model)}')

                if args.infer:
                    warnings.warn('No checkpoint loaded; bot will use random initialization')

        if args.infer:
            env.set_mode(MultiAgentEnv.ModelModes.infer)
            print('Starting inference mode...')
            await env.connect(args.url)

        elif args.eval:
            env.set_mode(MultiAgentEnv.ModelModes.eval)
            print('Starting evaluation mode...')
            await env.eval()

    else:
        raise ValueError('No mode specified')

if __name__ == '__main__':
    asyncio.run(main())

