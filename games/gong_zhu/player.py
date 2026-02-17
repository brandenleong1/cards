from __future__ import annotations

import argparse
import copy
import enum
import glob
import itertools as it
import json
import math
import os
import random
import re
import tempfile
import time
import urllib.parse
import warnings

import asyncio
import websockets
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from typing import Any


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


NUM_PLAYERS = 4

GAME_SETTINGS = {
	'NUM_PLAYERS':	4,
	'NUM_CARDS':	52 * 1
}

class GameStates(enum.Enum):
	SHOW_3 =		0
	SHOW_ALL =		1
	PLAY_0 =		2
	PLAY_1 =		3
	PLAY_2 =		4
	PLAY_3 =		5
	SCORE =			6
	LEADERBOARD =	7


def updateEnvironmentSettings(msg: dict) -> None:
	GAME_SETTINGS['NUM_CARDS'] =	52 * msg['data']['gameData']['numDecks']
	GAME_SETTINGS['turn_order'] =	msg['data']['gameData']['turnOrder']

def one_hot_encode(size: int, arr: np.ndarray[int], axis: int = 0, **kwargs) -> np.ndarray:
	def one_hot_slice(arr: np.ndarray[int]):
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

class ConsoleListeners:
	def __init__(self) -> None:
		self.listeners = []
		self.idx_to_name = {}
		self.name_to_idx = {}

	def add_ws(self, ws: websockets.asyncio.client.ClientConnection) -> None:
		self.listeners.append(ws)

	def remove_ws(self, ws: websockets.asyncio.client.ClientConnection) -> None:
		self.listeners.remove(ws)

	def broadcast_message(self, message: str) -> None:
		for ws in self.listeners:
			asyncio.create_task(ws.send(message))

	def create_console(self, ws_idx: int, ws_name: str) -> None:
		if ws_idx in self.idx_to_name.keys():
			raise ValueError(f'ws_idx [{ws_idx}] already exists')
		if ws_name in self.name_to_idx.keys():
			raise ValueError(f'ws_name [{ws_name}] already exists')
		self.idx_to_name[ws_idx] = ws_name
		self.name_to_idx[ws_name] = ws_idx
		self.broadcast_message(json.dumps({
			'tag': 'createConsole',
			'data': {'id': ws_name},
			'timestamp': int(time.time() * 100)
		}))

class ActorNN(torch.nn.Module):
	def __init__(self, module_type: str, dims: tuple[int, ...]) -> None:
		super().__init__()

		self.network = torch.nn.Sequential(
			*[e for dim in dims for e in (torch.nn.LazyLinear(dim), torch.nn.ReLU())][:-1]
		)

		if module_type not in {'SHOW', 'PLAY'}:
			raise ValueError(f'Type [{module_type}] undefined')

		self.module_type = module_type

	def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		# inputs.shape:	(B, dim(obs))
		# mask.shape:	(B, dim(act))

		logits = self.network(inputs)

		if self.module_type == 'SHOW':
			masked_logits = logits.masked_fill(mask == 0, torch.tensor(-torch.inf))
			probs = torch.sigmoid(masked_logits)
			if self.training:
				actions = torch.bernoulli(probs)
			else:
				actions = (probs > 0.5).float()

			# pass_bits = actions[..., -1:]
			# actions = torch.where(torch.broadcast_to(pass_bits.bool(), actions.shape), F.pad(torch.zeros_like(actions[..., :-1]), (0, 1), value = 1.0), actions)

			eps = 1e-9
			log_probs_bits = actions * torch.log(probs + eps) + (1 - actions) * torch.log(1 - probs + eps)
			log_probs = log_probs_bits.sum(dim = -1)

		elif self.module_type == 'PLAY':
			masked_logits = logits.masked_fill(mask == 0, torch.tensor(-torch.inf))
			probs = torch.softmax(masked_logits, dim = -1)

			if self.training:
				action = torch.multinomial(probs, num_samples = 1).squeeze(-1)
			else:
				action = torch.argmax(probs, dim = -1)

			log_probs = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1))
			actions = F.one_hot(action, num_classes = logits.shape[-1]).float()
		else:
			return torch.zeros((len(batch), self.network[-1].out_features)), 0.0

		return actions, log_probs

	def calculate_action_mask(self, state: dict[str, np.ndarray]) -> np.ndarray:
		legal_cards = set()

		game_state = [game_state.name for game_state in GameStates if game_state.value == np.argmax(state['game_state']).item()][0]

		if game_state in {'SHOW_3', 'SHOW_ALL'}:
			exposed = np.any(state['exposed'] > 1, axis = 0)
			legal_cards.update(card for card in [11, 13, 36, 48] if not exposed[card])

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
			trick = np.where(state['leader_history'] != -1)[0][-1]
			leader = state['leader_history'][trick]
			trick_suit = state['play_history'][trick][leader] // 13

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

		out_dim = self.network[-1].out_features

		hand = np.where(state['hand'] == 1)[0]
		masked_hand = np.where(np.isin(hand, list(legal_cards)), 1, 0)
		padded_hand = np.pad(masked_hand, pad_width = (0, out_dim - len(masked_hand)), mode = 'constant', constant_values = 0)

		return padded_hand

	def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		# states.shape:		(B, dim(obs))
		# actions.shape:	(B, dim(act))
		# masks.shape:		(B, dim(act))

		logits = self.network(states)

		eps = 1e-9

		if self.module_type == 'SHOW':
			num_valid = masks.sum(dim = -1).clamp(min = 1)

			masked_logits = logits.masked_fill(masks == 0, torch.tensor(-torch.inf))
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
			masked_logits = logits.masked_fill(masks == 0, torch.tensor(-torch.inf))
			probs = torch.softmax(masked_logits, dim = -1)
			probs = torch.clamp(probs, eps, 1 - eps)

			action_indices = actions.argmax(dim = -1)

			log_probs = torch.log(probs.gather(-1, action_indices.unsqueeze(-1)).squeeze(-1) + eps)

			valid_probs = probs * masks
			valid_probs = valid_probs / valid_probs.sum(dim = -1, keepdim = True).clamp(min = eps)
			entropy = -(valid_probs * torch.log(valid_probs + eps)).sum(dim = -1).mean()
		else:
			return torch.zeros(states.shape[0]), torch.tensor(0.0)

		# log_probs.shape:	(B,)
		# entropy.shape:	(,)
		return log_probs, entropy

	@staticmethod
	def serialize_state(state: dict[str, np.ndarray]) -> np.ndarray:
		order = ['game_state', 'hand', 'scores', 'current_trick', 'collected_cards', 'exposed', 'play_history', 'leader_history']

		return np.concatenate([state[k].flatten() for k in order], axis = 0)

	@staticmethod
	def get_module_type_from_game_state(game_state: str) -> str:
		if game_state.startswith('SHOW'):
			return 'SHOW'
		elif game_state.startswith('PLAY'):
			return 'PLAY'
		else:
			raise ValueError(f'Unknown game_state [{game_state}]')

class CriticNN(torch.nn.Module):
	def __init__(self, dims: tuple[int, ...]) -> None:
		super().__init__()

		self.network = torch.nn.Sequential(
			*[e for dim in dims for e in (torch.nn.LazyLinear(dim), torch.nn.ReLU())],
			torch.nn.LazyLinear(1)
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		# inputs.shape: (B, dim(obs))
		# return.shape: (B,)
		return self.network(inputs).squeeze(-1)


class MultiAgentEnv:

	class ModelModes(enum.Enum):
		train =			0
		infer =			1
		eval =			2

	def __init__(self,
		args: argparse.Namespace | None = None,
		mode: MultiAgentEnv.ModelModes | None = None
	) -> None:

		if mode is None:
			mode = MultiAgentEnv.ModelModes.train

		self.args = args if args is not None else argparse.Namespace()

		self.num_agents: int =														GAME_SETTINGS['NUM_PLAYERS']
		self.ws_list: list[websockets.asyncio.client.ClientConnection | None] =		[None			for _ in range(self.num_agents)]

		self.mode: MultiAgentEnv.ModelModes =										mode
		self.is_waiting_for_spectator: bool =										False
		self.batch_num: int =														0
		self.save_checkpoint_frequency: int =										10

		self._is_processing_event: list[bool] =										[False			for _ in range(self.num_agents)]
		self._event_queue: list[list[tuple[str, float]]] =							[list()			for _ in range(self.num_agents)]	# (message, timestamp)

		self.eval_game_complete_event: asyncio.Event =								asyncio.Event()

		self.training_history: dict[str, list[int | float]] = {
			'batch': [],
			'actor_loss': [],
			'critic_loss': [],
			'mean_kl_divergence': [],
			'mean_return': [],
			'max_episode_reward': [],
			'min_episode_reward': [],
			'mean_episode_reward': []
		}

		self.reset()

		self.actor = self.create_actor()
		self.critic = self.create_critic()

		self.elo_rating_system = EloRatingSystem(k_factor = self.args.k_factor)
		self.eval_games_per_match: int =			5
		self.eval_total_matches: int =				0
		self.eval_total_games: int =				0

		self.action_delay: float =					0.01

		self._init_hyperparameters()

	def _init_hyperparameters(self) -> None:
		self.timesteps_per_batch: int =				2048
		self.max_timesteps_per_episode: int =		200

		# PPO Parameters
		self.gamma: float =							0.99
		self.gae_lambda: float =					0.95
		self.clip_epsilon: float =					0.2
		self.n_updates_per_batch: int =				10
		self.mini_batch_size: int =					256
		self.actor_lr: float =						1e-4
		self.critic_lr: float =						3e-4
		self.entropy_coef: float =					0.01
		self.max_grad_norm: float =					0.5

	def reset(self) -> None:
		self.is_rollout: bool =											False
		self.is_training: bool =										False
		self.is_evaluating: bool =										False

		self._latest_observation: list[dict | None] =					[None					for _ in range(self.num_agents)]
		self._latest_state: list[dict[str, np.ndarray] | None] =		[None					for _ in range(self.num_agents)]
		self._latest_actions: list[list[dict[str, Any]]] =				[list()					for _ in range(self.num_agents)]
		self._latest_reward: list[float | None] =						[None					for _ in range(self.num_agents)]

		self._play_history: list[np.ndarray[np.int8]] =					[np.full((13, NUM_PLAYERS), -1, dtype = np.int8)	for _ in range(self.num_agents)]
		self._leader_history: list[np.ndarray[np.int8]] =				[np.full((13,), -1, dtype = np.int8)				for _ in range(self.num_agents)]

		self._batch_ts: list[int] =										[-1						for _ in range(self.num_agents)]
		self._batch_states: list[torch.Tensor] =						[torch.empty(0)			for _ in range(self.num_agents)]	# (B, dim(obs))
		self._batch_actions: list[torch.Tensor] =						[torch.empty(0)			for _ in range(self.num_agents)]	# (B, dim(act))
		self._batch_action_masks: list[torch.Tensor] =					[torch.empty(0)			for _ in range(self.num_agents)]	# (B, dim(act))
		self._batch_log_probs: list[torch.Tensor] =						[torch.empty(0)			for _ in range(self.num_agents)]	# (B,)
		self._batch_rewards: list[list[np.ndarray]] =					[list()					for _ in range(self.num_agents)]	# (E, t_per_E)
		self._batch_lens: list[np.ndarray | None] =						[np.empty(0)			for _ in range(self.num_agents)]	# (E,)

		self._episode_ts: list[int] =									[-1						for _ in range(self.num_agents)]
		self._episode_rewards: list[np.ndarray] =						[np.empty(0)			for _ in range(self.num_agents)]	# (t_per_E,)

		self.eval_actors: list[torch.nn.ModuleDict | None] =			[None					for _ in range(self.num_agents)]
		self.eval_model_paths: list[str | None] =						[None					for _ in range(self.num_agents)]
		self.eval_scores: list[float | None] =							[None					for _ in range(self.num_agents)]
		self.eval_last_frame: int =										-1

		self._rollout_progressbar: tqdm | None =						None

	def reset_game_state(self, ws_idx = int) -> None:
		num_tricks = math.ceil(GAME_SETTINGS['NUM_CARDS'] / GAME_SETTINGS['NUM_PLAYERS'])

		self._play_history[ws_idx] =			np.full((num_tricks, NUM_PLAYERS), -1, dtype = np.int8)
		self._leader_history[ws_idx] =			np.full((num_tricks,), -1, dtype = np.int8)
		self._latest_observation[ws_idx] =		None
		self._latest_state[ws_idx] =			None
		self._latest_actions[ws_idx] =			list()

	def set_mode(self, mode: MultiAgentEnv.ModelModes) -> None:
		self.mode = mode

	def create_actor(self) -> torch.nn.ModuleDict:
		return torch.nn.ModuleDict({
			'SHOW': ActorNN('SHOW', (256, 256, 128, 13)),
			'PLAY': ActorNN('PLAY', (256, 256, 128, 13))
		})

	def create_critic(self) -> torch.nn.Module:
		return CriticNN((256, 256, 128))

	def save_checkpoint(self) -> None:
		checkpoint = {
			'batch_num': self.batch_num,
			'actor_state_dict': self.actor.state_dict(),
			'critic_state_dict': self.critic.state_dict(),
			'training_history': self.training_history
		}
		path = os.path.join(self.args.save_dir, f'gong_zhu_player_{self.batch_num}.pt')

		fd, tmp_path = tempfile.mkstemp(dir = self.args.save_dir, suffix = '.pt.tmp')

		try:
			os.close(fd)
			torch.save(checkpoint, tmp_path)
			os.replace(tmp_path, path)
			print(f'Saved checkpoint at [{os.path.abspath(path)}]')
		except BaseException:
			if os.path.exists(tmp_path):
				os.remove(tmp_path)

	def load_checkpoint(self, path: str) -> None:
		checkpoint = torch.load(path, weights_only = False)

		self.batch_num = checkpoint['batch_num']
		self.actor.load_state_dict(checkpoint['actor_state_dict'])
		self.critic.load_state_dict(checkpoint['critic_state_dict'])
		self.training_history = checkpoint['training_history']
		print(f'Loaded checkpoint from [{os.path.abspath(path)}] (at batch {self.batch_num})')

	def load_actor_from_checkpoint(self, model_path: str) -> torch.nn.ModuleDict:
		actor = self.create_actor()

		checkpoint = torch.load(model_path, weights_only = False)
		input_dim = checkpoint['actor_state_dict']['SHOW.network.0.weight'].shape[1]

		# Init Lazy Modules
		dummy_input = torch.zeros(1, input_dim)
		dummy_mask = torch.ones(1, 13)

		with torch.no_grad():
			actor['SHOW'](dummy_input, dummy_mask)
			actor['PLAY'](dummy_input, dummy_mask)

		actor.load_state_dict(checkpoint['actor_state_dict'])
		actor.eval()

		return actor

	async def connect(self, url: str) -> None:
		global console_listeners

		if self.mode == MultiAgentEnv.ModelModes.train:
			async def connect_ws(ws_idx: int) -> None:
				async with connect(url) as ws:
					self.ws_list[ws_idx] = ws

					console_listeners.create_console(ws_idx, f'agent_{ws_idx}')

					await ws.send(json.dumps({'tag': 'requestSessionID', 'timestamp': int(time.time() * 1000)}))
					await self._listen(ws_idx)

			tasks = [asyncio.create_task(connect_ws(i)) for i in range(self.num_agents)]
			await asyncio.gather(*tasks)

		elif self.mode == MultiAgentEnv.ModelModes.infer:
			self.actor.eval()

			# Only One Agent
			ws_idx = 0
			async with connect(url) as ws:
				self.ws_list[ws_idx] = ws

				console_listeners.create_console(ws_idx, f'agent_{ws_idx}')

				await ws.send(json.dumps({'tag': 'requestSessionID', 'timestamp': int(time.time() * 1000)}))
				await self._listen(ws_idx)

		elif self.mode == MultiAgentEnv.ModelModes.eval:
			async def connect_ws(ws_idx: int) -> None:
				async with connect(url) as ws:
					self.ws_list[ws_idx] = ws

					console_listeners.create_console(ws_idx, f'agent_{ws_idx}')

					await ws.send(json.dumps({'tag': 'requestSessionID', 'timestamp': int(time.time() * 1000)}))
					await self._listen(ws_idx)

			async def launch_eval() -> None:
				while not all(
					self.ws_list[i] is not None and hasattr(self.ws_list[i], 'connected') and self.ws_list[i].connected
					for i in range(self.num_agents)
				):
					await asyncio.sleep(self.action_delay)

				if hasattr(self.args, 'spectate') and self.args.spectate:
					self.is_waiting_for_spectator = True

					loop = asyncio.get_running_loop()
					await loop.run_in_executor(None, lambda : input('Join as spectator now, [Enter] to continue...'))

					self.is_waiting_for_spectator = False

				await asyncio.sleep(0.5)
				await self.eval()

			tasks = [asyncio.create_task(connect_ws(i)) for i in range(self.num_agents)]
			eval_task = asyncio.create_task(launch_eval())
			await asyncio.gather(*tasks, eval_task)

	async def unlock_processing_event(self, ws_idx: int) -> None:
		if len(self._event_queue[ws_idx]):
			message, timestamp = self._event_queue[ws_idx].pop(0)

			asyncio.create_task(self._handle_message(ws_idx, message))

		else:
			self._is_processing_event[ws_idx] = False

	async def _listen(self, ws_idx: int) -> None:
		ws = self.ws_list[ws_idx]
		async for message in ws:
			if self._is_processing_event[ws_idx]:
				self._event_queue[ws_idx].append((message, time.time()))
				continue
			else:
				await self._handle_message(ws_idx, message)

	async def _handle_message(self, ws_idx: int, message: str) -> None:
		global console_listeners
		ws = self.ws_list[ws_idx]

		self._is_processing_event[ws_idx] = True

		msg = json.loads(message, object_hook = json_parser)
		# print(ws_idx, msg)

		if msg['tag'] == 'receiveSessionID':
			ws.sessionID =  msg['data']['sessionID']
			await ws.send(json.dumps({
				'tag': 'requestUsername',
				'data': 'bot',
				'timestamp': int(time.time() * 1000)
			}))

		elif msg['tag'] == 'receiveUsername':
			ws.username = msg['data']

			if self.mode == MultiAgentEnv.ModelModes.train:
				if ws_idx == 0:
					await ws.send(json.dumps({
						'tag': 'createLobby',
						'data': {
							'name': 'training',
							'time': int(time.time() * 1000),
							'creator': ws.username,
							'host': ws.username
						},
						'timestamp': int(time.time() * 1000)
					}))

			elif self.mode == MultiAgentEnv.ModelModes.infer:
				if ws_idx == 0:
					await ws.send(json.dumps({
						'tag': 'getLobbies',
						'timestamp': int(time.time() * 1000)
					}))

			elif self.mode == MultiAgentEnv.ModelModes.eval:
				if ws_idx == 0:
					await ws.send(json.dumps({
						'tag': 'createLobby',
						'data': {
							'name': 'evaluation',
							'time': int(time.time() * 1000),
							'creator': ws.username,
							'host': ws.username
						},
						'timestamp': int(time.time() * 1000)
					}))

		elif msg['tag'] == 'createdLobby':
			if self.mode in (MultiAgentEnv.ModelModes.train, MultiAgentEnv.ModelModes.eval):
				await ws.send(json.dumps({
					'tag': 'joinLobby',
					'data': msg['data'],
					'timestamp': int(time.time() * 1000)
				}))

		elif msg['tag'] == 'showLobby':
			self.reset_game_state(ws_idx)

			if self.mode == MultiAgentEnv.ModelModes.train:
				if not hasattr(ws, 'connected'):
					ws.connected = True

					if ws_idx == 0:
						settings = copy.deepcopy(msg['data']['gameData']['settings'])
						settings['spectatorPolicy'] = 'constant'
						settings['expose3'] = True
						settings['zhuYangManJuan'] = True
						await ws.send(json.dumps({
							'tag': 'updateLobbySettings',
							'data': {'settings': settings},
							'timestamp': int(time.time() * 1000)
						}))

						updateEnvironmentSettings(msg)

						self._play_history =	[np.full((math.ceil(GAME_SETTINGS['NUM_CARDS'] / GAME_SETTINGS['NUM_PLAYERS']), GAME_SETTINGS['NUM_PLAYERS']), -1, dtype = np.int8) for _ in range(self.num_agents)]
						self._leader_history =	[np.full((math.ceil(GAME_SETTINGS['NUM_CARDS'] / GAME_SETTINGS['NUM_PLAYERS']),), -1, dtype = np.int8) for _ in range(self.num_agents)]

						for ws_i in self.ws_list[1:]:
							await ws_i.send(json.dumps({
								'tag': 'getLobbies',
								'timestamp': int(time.time() * 1000)
							}))

						await self.unlock_processing_event(ws_idx)

						while not all(hasattr(self.ws_list[i], 'connected') and self.ws_list[i].connected for i in range(self.num_agents)):
							await asyncio.sleep(self.action_delay)

						if hasattr(self.args, 'spectate') and self.args.spectate:
							self.is_waiting_for_spectator = True

							loop = asyncio.get_running_loop()
							await loop.run_in_executor(None, lambda : input('Join as spectator now, [Enter] to continue...'))

							self.is_waiting_for_spectator = False

						await self.train()

				else:
					if ws_idx == 0:
						if len(msg['data']['connected']) == GAME_SETTINGS['NUM_PLAYERS']:
							if self.is_rollout:
								await ws.send(json.dumps({
									'tag': 'startGame',
									'timestamp': int(time.time() * 1000)
								}))

							elif self.is_training:
								pass

							elif self.is_waiting_for_spectator:
								pass

							else:
								self.is_waiting_for_spectator = True

								await self.unlock_processing_event(ws_idx)

								loop = asyncio.get_running_loop()
								await loop.run_in_executor(None, lambda : input('Join as spectator now, [Enter] to continue...'))

								self.is_waiting_for_spectator = False
								await self.train()

				await self.unlock_processing_event(ws_idx)

			elif self.mode == MultiAgentEnv.ModelModes.infer:
				if not hasattr(ws, 'connected'):
					ws.connected = True

					updateEnvironmentSettings(msg)

			elif self.mode == MultiAgentEnv.ModelModes.eval:
				if not hasattr(ws, 'connected'):
					ws.connected = True

					if ws_idx == 0:
						settings = copy.deepcopy(msg['data']['gameData']['settings'])
						settings['spectatorPolicy'] = 'constant'
						settings['expose3'] = True
						settings['zhuYangManJuan'] = True
						await ws.send(json.dumps({
							'tag': 'updateLobbySettings',
							'data': {'settings': settings},
							'timestamp': int(time.time() * 1000)
						}))

						updateEnvironmentSettings(msg)

						self._play_history =	[np.full((math.ceil(GAME_SETTINGS['NUM_CARDS'] / GAME_SETTINGS['NUM_PLAYERS']), GAME_SETTINGS['NUM_PLAYERS']), -1, dtype = np.int8) for _ in range(self.num_agents)]
						self._leader_history =	[np.full((math.ceil(GAME_SETTINGS['NUM_CARDS'] / GAME_SETTINGS['NUM_PLAYERS']),), -1, dtype = np.int8) for _ in range(self.num_agents)]

						for ws_i in self.ws_list[1:]:
							await ws_i.send(json.dumps({
								'tag': 'getLobbies',
								'timestamp': int(time.time() * 1000)
							}))

				await self.unlock_processing_event(ws_idx)

		elif msg['tag'] == 'updateLobbies':
			if self.mode in (MultiAgentEnv.ModelModes.train, MultiAgentEnv.ModelModes.eval):
				if not hasattr(ws, 'connected'):
					servers = msg['data']
					for server in servers:
						if (
							hasattr(self.ws_list[0], 'connected') and self.ws_list[0].connected and
							hasattr(self.ws_list[0], 'username') and server['host'] == self.ws_list[0].username
						):
							await ws.send(json.dumps({
								'tag': 'joinLobby',
								'data': server,
								'timestamp': int(time.time() * 1000)
							}))

							await self.unlock_processing_event(ws_idx)
							return

			elif self.mode == MultiAgentEnv.ModelModes.infer:
				if not hasattr(ws, 'connected'):
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
			if self.mode == MultiAgentEnv.ModelModes.train:
				# No Longer Collecting Data
				if not self.is_rollout:
					await self.unlock_processing_event(ws_idx)
					return

				updateEnvironmentSettings(msg)

				turn_idx = GAME_SETTINGS['turn_order'].index(ws.username)
				current_frame = msg['data']['gameData']['currentFrame']

				# Update Internals From Message
				prev_state = copy.deepcopy(self._latest_state[ws_idx])
				prev_observation = copy.deepcopy(self._latest_observation[ws_idx])
				self.update_latest_from_gui_observation(msg['data']['gameData'], ws_idx, turn_idx)

				self.cleanup_acknowledged_actions(ws_idx, current_frame)
				found_matching_command = self.acknowledged_previous_command(ws_idx, current_frame)

				# Nothing Happened -> Return
				if json.dumps(prev_observation, sort_keys = True, separators = (',', ':')) == json.dumps(self._latest_observation[ws_idx], sort_keys = True, separators = (',', ':')):
					await self.unlock_processing_event(ws_idx)
					return

				# Reward Calculation
				if prev_state is not None and len(self._episode_rewards[ws_idx]) > 0:
					reward = MultiAgentEnv.get_reward_from_state_transition(prev_state, self._latest_state[ws_idx], turn_idx)

					self._episode_rewards[ws_idx][-1] += reward

					console_listeners.broadcast_message(json.dumps({
						'tag': 'receiveCommand',
						'data': {
							'id': console_listeners.idx_to_name[ws_idx],
							'msg': ['=== reward ===\n' + f'Transition Reward [{reward}]\nCumulative Reward [{self._episode_rewards[ws_idx][-1]}]'],
							'status': 1
						}
					}))

				# Episode Completion + Reset
				game_state = self._latest_observation[ws_idx]['gameState']
				done = game_state in {'LEADERBOARD', 'SCORE'}

				if (done or (self._episode_ts[ws_idx] >= self.max_timesteps_per_episode)) and self._episode_ts[ws_idx] > 0:
					self.end_episode(ws_idx)

				if done:
					if ws_idx == 0 and self.is_rollout:
						self._play_history =	[np.full((13, NUM_PLAYERS), -1, dtype = np.int8)	for _ in range(self.num_agents)]
						self._leader_history =	[np.full((13,), -1, dtype = np.int8)				for _ in range(self.num_agents)]

						total_timesteps = sum(self._batch_ts)
						target_timesteps = self.timesteps_per_batch * self.num_agents

						total_episodes = sum(len(rewards) for rewards in self._batch_rewards)
						ongoing_episodes = sum(1 for ts in self._episode_ts if ts > 0)

						agent_stats = ' | '.join([
						f'A{i}:{self._batch_ts[i]}ts/{len(self._batch_rewards[i])}ep'
							for i in range(self.num_agents)
						])

						status_msg = (
							f'Progress: {total_timesteps}/{target_timesteps} | '
							f'{total_episodes}+{ongoing_episodes} episodes'
						)
						detail_msg = f'{agent_stats}'

						await ws.send(json.dumps({
							'tag': 'sendChat',
							'data': status_msg,
							'timestamp': int(time.time() * 1000),
						}));

						await ws.send(json.dumps({
							'tag': 'sendChat',
							'data': detail_msg,
							'timestamp': int(time.time() * 1000),
						}));

						self._rollout_progressbar.n = sum(self._batch_ts)
						self._rollout_progressbar.refresh()

						await ws.send(json.dumps({
							'tag': 'sendCommand',
							'data': 'DEAL',
							'timestamp': int(time.time() * 1000),
							'currentFrame': {'$bigint': str(self._latest_observation[ws_idx]['currentFrame'] if self._latest_observation[ws_idx] is not None else -1)}
						}))
					await self.unlock_processing_event(ws_idx)
					return

				console_listeners.broadcast_message(json.dumps({
					'tag': 'receiveCommand',
					'data': {
						'id': console_listeners.idx_to_name[ws_idx],
						'msg': [f'=== self._latest_observation[ws_idx] ===\n' + json.dumps(self._latest_observation[ws_idx])],
						'status': 0
					}
				}))
				console_listeners.broadcast_message(json.dumps({
					'tag': 'receiveCommand',
					'data': {
						'id': console_listeners.idx_to_name[ws_idx],
						'msg': [f'gameState: {self._latest_observation[ws_idx]["gameState"]} | needToAct: {self._latest_observation[ws_idx]["needToAct"][turn_idx]}'],
						'status': 1
					}
				}))

				# Needs to Act -> Make Action
				if self._latest_observation[ws_idx]['needToAct'][turn_idx] == 1:
					console_listeners.broadcast_message(json.dumps({
						'tag': 'receiveCommand',
						'data': {
							'id': console_listeners.idx_to_name[ws_idx],
							'msg': [f'NEED TO ACT -- batch {self._batch_ts[ws_idx]}/{self.timesteps_per_batch} | episode {self._episode_ts[ws_idx]}/{self.max_timesteps_per_episode}'],
							'status': 0
						}
					}))
					console_listeners.broadcast_message(json.dumps({
						'tag': 'receiveCommand',
						'data': {
							'id': console_listeners.idx_to_name[ws_idx],
							'msg': [f'=== self._latest_state[ws_idx] ===\n' + json.dumps({k: v.tolist() if type(v) == np.ndarray else v for k, v in self._latest_state[ws_idx].items()})],
							'status': 1
						}
					}))

					if self._batch_ts[ws_idx] >= 0 and self._batch_ts[ws_idx] < self.timesteps_per_batch:

						if self._episode_ts[ws_idx] < 0:
							self._episode_rewards[ws_idx] = np.empty(0)
							self._episode_ts[ws_idx] = 0

						elif self._episode_ts[ws_idx] >= self.max_timesteps_per_episode:
							self._episode_rewards[ws_idx] = np.empty(0)
							self._episode_ts[ws_idx] = 0

							# Episode Complete
							await ws.send(json.dumps({
								'tag': 'sendCommand',
								'data': 'EXIT',
								'timestamp': int(time.time() * 1000),
								'currentFrame': {'$bigint': str(self._latest_observation[ws_idx]['currentFrame'] if self._latest_observation[ws_idx] is not None else -1)}
							}))
							await self.unlock_processing_event(ws_idx)
							return

						# Skip Action if Waiting for ACK
						if self.is_waiting_for_ack(ws_idx, current_frame):
							await self.unlock_processing_event(ws_idx)
							return

						# Batch Full
						if self._batch_ts[ws_idx] + 1 >= self.timesteps_per_batch:
							self.is_rollout = False
							self.is_training = True
							for _ws_idx in range(self.num_agents):
								self.end_episode(_ws_idx)

							await ws.send(json.dumps({
								'tag': 'sendCommand',
								'data': 'EXIT',
								'timestamp': int(time.time() * 1000),
								'currentFrame': {'$bigint': str(self._latest_observation[ws_idx]['currentFrame'] if self._latest_observation[ws_idx] is not None else -1)}
							}))

							self._rollout_progressbar.close()
							asyncio.create_task(self.train_post_rollout())

							await self.unlock_processing_event(ws_idx)
							return

						with torch.no_grad():
							actor_module_type = ActorNN.get_module_type_from_game_state(self._latest_observation[ws_idx]['gameState'])

							if actor_module_type == '':
								await self.unlock_processing_event(ws_idx)
								return

							actions, log_probs, inputs, mask = self.get_action(ws_idx, actor_module_type)

							console_listeners.broadcast_message(json.dumps({
								'tag': 'receiveCommand',
								'data': {
									'id': console_listeners.idx_to_name[ws_idx],
									'msg': [f'=== actions | log_probs ({self._latest_observation[ws_idx]["gameState"]}) ===\n' + json.dumps(actions.tolist()) + '\n' + json.dumps(log_probs.tolist())],
									'status': 1
								}
							}))

							self._batch_states[ws_idx] = torch.concatenate((self._batch_states[ws_idx], inputs), dim = 0)
							self._batch_actions[ws_idx] = torch.concatenate((self._batch_actions[ws_idx], actions), dim = 0)
							self._batch_action_masks[ws_idx] = torch.concatenate((self._batch_action_masks[ws_idx], mask), dim = 0)
							self._batch_log_probs[ws_idx] = torch.concatenate((self._batch_log_probs[ws_idx], log_probs), dim = 0)
							self._episode_rewards[ws_idx] = np.concatenate((self._episode_rewards[ws_idx], np.array([0.0])), axis = 0)

						self._batch_ts[ws_idx] += 1
						self._episode_ts[ws_idx] += 1

						await asyncio.sleep(self.action_delay)
						await self.act(actions[0], ws_idx, turn_idx)

			elif self.mode == MultiAgentEnv.ModelModes.infer:
				updateEnvironmentSettings(msg)

				# Spectator or Not In Game
				if ws.username not in GAME_SETTINGS['turn_order']:
					await self.unlock_processing_event(ws_idx)
					return

				turn_idx = GAME_SETTINGS['turn_order'].index(ws.username)
				current_frame = msg['data']['gameData']['currentFrame']

				# Update Internals From Message
				prev_observation = copy.deepcopy(self._latest_observation[ws_idx])
				self.update_latest_from_gui_observation(msg['data']['gameData'], ws_idx, turn_idx)

				self.cleanup_acknowledged_actions(ws_idx, current_frame)
				found_matching_command = self.acknowledged_previous_command(ws_idx, current_frame)

				# Nothing Happened -> Return
				if json.dumps(prev_observation, sort_keys = True, separators = (',', ':')) == json.dumps(self._latest_observation[ws_idx], sort_keys = True, separators = (',', ':')):
					await self.unlock_processing_event(ws_idx)
					return

				# Game Ended
				game_state = self._latest_observation[ws_idx]['gameState']
				done = game_state in {'LEADERBOARD', 'SCORE'}

				if done:
					self.reset_game_state(ws_idx)
					await self.unlock_processing_event(ws_idx)
					return

				# Needs to Act -> Make Action
				if self._latest_observation[ws_idx]['needToAct'][turn_idx] == 1:

					# Skip Action if Waiting for ACK
					if self.is_waiting_for_ack(ws_idx, current_frame):
						await self.unlock_processing_event(ws_idx)
						return

					actor_module_type = ActorNN.get_module_type_from_game_state(self._latest_observation[ws_idx]['gameState'])

					if actor_module_type == '':
						await self.unlock_processing_event(ws_idx)
						return

					actions, log_probs, inputs, mask = self.get_action(ws_idx, actor_module_type)
					await self.act(actions[0], ws_idx, turn_idx)

			elif self.mode == MultiAgentEnv.ModelModes.eval:
				if self.eval_game_complete_event.is_set():
					await self.unlock_processing_event(ws_idx)
					return

				updateEnvironmentSettings(msg)

				if ws.username not in GAME_SETTINGS['turn_order']:
					await self.unlock_processing_event(ws_idx)
					return

				turn_idx = GAME_SETTINGS['turn_order'].index(ws.username)
				current_frame = msg['data']['gameData']['currentFrame']

				# Update Internals From Message
				prev_observation = copy.deepcopy(self._latest_observation[ws_idx])
				self.update_latest_from_gui_observation(msg['data']['gameData'], ws_idx, turn_idx)

				self.cleanup_acknowledged_actions(ws_idx, current_frame)
				found_matching_command = self.acknowledged_previous_command(ws_idx, current_frame)

				# Nothing Happened -> Return
				if json.dumps(prev_observation, sort_keys=True, separators=(',', ':')) == json.dumps(self._latest_observation[ws_idx], sort_keys=True, separators=(',', ':')):
					await self.unlock_processing_event(ws_idx)
					return

				# Round Completion + Reset
				game_state = self._latest_observation[ws_idx]['gameState']
				done = game_state in {'LEADERBOARD', 'SCORE'}

				if done:
					if ws_idx == 0:
						if not self.is_evaluating:
							await ws.send(json.dumps({
								'tag': 'sendCommand',
								'data': 'DEAL',
								'timestamp': int(time.time() * 1000),
								'currentFrame': {'$bigint': str(current_frame)}
							}))

						else:
							self.eval_scores = self._latest_state[ws_idx]['scores'].tolist()
							self.eval_last_frame = self._latest_observation[ws_idx]['currentFrame']
							self.eval_game_complete_event.set()

					await self.unlock_processing_event(ws_idx)
					return

				self.is_evaluating = True

				# Need to Act -> Make Action
				if self._latest_observation[ws_idx]['needToAct'][turn_idx] == 1:

					# Skip Action if Waiting for ACK
					if self.is_waiting_for_ack(ws_idx, current_frame):
						await self.unlock_processing_event(ws_idx)
						return

					with torch.no_grad():
						actor_module_type = ActorNN.get_module_type_from_game_state(self._latest_observation[ws_idx]['gameState'])

						if actor_module_type == '':
							await self.unlock_processing_event(ws_idx)
							return

						actions, log_probs, inputs, mask = self.get_action(ws_idx, actor_module_type, actor = self.eval_actors[ws_idx])

						await asyncio.sleep(self.action_delay)
						await self.act(actions[0], ws_idx, turn_idx)

		elif msg['tag'] == 'receiveCommand':
			console_listeners.broadcast_message(json.dumps({
				'tag': 'receiveCommand',
				'data': {
					'id': console_listeners.idx_to_name[ws_idx],
					'msg': [f'=== receiveCommand ===\n{msg["data"]}'],
					'status': 1
				}
			}))
			self.update_latest_from_console_observation(msg['data'], ws_idx)

		elif msg['tag'] == 'commandNACK':
			console_listeners.broadcast_message(json.dumps({
				'tag': 'receiveCommand',
				'data': {
					'id': console_listeners.idx_to_name[ws_idx],
					'msg': [f'=== commandNACK ===\n{msg["data"]}'],
					'status': 0
				}
			}))
			try:
				found_idx = next(i for i, e in enumerate(self._latest_actions[ws_idx]) if e['command'] == msg['data']['command'] and e['oldFrame'] == msg['data']['oldFrame'])
				self._latest_actions[ws_idx].pop(found_idx)

				if self.mode == MultiAgentEnv.ModelModes.train:
					# Rollback rejected command
					if self._batch_states[ws_idx].shape[0] > 0:
						self._batch_states[ws_idx] = self._batch_states[ws_idx][:-1]
						self._batch_actions[ws_idx] = self._batch_actions[ws_idx][:-1]
						self._batch_action_masks[ws_idx] = self._batch_action_masks[ws_idx][:-1]
						self._batch_log_probs[ws_idx] = self._batch_log_probs[ws_idx][:-1]

						if self._episode_rewards[ws_idx].shape[0] > 0:
							self._episode_rewards[ws_idx] = self._episode_rewards[ws_idx][:-1]

						self._batch_ts[ws_idx] -= 1
						self._episode_ts[ws_idx] -= 1

			except StopIteration:
				pass

		elif msg['tag'] == 'commandACK':
			try:
				found_idx = next(i for i, e in enumerate(self._latest_actions[ws_idx]) if e['command'] == msg['data']['command'] and e['oldFrame'] == msg['data']['oldFrame'])
				self._latest_actions[ws_idx][found_idx]['newFrame'] = msg['data']['newFrame']
				self._latest_actions[ws_idx][found_idx]['ack'] = 1

			except StopIteration:
				pass

			console_listeners.broadcast_message(json.dumps({
				'tag': 'receiveCommand',
				'data': {
					'id': console_listeners.idx_to_name[ws_idx],
					'msg': [f'=== commandACK ===\n{msg["data"]}', json.dumps(self._latest_actions[ws_idx])],
					'status': 1
				}
			}))

		await self.unlock_processing_event(ws_idx)

	def update_latest_from_gui_observation(self, observation: dict, ws_idx: int, turn_idx: int) -> None:
		trick: int = round(len(observation['stacks'][0]) / len(observation['turnOrder']))
		partial_state: dict[str, np.ndarray] = MultiAgentEnv.encode_observation(observation, turn_idx)

		if observation['gameState'].startswith('PLAY_'):
			self._leader_history[ws_idx][trick] = observation['turnFirstIdx']

			for i, hand in enumerate(observation['hands']):
				if len(hand[3]) > 0:
					self._play_history[ws_idx][trick][i] = hand[3][0]

		state = partial_state | {'play_history': self._play_history[ws_idx], 'leader_history': self._leader_history[ws_idx]}

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

				trick: int = round(len(self._latest_observation[ws_idx]['stacks'][0]) / len(self._latest_observation[ws_idx]['turnOrder']))

				if trick >= self._play_history[ws_idx].shape[0]:
					continue

				player_idx: int = self._latest_observation[ws_idx]['turnOrder'].index(username)
				self._play_history[ws_idx][trick][player_idx] = card_int

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
		actor: torch.nn.ModuleDict | None = None
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

		if actor is None:
			actor = self.actor

		inputs = torch.tensor(np.array([ActorNN.serialize_state(self._latest_state[ws_idx])]))
		mask = torch.tensor(np.array([actor[actor_module_type].calculate_action_mask(self._latest_state[ws_idx])]))

		actions, log_probs = actor[actor_module_type](inputs, mask)

		return actions, log_probs, inputs, mask

	async def act(self, action: torch.Tensor, ws_idx: int, turn_idx: int) -> None:
		commands = []
		if self._latest_observation[ws_idx]['gameState'].startswith('SHOW'):
			if torch.sum(action).item() != 0:
				cards_to_play = torch.nonzero(action, as_tuple = True)[0]
				hand = np.where(self._latest_state[ws_idx]['hand'] == 1)[0]
				cards_to_play_idxs = np.where(np.isin(self._latest_observation[ws_idx]['hands'][turn_idx][0], hand[cards_to_play].tolist()))[0].tolist()

				args = ' '.join([str(e) for e in cards_to_play_idxs])
				commands.append('PLAY ' + args)
			commands.append('PASS')
		elif self._latest_observation[ws_idx]['gameState'].startswith('PLAY'):
			cards_to_play = torch.nonzero(action, as_tuple = True)[0]
			hand = np.where(self._latest_state[ws_idx]['hand'] == 1)[0]
			cards_to_play_idxs = np.where(np.isin(self._latest_observation[ws_idx]['hands'][turn_idx][0], hand[cards_to_play].tolist()))[0].tolist()

			args = ' '.join([str(e) for e in cards_to_play_idxs])
			commands.append('PLAY ' + args)

		command_str = ''
		for i, command in enumerate(commands):
			if i != 0:
				command_str += '\n'
			command_str += f'=> [{command}]'

		console_listeners.broadcast_message(json.dumps({
			'tag': 'receiveCommand',
			'data': {
				'id': console_listeners.idx_to_name[ws_idx],
				'msg': [f' === PLAYING WITH OBSERVATION: ===\n' + json.dumps(self._latest_observation[ws_idx]), command_str],
				'status': 1
			}
		}))

		ws = self.ws_list[ws_idx]
		for command in commands:
			self._latest_actions[ws_idx].append({
				'command': command,
				'oldFrame': self._latest_observation[ws_idx]['currentFrame'] if self._latest_observation[ws_idx] is not None else -1,
				'newFrame': None,
				'ack': -1
			})
			await ws.send(json.dumps({
				'tag': 'sendCommand',
				'data': command,
				'timestamp': int(time.time() * 1000),
				'currentFrame': {'$bigint': str(self._latest_observation[ws_idx]['currentFrame'] if self._latest_observation[ws_idx] is not None else -1)}
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

		current_trick: np.ndarray = np.array([(hand[3][0] if len(hand[3]) else -1) for hand in observation['hands']], dtype = np.float32)

		collected_cards: np.ndarray = np.array([one_hot_encode(
			size = 52,
			arr = hand[2],
			dtype = np.float32
		) for hand in observation['hands']])

		exposed: np.ndarray = np.ones((GAME_SETTINGS['NUM_PLAYERS'], GAME_SETTINGS['NUM_CARDS']), dtype = np.float32)
		values: dict[int, int] = {shown_card: value for shown_card, value in observation['stacks'][1]}
		for i, _hand in enumerate(observation['hands']):
			for card in _hand[1]:
				if card in values.keys():
					exposed[i, card] = values[card]

		return {
			'game_state':		game_state,
			'hand':				hand,
			'scores':			scores,
			'current_trick':	current_trick,
			'collected_cards':	collected_cards,
			'exposed':			exposed
		}

	@staticmethod
	def get_value_from_state(state: dict[str, np.ndarray], turn_idx: int) -> int:
		return state['scores'][turn_idx] - np.mean(state['scores'])

	@staticmethod
	def get_reward_from_state_transition(old_state: dict[str, np.ndarray], new_state: dict[str, np.ndarray], turn_idx: int) -> int:
		raw_reward = MultiAgentEnv.get_value_from_state(new_state, turn_idx) - MultiAgentEnv.get_value_from_state(old_state, turn_idx)
		return raw_reward / 100.0

	async def wait_for_lobby(self, timeout: float = 10.0) -> bool:
		start_time = time.time()

		while time.time() - start_time < timeout:
			all_in_lobby = all(self._latest_observation[ws_idx] is None for ws_idx in range(self.num_agents))

			if all_in_lobby:
				return True

			await asyncio.sleep(0.1)

		return False

	def end_episode(self, ws_idx: int) -> None:
		self._batch_lens[ws_idx] = np.concatenate((self._batch_lens[ws_idx], np.array([self._episode_ts[ws_idx]])), axis = 0)
		self._batch_rewards[ws_idx].append(np.copy(self._episode_rewards[ws_idx]))
		self._episode_ts[ws_idx] = -1
		self._episode_rewards[ws_idx] = np.empty(0)

	async def train(self) -> None:
		is_in_game = False
		if self._latest_observation[0] is not None:
			print(f'{self._latest_observation[0] = }')
			is_in_game = self._latest_observation[0].get('gameState', '') != ''

		self.reset()
		self._batch_ts =				[0 for _ in range(self.num_agents)]
		self._episode_ts =				[0 for _ in range(self.num_agents)]
		self.is_rollout =				True
		self._rollout_progressbar =		tqdm(range(self.timesteps_per_batch * self.num_agents), dynamic_ncols = True, desc = f'Batch {self.batch_num} Rollout')
		self._rollout_progressbar.n =	0
		self._rollout_progressbar.refresh()
		self.actor.train()
		self.critic.train()

		if is_in_game:
			await self.ws_list[0].send(json.dumps({'tag': 'sendCommand', 'data': 'EXIT', 'timestamp': int(time.time() * 1000), 'currentFrame': {'$bigint': str(self._latest_observation[0]['currentFrame']) if self._latest_observation[0] is not None else -1}}))
		else:
			await self.ws_list[0].send(json.dumps({'tag': 'startGame', 'timestamp': int(time.time() * 1000)}))

	async def train_post_rollout(self) -> None:
		batch_advantages, batch_returns =	self.compute_gaes()							# (B,), (B,)

		batch_states =			torch.concatenate(self._batch_states, dim = 0)			# (B, dim(obs))
		batch_actions =			torch.concatenate(self._batch_actions, dim = 0)			# (B, dim(act))
		batch_action_masks =	torch.concatenate(self._batch_action_masks, dim = 0)	# (B, dim(act))
		batch_log_probs =		torch.concatenate(self._batch_log_probs, dim = 0)		# (B,)

		game_states = batch_states[:, :len(GameStates)]
		show_mask = game_states[:, 0:2].sum(dim = 1) > 0
		play_mask = game_states[:, 2:6].sum(dim = 1) > 0

		eps = 1e-9
		batch_advantages =	(batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + eps)

		with torch.no_grad():
			batch_values_old = self.critic(batch_states)

		actor_optimizer =	torch.optim.Adam(self.actor.parameters(),	lr = self.actor_lr)
		critic_optimizer =	torch.optim.Adam(self.critic.parameters(),	lr = self.critic_lr)

		batch_len =			batch_states.shape[0]
		mini_batch_len =	self.mini_batch_size

		actor_losses = []
		critic_losses = []
		kl_divergences = []

		for epoch in tqdm(range(self.n_updates_per_batch), desc = 'Training: ', dynamic_ncols = True):
			indices = torch.randperm(batch_len)

			for start in range(0, batch_len, mini_batch_len):
				end = start + mini_batch_len
				mini_batch_indices = indices[start:end]

				mini_batch_states =			batch_states[mini_batch_indices]
				mini_batch_actions =		batch_actions[mini_batch_indices]
				mini_batch_action_masks =	batch_action_masks[mini_batch_indices]
				mini_batch_log_probs =		batch_log_probs[mini_batch_indices]
				mini_batch_advantages =		batch_advantages[mini_batch_indices]
				mini_batch_returns =		batch_returns[mini_batch_indices]
				mini_batch_values_old =		batch_values_old[mini_batch_indices]
				mini_batch_show_mask =		show_mask[mini_batch_indices]
				mini_batch_play_mask =		play_mask[mini_batch_indices]

				mini_batch_log_probs_new =	torch.zeros(mini_batch_indices.shape[0])
				mini_batch_entropy =		torch.tensor(0.0)

				if mini_batch_show_mask.any():
					show_log_probs, show_entropy = self.actor['SHOW'].evaluate_actions(
						mini_batch_states[mini_batch_show_mask],
						mini_batch_actions[mini_batch_show_mask],
						mini_batch_action_masks[mini_batch_show_mask]
					)
					mini_batch_log_probs_new[mini_batch_show_mask] = show_log_probs
					mini_batch_entropy += show_entropy

				if mini_batch_play_mask.any():
					play_log_probs, play_entropy = self.actor['PLAY'].evaluate_actions(
						mini_batch_states[mini_batch_play_mask],
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

				mini_batch_values_new = self.critic(mini_batch_states)
				values_clipped = mini_batch_values_old + torch.clamp(mini_batch_values_new - mini_batch_values_old, -self.clip_epsilon, self.clip_epsilon)
				critic_loss_unclipped = (mini_batch_values_new - mini_batch_returns) ** 2
				critic_loss_clipped = (values_clipped - mini_batch_returns) ** 2
				critic_loss = 0.5 * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

				actor_optimizer.zero_grad()
				actor_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
				actor_optimizer.step()

				critic_optimizer.zero_grad()
				critic_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
				critic_optimizer.step()

				actor_losses.append(actor_loss.item())
				critic_losses.append(critic_loss.item())

				with torch.no_grad():
					kl = (mini_batch_log_probs - mini_batch_log_probs_new).mean()
					kl_divergences.append(kl.item())

		all_episode_rewards = [rewards.sum() for ws_rewards in self._batch_rewards for rewards in ws_rewards]

		self.training_history['batch'].append(self.batch_num)
		self.training_history['actor_loss'].append(actor_losses[-1])
		self.training_history['critic_loss'].append(critic_losses[-1])
		self.training_history['mean_kl_divergence'].append(np.mean(kl_divergences))
		self.training_history['mean_return'].append(batch_returns.mean().item())
		self.training_history['max_episode_reward'].append(max(all_episode_rewards) if all_episode_rewards else 0)
		self.training_history['min_episode_reward'].append(min(all_episode_rewards) if all_episode_rewards else 0)
		self.training_history['mean_episode_reward'].append(np.mean(all_episode_rewards) if all_episode_rewards else 0)

		print(f'-- Training Complete (Batch {self.batch_num}) --')
		print(f'  Total Samples: {batch_len}')
		print(f'  Actor Loss: {actor_losses[0]:.4f} -> {actor_losses[-1]:.4f}')
		print(f'  Critic Loss: {critic_losses[0]:.4f} -> {critic_losses[-1]:.4f}')
		print(f'  Mean KL Divergence: {np.mean(kl_divergences):.4f}')
		print(f'  Return range: [{batch_returns.min().item():.2f}, {batch_returns.max().item():.2f}]')
		print(f'  Episode rewards: min={min(all_episode_rewards):.1f}, max={max(all_episode_rewards):.1f}, mean={np.mean(all_episode_rewards):.1f}')

		self.batch_num += 1

		if self.batch_num % self.save_checkpoint_frequency == 0:
			self.save_checkpoint()

		await self.wait_for_lobby()
		await self.train()

	def compute_gaes(self) -> tuple[torch.Tensor, torch.Tensor]:
		all_advantages = []
		all_returns = []

		for ws_idx in range(self.num_agents):
			batch_advantages = []
			batch_returns = []

			with torch.no_grad():
				batch_values = self.critic(self._batch_states[ws_idx]).numpy()

			value_idx = 0
			for episode_idx, episode_rewards in enumerate(self._batch_rewards[ws_idx]):
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

		return torch.tensor(np.concatenate(all_advantages, axis = 0)), torch.tensor(np.concatenate(all_returns, axis = 0))

	def compute_reward_to_gos(self) -> torch.Tensor:
		all_reward_to_gos = []

		for ws_idx in range(self.num_agents):
			batch_rewards = self._batch_rewards[ws_idx]
			batch_reward_to_gos = []

			for episode_rewards in batch_rewards:
				discounted_reward = 0
				episode_reward_to_gos = []

				for reward in reversed(episode_rewards):
					discounted_reward = reward + discounted_reward * self.gamma
					episode_reward_to_gos.append(discounted_reward)

				batch_reward_to_gos.extend(reversed(episode_reward_to_gos))

			all_reward_to_gos.append(batch_reward_to_gos)

		return torch.tensor(np.concatenate(all_reward_to_gos, axis = 0))

	def get_models_in_directory(self, model_dir: str) -> list[str]:
		model_dir = os.path.abspath(model_dir)

		patterns = ['*.pt', '*.pth']
		model_files = []
		for pattern in patterns:
			model_files.extend(glob.glob(os.path.join(model_dir, pattern)))

		return sorted(model_files)

	def load_models(self, model_paths: list[str]) -> None:
		assert len(model_paths) == self.num_agents

		for ws_idx, model_path in enumerate(model_paths):
			self.eval_actors[ws_idx] = self.load_actor_from_checkpoint(model_path)
			self.eval_model_paths[ws_idx] = model_path

	async def eval(self) -> None:
		model_paths = self.get_models_in_directory(self.args.model_dir)

		if len(model_paths) == 0:
			print(f'No models found in directory [{os.path.abspath(self.args.model_dir)}]')
			return

		ws = self.ws_list[0]

		# Eval Loop
		while True:
			# Load Matchup
			batch_models = [random.choice(model_paths) for _ in range(self.num_agents)]
			self.load_models(batch_models)

			batch_model_names = [os.path.basename(p) for p in batch_models]
			batch_model_ratings = [f'{self.elo_rating_system.get_rating(p):.2f}' for p in batch_models]
			print(f'')
			print(f' === Starting Evaluation Matchup === ')
			print(f'  Models: {batch_model_names}')
			print(f'  Ratings: {batch_model_ratings}')
			print(f'  Games to Play: {self.eval_games_per_match}')

			# Play [self.eval_games_per_match] Games
			for game_idx in range(self.eval_games_per_match):

				# Reset State
				for ws_idx in range(self.num_agents):
					self.reset_game_state(ws_idx)

				self.is_evaluating = False

				self.eval_game_complete_event.clear()
				self.eval_scores = [None for _ in range(self.num_agents)]

				await ws.send(json.dumps({
					'tag': 'startGame',
					'timestamp': int(time.time() * 1000)
				}))

				await self.eval_game_complete_event.wait()

				if any(score is None for score in self.eval_scores):
					warnings.warn('Game ended buy scores were not captured, skipping ELO update')

				else:
					self.eval_total_games += 1

					rating_changes = self.elo_rating_system.update_ratings(self.eval_model_paths, self.eval_scores)

					print(f' --- Eval Game {game_idx + 1} / {self.eval_games_per_match} ---')
					print(f'  Total Games: {self.eval_total_games}')
					print(f'  Scores: {self.eval_scores}')
					for i in range(self.num_agents):
						model_path = self.eval_model_paths[i]
						model_name = os.path.basename(model_path)
						rating = self.elo_rating_system.get_rating(model_path)
						change = rating_changes.get(model_path, 0)
						print(f'  Agent {i} [{model_name}]:')
						print(f'    Score = {self.eval_scores[i]}')
						print(f'    ELO = {rating:.4f} ({change:.4f})')

					self.save_all_ratings()

				await ws.send(json.dumps({
					'tag': 'sendCommand',
					'data': 'EXIT',
					'timestamp': int(time.time() * 1000),
					'currentFrame': {'$bigint': str(self.eval_last_frame)}
				}))

				lobby_ready = await self.wait_for_lobby()

				if not lobby_ready:
					while not all(self._latest_observation[ws_idx] is None for ws_idx in range(self.num_agents)):
						await asyncio.sleep(0.5)

			# Matchup Complete
			self.eval_total_matches += 1
			self.print_leaderboard()

	def print_leaderboard(self) -> None:
		leaderboard = self.elo_rating_system.get_leaderboard()
		print(f'')
		print(f'{"=" * 60}')
		print(f' ELO LEADERBOARD (after {self.eval_total_games} games)')
		print(f' {"Rank":<6}{"Model":<35}{"ELO":<8}{"Games":<8}')
		print(f' {"-" * 58} ')
		for rank, (path, rating, games) in enumerate(leaderboard, 1):
			name = os.path.basename(path)
			print(f' {rank:<6}{name:<35}{rating:<8.2f}{games:<8}')
		print(f'{"=" * 60}')

	def save_all_ratings(self) -> None:
		saved_ratings_count = 0
		for model_path in self.elo_rating_system.ratings:
			try:
				self.elo_rating_system.save_rating(model_path)
				saved_ratings_count += 1
			except Exception as e:
				warnings.warn(f'Failed to save rating for [{model_path}]: {e}')

		print(f'Saved ELO ratings for {saved_ratings_count} / {len(self.elo_rating_system.ratings)} models')

class EloRatingSystem:

	def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0) -> None:
		self.k_factor =								k_factor
		self.initial_rating =						initial_rating
		self.ratings: dict[str, float] =			{}
		self.games_played: dict[str, float] =		{}
		self.history: list[dict] =					[]

	def get_rating(self, model_path: str) -> float:
		model_path = os.path.abspath(model_path)

		if model_path not in self.ratings.keys():
			self.load_rating(model_path)

		return self.ratings[model_path]

	def load_rating(self, model_path: str) -> None:
		model_path = os.path.abspath(model_path)
		try:
			checkpoint = torch.load(model_path, weights_only = False)
			self.ratings[model_path] = checkpoint.get('elo_rating', self.initial_rating)
			self.games_played[model_path] = checkpoint.get('elo_games_played', 0)
		except Exception as e:
			warnings.warn(f'Could not load rating from [{model_path}]: {e}')
			return None

	def save_rating(self, model_path: str) -> None:
		model_path = os.path.abspath(model_path)
		try:
			checkpoint = torch.load(model_path, weights_only = False)
			checkpoint['elo_rating'] = self.ratings[model_path]
			checkpoint['elo_games_played'] = self.games_played[model_path]
			torch.save(checkpoint, model_path)
		except Exception as e:
			warnings.warn(f'Could not save rating to [{model_path}]: {e}')

	def get_expected_score(self, rating_a: float, rating_b: float) -> float:
		return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

	def update_ratings(self, model_paths: list[str], final_scores: list[float]) -> dict[str, float]:
		model_paths = [os.path.abspath(p) for p in model_paths]

		assert len(model_paths) == len(final_scores)

		n = len(model_paths)

		current_ratings: list[float] = [self.get_rating(path) for path in model_paths]
		rating_changes: dict[str, list[float]] = {path: [] for path in model_paths}

		expected_scores = [0 for _ in range(n)]
		actual_scores = [0 for _ in range(n)]

		for i in range(n):
			for j in range(i + 1, n):
				expected_i = self.get_expected_score(current_ratings[i], current_ratings[j])
				expected_j = 1 - expected_i

				expected_scores[i] += expected_i
				expected_scores[j] += expected_j


				if final_scores[i] > final_scores[j]:
					actual_scores[i] += 1
					actual_scores[j] += 0
				elif final_scores[i] == final_scores[j]:
					actual_scores[i] += 0.5
					actual_scores[j] += 0.5
				else:
					actual_scores[i] += 0
					actual_scores[j] += 1

		expected_scores = [score / (n - 1) for score in expected_scores]
		actual_scores = [score / (n - 1) for score in actual_scores]

		changes = [self.k_factor * (actual - expected) for expected, actual in zip(expected_scores, actual_scores)]

		for i in range(n):
			rating_changes[model_paths[i]].append(changes[i])

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

async def console_server_handler(ws) -> None:
	global console_listeners
	global env
	try:
		console_listeners.add_ws(ws)
		async for message in ws:
			msg = json.loads(message)
			# print(msg)
			if msg['tag'] == 'sendCommand':
				ws_idx = console_listeners.name_to_idx[msg['id']]
				try:
					data = json.loads(msg['data'])
					assert type(data) == dict
					data['timestamp'] = int(time.time() * 1000)
					await env.ws_list[ws_idx].send(json.dumps(data))
				except:
					print(f'Invalid JSON string {msg["data"]}')
			if msg['tag'] == 'reset':
				env.reset()

		await ws.wait_closed()
	finally:
		console_listeners.remove_ws(ws)

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
		'--console-server-url', type = str,
		help = 'Console server URL to connect to'
	)
	parser.add_argument(
		'--spectate', action = 'store_true',
		help = 'Wait for spectators to join'
	)
	parser.add_argument(
		'--save-dir', type = str, default = '.',
		help = 'Where to save model files during training'
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
		'--model-dir', type = str,
		help = 'Directory containing model checkpoints'
	)
	parser.add_argument(
		'--k-factor', type = float, default = 32.0,
		help = 'K-Factor for Elo calculation'
	)

	args = parser.parse_args()

	if args.infer:
		if args.model is None:
			parser.error('--infer requires --model to be specified')
		if not os.path.isfile(args.model):
			parser.error(f'--model [{args.model}] is not a valid file')
		if args.lobby is None:
			parser.error('--infer requires --lobby to be specified')
		if args.lobby is None:
			parser.error('--infer requires --lobby to be specified')

	if args.eval:
		if args.model_dir is None:
			parser.error('--eval requires --model-dir to be specified')
		if not os.path.isdir(args.model_dir):
			parser.error(f'--model-dir [{args.model_dir}] is not a valid directory')

	return args

async def main() -> None:
	args: argparse.Namespace = parse_arguments()

	global console_listeners
	global env

	console_listeners = ConsoleListeners()

	if args.console_server_url is not None:
		console_server_url = urllib.parse.urlparse(args.console_server_url)
		console_server_host = console_server_url.hostname
		console_server_port = console_server_url.port

		if console_server_host is None:
			console_server_host = 'localhost'
		if console_server_port is None:
			console_server_port = '8080'

		console_server = await serve(handler = console_server_handler, host = console_server_host, port = console_server_port)

		input('Console server started, [Enter] to continue...')

	env = MultiAgentEnv(args)

	if args.model is not None:
		if os.path.exists(args.model):
			env.load_checkpoint(args.model)
		else:
			warnings.warn(f'Checkpoint not found at {os.path.abspath(args.model)}')

			if args.infer:
				warnings.warn('No checkpoint loaded; bot will use random initialization')

	if args.train:
		env.set_mode(MultiAgentEnv.ModelModes.train)
		print('Starting training mode...')

	elif args.infer:
		env.set_mode(MultiAgentEnv.ModelModes.infer)
		print('Starting inference mode...')

	elif args.eval:
		env.set_mode(MultiAgentEnv.ModelModes.eval)
		print('Starting evaluation mode...')

	else:
		raise ValueError('No mode specified')

	await env.connect(args.url)

if __name__ == '__main__':
	asyncio.run(main())

