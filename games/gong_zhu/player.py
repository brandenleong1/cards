import json
import time
import copy
import math
import re
import random

import asyncio
# import contextlib
import websockets
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
import numpy as np
import itertools as it
import torch
import torch.nn.functional as F

from typing import Any


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


NUM_PLAYERS = 4

GAME_SETTINGS = {
	'NUM_PLAYERS': 4
}
GAME_STATE_MAP = {
	'SHOW_3':		0,
	'SHOW_ALL':		1,
	'PLAY_0':		2,
	'PLAY_1':		3,
	'PLAY_2':		4,
	'PLAY_3':		5,
	'SCORE':		6,
	'LEADERBOARD':	7
}


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

	def forward(self, inputs: torch.Tensor, mask: torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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

		game_state = [k for k, v in GAME_STATE_MAP.items() if v == np.argmax(state['game_state']).item()][0]

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
			masked_logits = logits.masked_fill(masks == 0, torch.tensor(-torch.inf))
			probs = torch.sigmoid(masked_logits)

			log_probs_bits = actions * torch.log(probs + eps) + (1 - actions) * torch.log(1 - probs + eps)
			log_probs = log_probs_bits.sum(dim = -1)

			valid_probs = probs * masks
			entropy_bits = -(valid_probs * torch.log(probs + eps) + (1 - valid_probs) * torch.log(1 - probs + eps))
			entropy = (entropy_bits * masks).sum(dim = -1).mean()

		elif self.module_type == 'PLAY':
			masked_logits = logits.masked_fill(masks == 0, torch.tensor(-torch.inf))
			probs = torch.softmax(masked_logits, dim = -1)

			action_indices = actions.argmax(dim = -1)

			log_probs = torch.log(probs.gather(-1, action_indices.unsqueeze(-1)).squeeze(-1) + eps)

			entropy = -(probs * torch.log(probs)).sum(dim = -1).mean()
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
			raise ''

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
	def __init__(self) -> None:
		self.num_agents: int =														GAME_SETTINGS['NUM_PLAYERS']
		self.ws_list: list[websockets.asyncio.client.ClientConnection | None] =		[None	for _ in range(self.num_agents)]

		self.reset()

		self.actor = torch.nn.ModuleDict({
			'SHOW': ActorNN('SHOW', (64, 64, 13)),
			'PLAY': ActorNN('PLAY', (64, 64, 13))
		})
		self.critic = CriticNN((64, 64))

		self._init_hyperparameters()

	def reset(self) -> None:
		self.is_training: bool =										False
		self.reset_timestamp: float =									time.time()

		self._latest_observation: list[dict | None] =					[None					for _ in range(self.num_agents)]
		self._latest_state: list[dict[str, np.ndarray] | None] =		[None					for _ in range(self.num_agents)]
		self._latest_actions: list[list[dict[str, Any]]] =				[list()					for _ in range(self.num_agents)]
		self._latest_reward: list[float | None] =						[None					for _ in range(self.num_agents)]
		self._latest_done: list[bool] =									[False					for _ in range(self.num_agents)]

		self._play_history: list[np.ndarray[np.int8]] =					[np.zeros((13, NUM_PLAYERS), dtype = np.int8) - 1	for _ in range(self.num_agents)]
		self._leader_history: list[np.ndarray[np.int8]] =				[np.zeros((13,), dtype = np.int8) - 1				for _ in range(self.num_agents)]

		self._batch_ts: list[int] =										[-1						for _ in range(self.num_agents)]
		self._batch_states: list[torch.Tensor] =						[torch.empty(0)			for _ in range(self.num_agents)]	# (B, dim(obs))
		self._batch_actions: list[torch.Tensor] =						[torch.empty(0)			for _ in range(self.num_agents)]	# (B, dim(act))
		self._batch_action_masks: list[torch.Tensor] =					[torch.empty(0)			for _ in range(self.num_agents)]	# (B, dim(act))
		self._batch_log_probs: list[torch.Tensor] =						[torch.empty(0)			for _ in range(self.num_agents)]	# (B,)
		self._batch_rewards: list[list[np.ndarray]] =					[list()					for _ in range(self.num_agents)]	# (E, t_per_E)
		self._batch_lens: list[np.ndarray | None] =						[np.empty(0)			for _ in range(self.num_agents)]	# (E,)

		self._episode_ts: list[int] =									[-1						for _ in range(self.num_agents)]
		self._episode_rewards: list[np.ndarray] =						[np.empty(0)			for _ in range(self.num_agents)]	# (t_per_E,)

		self._is_processing_event: list[bool] =							[False					for _ in range(self.num_agents)]
		self._event_queue: list[list[tuple[str, float]]] =				[list()					for _ in range(self.num_agents)]	# (message, timestamp)
		self._event_queue_last_clear: list[float] =						[self.reset_timestamp	for _ in range(self.num_agents)]

	def _init_hyperparameters(self) -> None:
		self.timesteps_per_batch: int =				256
		self.max_timesteps_per_episode: int =		200

		# PPO Parameters
		self.gamma: float =							0.99
		self.gae_lambda: float =					0.95
		self.clip_epsilon: float =					0.2
		self.n_updates_per_batch: int =				5
		self.lr: float =							3e-4
		self.entropy_coef: float =					0.01
		self.max_grad_norm: float =					0.5

	async def connect(self, url: str) -> None:
		global console_listeners
		async def connect_ws(ws_idx):
			async with connect(url) as ws:
				self.ws_list[ws_idx] = ws

				console_listeners.create_console(ws_idx, f'agent_{ws_idx}')

				await ws.send(json.dumps({'tag': 'requestSessionID', 'timestamp': int(time.time() * 1000)}))
				await self._listen(ws_idx)

		tasks = [asyncio.create_task(connect_ws(i)) for i in range(self.num_agents)]
		await asyncio.gather(*tasks)

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
		print(ws_idx, msg)

		if msg['tag'] == 'receiveSessionID':
			ws.sessionID =  msg['data']['sessionID']
			await ws.send(json.dumps({'tag': 'requestUsername', 'data': 'bot', 'timestamp': int(time.time() * 1000)}))

		elif msg['tag'] == 'receiveUsername':
			ws.username = msg['data']
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

		elif msg['tag'] == 'createdLobby':
			await ws.send(json.dumps({'tag': 'joinLobby', 'data': msg['data'], 'timestamp': int(time.time() * 1000)}))

		elif msg['tag'] == 'showLobby':

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

			else:
				if ws_idx == 0:
					if len(msg['data']['connected']) == GAME_SETTINGS['NUM_PLAYERS']:
						if self.is_training:
							await ws.send(json.dumps({
								'tag': 'startGame',
								'timestamp': int(time.time() * 1000)
							}))
						else:
							await self.train()		# can change this later

		elif msg['tag'] == 'updateLobbies':
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
						break

		elif msg['tag'] == 'updateGUI':

			# No Longer Training
			if not self.is_training:
				await self.unlock_processing_event(ws_idx)
				return

			updateEnvironmentSettings(msg)

			turn_idx = GAME_SETTINGS['turn_order'].index(ws.username)

			# Update Internals From Message
			prev_state = copy.deepcopy(self._latest_state[ws_idx])
			prev_observation = copy.deepcopy(self._latest_observation[ws_idx])
			self.update_latest_from_gui_observation(msg['data']['gameData'], ws_idx, GAME_SETTINGS['turn_order'].index(ws.username))

			self._latest_actions[ws_idx] = [e for e in self._latest_actions[ws_idx] if not (e['ack'] == 1 and e.get('newFrame') is not None and e['newFrame'] < msg['data']['gameData']['currentFrame'])]

			# GUI Update From Previous Command ACK
			found_matching_command = False
			try:
				found_idx = next(i for i, e in enumerate(self._latest_actions[ws_idx]) if e['newFrame'] == msg['data']['gameData']['currentFrame'] and e['ack'] == 1)
				self._latest_actions[ws_idx].pop(found_idx)
				found_matching_command = True
			except StopIteration:
				pass

			# Nothing Happened -> Return
			if json.dumps(prev_observation, sort_keys = True, separators = (',', ':')) == json.dumps(self._latest_observation[ws_idx], sort_keys = True, separators = (',', ':')):
				await self.unlock_processing_event(ws_idx)
				return

			# Reward Calculation
			frame_advanced = (
				prev_observation is not None and
				msg['data']['gameData']['currentFrame'] > prev_observation['currentFrame'] and
				found_matching_command
			)

			reward = None
			if prev_state is not None and frame_advanced:
				reward = MultiAgentEnv.get_reward_from_state_transition(prev_state, self._latest_state[ws_idx], turn_idx)
				self._episode_rewards[ws_idx] = np.concatenate((self._episode_rewards[ws_idx], np.array([reward])), axis = 0)

			# Episode Completion + Reset
			game_state = self._latest_observation[ws_idx]['gameState']
			done = game_state in {'LEADERBOARD', 'SCORE'}

			if (done or (self._episode_ts[ws_idx] >= self.max_timesteps_per_episode)) and self._episode_ts[ws_idx] >= 0:
				self.end_episode(ws_idx)

			if ws_idx == 0 and msg['data']['gameData']['gameState'] == 'LEADERBOARD':
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
					'msg': ['=== reward ===\n' + f'new [{MultiAgentEnv.get_value_from_state(self._latest_state[ws_idx], turn_idx)}] - old [{MultiAgentEnv.get_value_from_state(prev_state, turn_idx) if prev_state is not None else None}] = reward [{reward}]'],
					'status': 1
				}
			}))

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
			if self._latest_observation[ws_idx]['needToAct'][turn_idx] == 1:	# Need to also include end of game
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

					if self._episode_ts[ws_idx] < 0 or self._episode_ts[ws_idx] >= self.max_timesteps_per_episode:
						self._episode_rewards[ws_idx] = np.empty(0)
						self._episode_ts[ws_idx] = 0
						self._latest_state[ws_idx] = None

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
					try:
						found_idx = next(i for i, e in enumerate(self._latest_actions[ws_idx]) if e['oldFrame'] == self._latest_observation[ws_idx]['currentFrame'] and e['ack'] == -1)
						await self.unlock_processing_event(ws_idx)
						return
					except StopIteration:
						pass

					serialized_state = torch.tensor([ActorNN.serialize_state(self._latest_state[ws_idx])])
					self._batch_states[ws_idx] = torch.concatenate((self._batch_states[ws_idx], serialized_state), dim = 0)

					self._batch_ts[ws_idx] += 1
					self._episode_ts[ws_idx] += 1

					# Batch Full
					if self._batch_ts[ws_idx] >= self.timesteps_per_batch:
						self.is_training = False
						for _ws_idx in range(self.num_agents):
							self.end_episode(_ws_idx)

						await ws.send(json.dumps({
							'tag': 'sendCommand',
							'data': 'EXIT',
							'timestamp': int(time.time() * 1000),
							'currentFrame': {'$bigint': str(self._latest_observation[ws_idx]['currentFrame'] if self._latest_observation[ws_idx] is not None else -1)}
						}))

						asyncio.get_running_loop().run_in_executor(None, self.train_post_rollout)

						await self.unlock_processing_event(ws_idx)
						return

					with torch.no_grad():
						actor_module_type = ActorNN.get_module_type_from_game_state(self._latest_observation[ws_idx]['gameState'])

						if actor_module_type == '':
							await self.unlock_processing_event(ws_idx)
							return

						inputs = torch.tensor([ActorNN.serialize_state(self._latest_state[ws_idx])])
						mask = torch.tensor([self.actor[actor_module_type].calculate_action_mask(state) for state in batch])

						actions, log_probs = self.actor[actor_module_type](inputs, mask)

						console_listeners.broadcast_message(json.dumps({
							'tag': 'receiveCommand',
							'data': {
								'id': console_listeners.idx_to_name[ws_idx],
								'msg': [f'=== actions | log_probs ({self._latest_observation[ws_idx]["gameState"]}) ===\n' + json.dumps(actions.tolist()) + '\n' + json.dumps(log_probs.tolist())],
								'status': 1
							}
						}))

						self._batch_actions[ws_idx] = torch.concatenate((self._batch_actions[ws_idx], actions), dim = 0)
						self._batch_action_masks[ws_idx] = torch.concatenate((self._batch_action_masks[ws_idx], mask), dim = 0)
						self._batch_log_probs[ws_idx] = torch.concatenate((self._batch_log_probs[ws_idx], log_probs), dim = 0)

					await asyncio.sleep(0.5)
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

				trick: int = round(len(self._latest_observation[ws_idx]['stacks'][0]) / len(self._latest_observation[ws_idx]['turnOrder']))
				player_idx: int = self._latest_observation[ws_idx]['turnOrder'].index(username)
				self._play_history[ws_idx][trick][player_idx] = card_int

	def end_episode(self, ws_idx: int) -> None:
		self._batch_lens[ws_idx] = np.concatenate((self._batch_lens[ws_idx], np.array([self._episode_ts[ws_idx]])), axis = 0)
		self._batch_rewards[ws_idx].append(np.copy(self._episode_rewards[ws_idx]))
		self._episode_ts[ws_idx] = -1
		self._episode_rewards[ws_idx] = np.empty(0)

	async def train(self) -> None:
		is_in_game = False
		if self._latest_observation[0] is not None:
			is_in_game = self._latest_observation[0].get('gameState', '') != ''

		self.reset()
		self._batch_ts =		[0 for _ in range(self.num_agents)]
		self._episode_ts =		[0 for _ in range(self.num_agents)]
		self.is_training =		True
		self.actor.train()
		self.critic.train()

		if is_in_game:
			await self.ws_list[0].send(json.dumps({'tag': 'sendCommand', 'data': 'EXIT', 'timestamp': int(time.time() * 1000), 'currentFrame': {'$bigint': str(self._latest_observation[0]['currentFrame']) if self._latest_observation[0] is not None else -1}}))
		else:
			await self.ws_list[0].send(json.dumps({'tag': 'startGame', 'timestamp': int(time.time() * 1000)}))

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
			size = len(GAME_STATE_MAP),
			arr = np.array([GAME_STATE_MAP[observation['gameState']]]),
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
		return state['scores'][turn_idx] - (np.sum(state['scores']) - state['scores'][turn_idx])

	@staticmethod
	def get_reward_from_state_transition(old_state: dict[str, np.ndarray], new_state: dict[str, np.ndarray], turn_idx: int) -> int:
		return MultiAgentEnv.get_value_from_state(new_state, turn_idx) - MultiAgentEnv.get_value_from_state(old_state, turn_idx)

	def train_post_rollout(self) -> None:
		batch_reward_to_gos =				compute_reward_to_gos()		# (B,)
		batch_advantages, batch_returns =	compute_gaes()				# (B,), (B,)

		batch_states =			torch.concatenate(self._batch_states, dim = 0)			# (B, dim(obs))
		batch_actions =			torch.concatenate(self._batch_actions, dim = 0)			# (B, dim(act))
		batch_action_masks =	torch.concatenate(self._batch_action_masks, dim = 0)	# (B, dim(act))
		batch_log_probs =		torch.concatenate(self._batch_log_probs, dim = 0)		# (B,)
		batch_reward_to_gos =	torch.concatenate(self._batch_reward_to_gos, dim = 0)	# (B,)

		game_states = batch_states[:, :len(GAME_STATE_MAP)]
		show_mask = game_states[:, 0:2].sum(dim = 1) > 0
		play_mask = game_states[:, 2:6].sum(dim = 1) > 0

		eps = 1e-9
		batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + eps)

		actor_optimizer =	torch.optim.Adam(self.actor.parameters,		lr = self.lr)
		critic_optimizer =	torch.optim.Adam(self.critic.parameters,	lr = self.lr)

		batch_len = batch_states.shape[0]

		for update_idx in range(self.n_updates_per_batch):
			batch_log_probs_new = torch.zeros(batch_len)
			batch_entropy = torch.tensor(0.0)

			if mask_show_state.any():
				show_log_probs, show_entropy = self.actor['SHOW'].evaluate_actions(
					batch_states, batch_actions, batch_action_masks
				)
				batch_log_probs_new[show_mask] = show_log_probs
				batch_entropy += show_entropy

			if mask_play_state.any():
				play_log_probs, play_entropy = self.actor['PLAY'].evaluate_actions(
					batch_states, batch_actions, batch_action_masks
				)
				batch_log_probs_new[play_mask] = play_log_probs
				batch_entropy += play_entropy

			ratios = torch.exp(batch_log_probs_new - batch_log_probs.detatch())

			partial_min_1 = ratios * batch_advantages
			partial_min_2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

			actor_loss = -torch.min(partial_min_1, partial_min_2).mean() - self.entropy_coef * batch_entropy

			batch_values_new = self.critic(batch_states)
			critic_loss = F.mse_loss(batch_values_new, batch_returns)

			actor_optimizer.zero_grad()
			actor_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
			actor_optimizer.step()

			critic_optimizer.zero_grad()
			critic_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
			critic_optimizer.step()

		print(f'Training Complete:')
		print(f'  Total Samples: {batch_len}')
		print(f'  Actor Loss: {actor_loss.item():.4f}')
		print(f'  Critic Loss: {critic_loss.item():.4f}')
		print(f'  Mean Advantage: {batch_advantages.mean().item():.4f}')
		print(f'  Mean Return: {batch_returns.mean().item():.4f}')

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
						next_val = values[t + 1]

					delta = episode_rewards[t] + self.gamma * next_val - episode_values[t]
					last_gae = delta + self.gamma * self.gae_lambda * last_gae
					episode_advantages[t] = last_gae

				episode_returns = episode_advantages + episode_values

				batch_advantages.append(episode_advantages)
				batch_returns.append(episode_returns)

			all_advantages.append(np.concatenate(batch_advantages))
			all_returns.append(np.concatenate(batch_returns))

		return torch.concatenate(all_advantages, dim = 0), torch.concatenate(all_returns, dim = 0)

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

		return torch.concatenate(all_reward_to_gos, dim = 0)

	def evaluate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		batch_size = batch_states.shape[0]

		log_probs = torch.zeros(batch_size)
		entropy_list = []

		values = self.critic(batch_states)

		# TODO: continue


async def console_server_handler(ws) -> None:
	global console_listeners
	global env
	try:
		console_listeners.add_ws(ws)
		async for message in ws:
			msg = json.loads(message)
			print(msg)
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

async def main() -> None:
	url = 'ws://localhost:8080'

	global console_listeners
	global env

	console_listeners = ConsoleListeners()

	console_server = await serve(handler = console_server_handler, host = 'localhost', port = '8000')

	input('Console server started, [Enter] to continue...')

	env = MultiAgentEnv()

	await env.connect(url)


if __name__ == '__main__':
	asyncio.run(main())

