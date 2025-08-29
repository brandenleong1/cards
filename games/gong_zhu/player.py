import json
import time
import copy
import math
import re

import asyncio
# import contextlib
import websockets
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
import numpy as np
import itertools as it
import torch

# import environment


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
	suit_map: dict = {e: i for i, e in enumerate('♠♥♦♣')}
	rank: int = int(card[:-1])
	suit: int = suit_map(card[-1])
	return rank + (suit * 13)

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
			'data': {'id': ws_name}
		}))

class FeedForwardNN(torch.nn.Module):
	def __init__(self, module_type: str, *dims: tuple[int, ...]) -> None:
		super().__init__()

		self.network = torch.nn.Sequential(
			*[e for dim in dims for e in (torch.nn.LazyLinear(dim), torch.nn.ReLU())][:-1]
		)

		if type not in {'SHOW', 'PLAY'}:
			raise ValueError(f'Type [{type}] undefined')

		self.module_type = module_type

	def forward(self, state: dict[str, np.ndarray]) -> torch.Tensor:
		inputs = FeedForwardNN.serialize_state(state)

		if self.module_type == 'SHOW':
			logits = self.network(inputs)
			probs = torch.sigmoid(logits)
			if self.training:
				actions = torch.bernoulli(probs)
			else:
				actions = (probs > 0.5).float()
		elif self.module_type == 'PLAY':
			pass					# TODO
		else:
			return torch.empty((0,))

	def calculate_action_mask(self, state: dict[str, np.ndarray]) -> torch.Tensor:
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
				hidden = set(hand.tolist()) - set(exposed)
				legal_cards.update(hidden)
			else:
				legal_cards.update(hand.tolist())
		else:
			raise ValueError(f'Unknown game_state [{game_state}]')

		out_dim = self.network[-1].out_features
		mask = torch.zeros((out_dim,))

		hand = np.where(state['hand'] == 1)[0]
		padded_hand = np.pad(hand, (0, out_dim - len(hand)), mode = 'constant', constant = 0)
		# TODO


	@staticmethod
	def serialize_state(state: dict[str, np.ndarray]) -> np.ndarray:
		order = ['game_state', 'hand', 'scores', 'current_trick', 'collected_cards', 'exposed', 'play_history', 'leader_history']

		return np.concatenate([state[k].flatten() for k in order], axis = 0)

class MultiAgentEnv:
	def __init__(self) -> None:
		self.num_agents: int =														GAME_SETTINGS['NUM_PLAYERS']
		self.ws_list: list[websockets.asyncio.client.ClientConnection | None] =		[None] * self.num_agents

		self._latest_observation: list[dict] =										[{}] * self.num_agents
		self._latest_state: list[dict[str, np.ndarray] | None] =					[None] * self.num_agents
		self._latest_action: list[np.ndarray | None] =								[None] * self.num_agents
		self._latest_reward: list[float] =											[0.0] * self.num_agents
		self._latest_done: list[bool] =												[False] * self.num_agents

		self._play_history: list[np.ndarray[np.int8] | None] =						[None] * self.num_agents
		self._leader_history: list[np.ndarray[np.int8] | None] =					[None] * self.num_agents

		self._batch_ts: list[int] =													[-1] * self.num_agents
		self._batch_states: list[np.ndarray | None] =								[None] * self.num_agents	# (B, dim(obs))
		self._batch_actions: list[np.ndarray | None] =								[None] * self.num_agents	# (B, dim(act))
		self._batch_log_probs: list[np.ndarray | None] =							[None] * self.num_agents	# (B,)
		self._batch_rewards: list[np.ndarray | None] =								[None] * self.num_agents	# (E, t_per_E)
		self._batch_reward_to_gos: list[np.ndarray | None] =						[None] * self.num_agents	# (t_per_B,)
		self._batch_lens: list[np.ndarray | None] =									[None] * self.num_agents	# (E,)

		self._episode_ts: list[int] =												[-1] * self.num_agents
		self._episode_rewards: list[np.ndarray | None] =							[None] * self.num_agents	# (t_per_E,)

		self._init_hyperparameters()

	def _init_hyperparameters(self) -> None:
		self.timesteps_per_batch: int =				4800
		self.max_timesteps_per_episode: int =		1600

	async def connect(self, url: str) -> None:
		global console_listeners
		async def connect_ws(ws_idx):
			async with connect(url) as ws:
				self.ws_list[ws_idx] = ws

				console_listeners.create_console(ws_idx, f'agent_{ws_idx}')

				await ws.send(json.dumps({'tag': 'requestSessionID'}))
				await self._listen(ws_idx)

		tasks = [asyncio.create_task(connect_ws(i)) for i in range(self.num_agents)]
		await asyncio.gather(*tasks)

	async def _listen(self, ws_idx: int) -> None:
		global console_listeners
		ws = self.ws_list[ws_idx]

		async for message in ws:
			msg = json.loads(message)
			print(ws_idx, msg)
			console_listeners.broadcast_message(json.dumps({
				'tag': 'receiveCommand',
				'data': {
					'id': console_listeners.idx_to_name[ws_idx],
					'msg': [message],
					'status': 1
				}
			}))

			if msg['tag'] == 'receiveSessionID':
				ws.sessionID =  msg['data']['sessionID']
				await ws.send(json.dumps({'tag': 'requestUsername', 'data': 'bot'}))

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
						}
					}))

			elif msg['tag'] == 'createdLobby':
				await ws.send(json.dumps({'tag': 'joinLobby', 'data': msg['data']}))

			elif msg['tag'] == 'showLobby':

				if not hasattr(ws, 'connected'):
					ws.connected = True

					if ws_idx == 0:
						settings = copy.deepcopy(msg['data']['gameData']['settings'])
						settings['spectatorPolicy'] = 'constant'
						settings['expose3'] = True
						settings['zhuYangManJuan'] = True
						await ws.send(json.dumps({'tag': 'updateLobbySettings', 'data': {'settings': settings}}))

						updateEnvironmentSettings(msg)

						self._play_history =	[np.full((math.ceil(GAME_SETTINGS['NUM_CARDS'] / GAME_SETTINGS['NUM_PLAYERS']), GAME_SETTINGS['NUM_PLAYERS']), -1, dtype = np.int8)] * self.num_agents
						self._leader_history =	[np.full((math.ceil(GAME_SETTINGS['NUM_CARDS'] / GAME_SETTINGS['NUM_PLAYERS']),), -1, dtype = np.int8)] * self.num_agents

						for ws_i in self.ws_list[1:]:
							await ws_i.send(json.dumps({'tag': 'getLobbies'}))

				else:
					if ws_idx == 0:
						if len(msg['data']['connected']) == GAME_SETTINGS['NUM_PLAYERS']:
							await ws.send(json.dumps({'tag': 'startGame'}))

			elif msg['tag'] == 'updateLobbies':
				if not hasattr(ws, 'connected'):
					servers = msg['data']
					for server in servers:
						if (
							hasattr(self.ws_list[0], 'connected') and self.ws_list[0].connected and
							hasattr(self.ws_list[0], 'username') and server['host'] == self.ws_list[0].username
						):
							await ws.send(json.dumps({'tag': 'joinLobby', 'data': server}))
							break

			elif msg['tag'] == 'updateGUI':
				updateEnvironmentSettings(msg)
				if ws_idx == 0 and msg['data']['gameData']['gameState'] == 'LEADERBOARD':
					await ws.send(json.dumps({'tag': 'sendCommand', 'data': 'deal'}))

				# print(f'===> my_idx = {msg['data']['gameData']['turnOrder'].index(ws.username)} | ws.username = {ws.username}')
				# else:
				my_idx = GAME_SETTINGS['turn_order'].index(ws.username)
				self.update_latest_from_gui_observation(msg['data']['gameData'], ws_idx, GAME_SETTINGS['turn_order'].index(ws.username))
				console_listeners.broadcast_message(json.dumps({
					'tag': 'receiveCommand',
					'data': {
						'id': console_listeners.idx_to_name[ws_idx],
						'msg': [
							f'my_idx = {GAME_SETTINGS["turn_order"].index(ws.username)}',
							str(self._latest_observation[ws_idx]),
							str(FeedForwardNN.serialize_state(self._latest_state[ws_idx]).shape),
							json.dumps(self._leader_history[ws_idx].tolist())
						],
						'status': 1
					}
				}))

				if self._batch_ts[ws_idx] >= 0 and self._batch_ts[ws_idx] < self.timesteps_per_batch:
					# TODO episode_rewards.append(latest_reward)
					if self._episode_ts[ws_idx] >= 0 and self._episode_ts[ws_idx] < self.max_timesteps_per_episode:

						self._batch_ts[ws_idx] += 1
						self._episode_tx[ws_idx] += 1

						self._batch_states[ws_idx] = np.concatenate((self._batch_states[ws_idx], FeedForwardNN.serialize_state(self._latest_state[ws_idx])[np.newaxis, ...]), axis = 0)
						# TODO action, log_prob = get_action(self._latest_state[ws_idx])
						# TODO batch_actions.append(action)
						# TODO batch_log_probs.append(action)
						# TODO act(action)

					# TODO batch_lens.append(episode_t + 1)
					# TODO batch_rewards.append(episode_rewards)

				if self._latest_observation[ws_idx]['needToAct'][my_idx] == 1:
					console_listeners.broadcast_message(json.dumps({
						'tag': 'receiveCommand',
						'data': {
							'id': console_listeners.idx_to_name[ws_idx],
							'msg': ['NEED TO ACT'],
							'status': 0
						}
					}))
					# ws.act()

			elif msg['tag'] == 'receiveCommand':
				pass		# TODO update_latest_from_console_observation

	async def reset(self) -> None:
		latest_state = self._latest_observation[0]['gameState'] != ''

		self._latest_observation =		[{}] * self.num_agents
		self._latest_state =			[None] * self.num_agents
		self._latest_action =			[None] * self.num_agents
		self._latest_reward =			[0.0] * self.num_agents
		self._latest_done =				[False] * self.num_agents

		if latest_state:
			await self.ws_list[0].send(json.dumps({'tag': 'sendCommand', 'data': 'exit'}))
		else:
			await self.ws_list[0].send(json.dumps({'tag': 'startGame'}))

		# Sleep while any(s is not None for s in self._latest_state)
		while any(s is not None for s in self._latest_state):
			await asyncio.sleep(0.1)

	def update_latest_from_gui_observation(self, observation: dict, idx: int, turn_idx: int) -> None:
		trick: int = round(len(observation['stacks'][0]) / len(observation['turnOrder']))
		partial_state: dict[str, np.ndarray] = MultiAgentEnv.encode_observation(observation, turn_idx)

		if observation['gameState'].startswith('PLAY_'):
			self._leader_history[idx][trick] = observation['turnFirstIdx']

		# state = np.concatenate([partial_state, self._play_history[idx].flatten(), self._leader_history[idx]], axis = 0)
		state = partial_state | {'play_history': self._play_history[idx], 'leader_history': self._leader_history[idx]}

		self._latest_observation[idx] = observation
		self._latest_state[idx] = state

		# If latest action exists, log latest reward

	def update_latest_from_console_observation(self, observation: list[str], idx: int) -> None:
		for obs in observation:
			match: re.Match | None = re.match('Player \\[(.*)\\] played card \\[(.*)\\]', obs)
			if match:
				username: str; card: str
				username, card = match.groups()
				card_int: int = card_str_to_idx(card)

				trick: int = round(len(self._latest_observation[idx]['stacks'][0]) / len(self._latest_observation[idx]['turnOrder']))
				player_idx: int = self._latest_observation[idx]['turnOrder'].index(username)
				self._play_history[idx][trick][player_idx] = card_int

	async def train(self) -> None:
		await self.reset()
		self._batch_ts = [0] * self.num_agents
		self._episode_ts = [0] * self.num_agents
		self._episode_rewards = [np.empty((0,))] * self.num_agents
		# get game state

	@staticmethod
	def encode_observation(observation: dict, my_idx: int) -> dict[str, np.ndarray]:

		game_state: np.ndarray = one_hot_encode(
			size = len(GAME_STATE_MAP),
			arr = np.array([GAME_STATE_MAP[observation['gameState']]]),
			dtype = np.float32
		)

		hand: np.ndarray = one_hot_encode(
			size = 52,
			arr = np.array(observation['hands'][my_idx][0]),
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
		values: dict[int, int] = {card[0]: card[1] for card in observation['stacks'][1]}
		for i, _hand in enumerate(observation['hands']):
			for card in _hand[1]:
				exposed[i, card] = values[card]

		# return np.concatenate([e.flatten() for e in (game_state, hand, scores, current_trick, collected_cards, exposed)], axis = 0)

		return {
			'game_state':		game_state,
			'hand':				hand,
			'scores':			scores,
			'current_trick':	current_trick,
			'collected_cards':	collected_cards,
			'exposed':			exposed
		}

	@staticmethod
	def get_value_from_state(state: dict[str, np.ndarray], my_idx: int) -> int:
		return state['scores'][my_idx]

	@staticmethod
	def get_reward_from_state_transition(old_state: dict[str, np.ndarray], new_state: dict[str, np.ndarray], my_idx: int) -> int:
		return MultiAgentEnv.get_value_from_state(new_state) - MultiAgentEnv.get_value_from_state(old_state)

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
					await env.ws_list[ws_idx].send(json.dumps(json.loads(msg['data'])))
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
