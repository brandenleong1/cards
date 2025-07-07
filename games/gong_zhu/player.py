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

console_listeners = []

def broadcast_message_to_console_listeners(message: str):
	for ws in console_listeners:
		asyncio.create_task(ws.send(message))

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

class MultiAgentEnv:
	def __init__(self) -> None:
		self.num_agents: int =														GAME_SETTINGS['NUM_PLAYERS']
		self.ws_list: list[websockets.asyncio.client.ClientConnection | None] =		[None] * self.num_agents
		self._latest_observation: list[dict] =										[{}] * self.num_agents
		self._latest_state: list[dict | None] =										[None] * self.num_agents
		self._latest_reward: list[float] =											[0.0] * self.num_agents
		self._latest_done: list[bool] =												[False] * self.num_agents

		self._play_history: list[np.ndarray[np.int8] | None] =						[None] * self.num_agents
		self._leader_history: list[np.ndarray[np.int8] | None] =					[None] * self.num_agents

	async def connect(self, url: str) -> None:
		async def connect_ws(ws_idx):
			async with connect(url) as ws:
				self.ws_list[ws_idx] = ws

				broadcast_message_to_console_listeners(json.dumps({
					'tag': 'createConsole',
					'data': {'id': f'agent_{ws_idx}'}
				}))

				await ws.send(json.dumps({'tag': 'requestSessionID'}))
				await self._listen(ws_idx)

		tasks = [asyncio.create_task(connect_ws(i)) for i in range(self.num_agents)]
		await asyncio.gather(*tasks)

	async def _listen(self, ws_idx: int) -> None:
		ws = self.ws_list[ws_idx]

		async for message in ws:
			msg = json.loads(message)
			print(ws_idx, msg)
			broadcast_message_to_console_listeners(json.dumps({
				'tag': 'receiveCommand',
				'data': {
					'id': f'agent_{ws_idx}',
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
				MultiAgentEnv.encode_observation(msg['data']['gameData'], msg['data']['gameData']['turnOrder'].index(ws.username))

	def update_state(self, idx: int, state: dict) -> None:
		self._latest_state[idx] = state
		# Update self._latest_reward[idx] and self._latest_done[idx]

	async def reset(self) -> None:
		if self._latest_state[0]['gameState'] != '':
			await self.ws_list[0].send(json.dumps({'tag': 'sendCommand', 'data': 'exit'}))

		# Sleep while any(s is not None for s in self._latest_state)

		self._latest_observation =		[{}] * self.num_agents
		self.latest_state =				[None] * self.agents
		self._latest_reward =			[0.0] * self.num_agents
		self._latest_done =				[False] * self.num_agents

	def update_latest_from_gui_observation(self, observation: dict, idx: int) -> None:
		trick: int = round(len(observation['stacks'][0]) / len(observation['turnOrder']))
		# encoded_observation: dict = MultiAgentEnv.encode_observation(observation, idx)

		if observation['gameState'].startswith('PLAY_'):
			self._leader_history[idx][trick] = observation['turnFirstIdx']

		self._latest_observation[idx] = observation

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

	def get_observation(self, idx: int):		# TODO
		# play_history
		# leader_history
		latest_state: dict = self._latest_state[idx]

	@staticmethod
	def encode_observation(observation: dict, my_idx: int) -> dict[str, np.ndarray]:

		game_state: np.ndarray = one_hot_encode(
			size = len(GAME_STATE_MAP),
			arr = np.array([GAME_STATE_MAP[observation['gameState']]]),
			dtype = np.float32
		)

		hand: np.ndarray = one_hot_encode(
			size = 52,
			arr = observation['hands'][my_idx][0],
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
		for i, hand in enumerate(observation['hands']):
			for card in hand[1]:
				exposed[i, card] = values[card]

		return {
			'game_state':		game_state,
			'hand':				hand,
			'scores':			scores,
			'current_trick':	current_trick,
			'collected_cards':	collected_cards,
			'exposed':			exposed
		}


async def console_server_handler(ws) -> None:
	try:
		console_listeners.append(ws)
		await ws.wait_closed()
	finally:
		console_listeners.remove(ws)

async def main() -> None:
	url = 'ws://localhost:8080'

	console_server = await serve(handler = console_server_handler, host = 'localhost', port = '8000')

	input('Console server started, [Enter] to continue...')

	env = MultiAgentEnv()

	await env.connect(url)


if __name__ == '__main__':
	asyncio.run(main())
