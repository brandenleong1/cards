import json
import time

import asyncio
# import contextlib
import websockets
from websockets.asyncio.client import connect


NUM_PLAYERS = 4

async def wsJoin(url: str, ws_list: list[websockets.asyncio.client.ClientConnection | None], ws_idx: int) -> None:
	async with connect(url) as ws:
		ws_list[ws_idx] = ws

		await ws.send(json.dumps({'tag': 'requestSessionID'}))

		async for message in ws:
			msg = json.loads(message)
			print(ws_idx, msg)

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
						for ws_i in ws_list[1:]:
							await ws_i.send(json.dumps({'tag': 'getLobbies'}))

				else:
					if ws_idx == 0:
						if len(msg['data']['connected']) == NUM_PLAYERS:
							pass	# TODO start game

			elif msg['tag'] == 'updateLobbies':
				if not hasattr(ws, 'connected'):
					servers = msg['data']
					for server in servers:
						if (
							hasattr(ws_list[0], 'connected') and ws_list[0].connected and
							hasattr(ws_list[0], 'username') and server['host'] == ws_list[0].username
						):
							await ws.send(json.dumps({'tag': 'joinLobby', 'data': server}))
							break

async def main() -> None:
	url = 'ws://localhost:8080'

	ws = [None] * NUM_PLAYERS
	tasks = [asyncio.create_task(wsJoin(url = url, ws_list = ws, ws_idx = i)) for i in range(NUM_PLAYERS)]

	await asyncio.gather(*tasks)


if __name__ == '__main__':
	asyncio.run(main())
