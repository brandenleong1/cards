import json
import time

import asyncio
from websockets.asyncio.client import connect


async def main():
	url = 'wss://gx00rdw5-8080.usw3.devtunnels.ms/'

	async with connect(url) as ws:
		await ws.send(json.dumps({'tag': 'requestSessionID'}))
		async for message in ws:
			msg = json.loads(message)
			print(msg)

			if msg['tag'] == 'receiveSessionID':
				ws.sessionID =  msg['data']['sessionID']
				await ws.send(json.dumps({'tag': 'requestUsername', 'data': 'bot'}))
			elif msg['tag'] == 'receiveUsername':
				ws.username = msg['data']
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
				pass


if __name__ == '__main__':
	asyncio.run(main())
