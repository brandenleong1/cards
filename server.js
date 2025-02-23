const express = require('express');
const path = require('path');
const http = require('http');
const ws = require('ws');

const game = {
	gong_zhu: require(path.resolve(__dirname, './games/gong_zhu/server.js'))
};

const gameUtils = require(path.resolve(__dirname, './utils/game_utils.js'));
const commandParse = require(path.resolve(__dirname, './utils/command_parse.js'));

const app = express();
const app_port = process.env.PORT || 8080;

const server = http.createServer(app);
const wss = new ws.Server({server});

let messageDecoder = {
	'requestUsername': (ws, data) => {
		let res = game.gong_zhu.addUser(data.data);
		if (res[0]) {
			ws.username = res[1];
			game.gong_zhu.users[res[1]] = ws;
		}
		ws.send(JSON.stringify({tag: 'receiveUsername', status: res[0], data: res[1]}));
	},
	'getLobbies': (ws, data) => {
		ws.send(JSON.stringify({tag: 'updateLobbies', data: game.gong_zhu.servers}));
	},
	'createLobby': (ws, data) => {
		let res = game.gong_zhu.addServer(data.data);
		ws.send(JSON.stringify({tag: 'createdLobby', status: res[0], data: res[1]}));
	},
	'joinLobby': (ws, data) => {
		let idx = game.gong_zhu.getServerIdx(data.data);
		let server = game.gong_zhu.servers[idx];
		if (server.connected.length == server.maxPlayers) {
			ws.send(JSON.stringify({tag: 'broadcastedMessage', data: 'Lobby full'}));
		} else {
			let res = game.gong_zhu.joinServer(ws, data.data);
			if (res[0]) {
				for (let username of res[1].connected) {
					game.gong_zhu.users[username].send(JSON.stringify({tag: 'joinedLobby', status: res[0], data: res[1]}));
				}
				// console.log('joinServer', res[1].connected);
				// ws.connected = data.data;
				// console.log('joinServer', ws.connected, data.data.connected);
			} else {
				ws.send(JSON.stringify({tag: 'joinedLobby', status: res[0], data: res[1]}));
			}
		}
	},
	'leaveLobby': (ws, data) => {
		if (ws.connected) {
			let res = game.gong_zhu.leaveServer(ws, ws.connected);
			ws.send(JSON.stringify({tag: 'leftLobby', status: res[0], data: res[1]}));
			if (res[0]) {
				for (let username of ws.connected.connected) {
					let ws1 = game.gong_zhu.users[username];
					ws1.send(JSON.stringify({tag: 'joinedLobby', status: res[0], data: res[1]}));
				}
				delete ws.connected;
			}
		}
	},
	'startGame': (ws, data) => {
		let idx = game.gong_zhu.getServerIdx(ws.connected);
		let server = game.gong_zhu.servers[idx];
		// server.gameData = structuredClone(game.gong_zhu.defaultSettings);
		// console.log(server);
		if (server.connected.length < server.gameData.minPlayers) {
			ws.send(JSON.stringify({
				tag: 'broadcastedMessage',
				data: `Insufficient players ${server.connected.length} < ${server.gameData.minPlayers}`
			}));
			return;
		}

		game.gong_zhu.initGame(server);
		// console.log(server.gameData);
		let serverInfo = structuredClone(server);
		let gameData = structuredClone(server.gameData); // TODO - obfuscate
		delete serverInfo.gameData;
		game.gong_zhu.broadcastToConnected(server, {tag: 'startedGame', data: {gameData: gameData, serverData: serverInfo}});
	},
	'sendCommand': (ws, data) => {
		let idx = game.gong_zhu.getServerIdx(ws.connected);
		let server = game.gong_zhu.servers[idx];
		console.log(ws.username);
		game.gong_zhu.processCommand(data.data, ws, server);
	},
	'sendChat': (ws, data) => {
		let idx = game.gong_zhu.getServerIdx(ws.connected);
		let server = game.gong_zhu.servers[idx];
		let msgTime = Date.now();
		game.gong_zhu.broadcastToConnected(server, {tag: 'receiveChat', data: {
			username: ws.username,
			text: data.data,
			time: msgTime
		}});
	}
};

wss.on('connection', function(ws, req) {
	console.log('Connection:', req.url);
	console.log('Users:', wss.clients.size);

	ws.on('message', function(data, isBinary) {
		data = isBinary ? data : data.toString();
		data = JSON.parse(data);

		let tags = data.tag.split('/');
		let func;
		for (let i = 0; i < tags.length; i++) {
			if (!i) func = messageDecoder[tags[i]];
			else func = func[tags[i]];
		}
		if (func) func(ws, data);
	});

	for (let ws of wss.clients) {
		ws.send(JSON.stringify({tag: 'updateUserCount', data: wss.clients.size}));
	}

	ws.on('close', function() {
		console.log('Closed', this.username, this.connected);
		console.log(game.gong_zhu.servers);
		if (this.username) game.gong_zhu.removeUser(this.username);
		if (this.connected) {
			game.gong_zhu.leaveServer(this, this.connected);
			if (this.connected.gameData.gameState == '') game.gong_zhu.broadcastToConnected(
				this.connected,
				{tag: 'joinedLobby', status: 1, data: this.connected}
			);
			else game.gong_zhu.broadcastToConnected(
				this.connected,
				{tag: 'otherLeftLobby', status: 1, data: this.connected}
			)
		}
		console.log(game.gong_zhu.servers);
		for (let ws of wss.clients) {
			ws.send(JSON.stringify({tag: 'updateUserCount', data: wss.clients.size}));
		}
	});
});

server.listen(app_port);
