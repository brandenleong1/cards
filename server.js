const express = require('express');
const path = require('path');
const http = require('http');
const ws = require('ws');

const game = {
	gong_zhu: require(path.resolve(__dirname, './games/gong_zhu/server.js'))
};

const gameUtils = require(path.resolve(__dirname, './utils/game_utils.js'));
const commandParse = require(path.resolve(__dirname, './utils/command_parse.js'));
const Utils = require(path.resolve(__dirname, './utils/utils.js'))

const app = express();
const app_port = process.env.PORT || 8080;

const server = http.createServer(app);
const wss = new ws.Server({server});

let wssClientsArchive = new Map();	// WS Info => Disconnect Time

let messageDecoder = {
	'checkSessionID': (ws, data) => {
		let archive = Array.from(wssClientsArchive.keys()).map(e => JSON.parse(e));
		let res = archive.find(ws => ws.sessionID == data.data);
		if (res) {
			game.gong_zhu.users.set(res.username, ws);
			ws.username = res.username;
			ws.sessionID = res.sessionID;
			let idx = game.gong_zhu.getServerIdx(res.connected);
			let server = null;
			if (idx != -1) {
				server = game.gong_zhu.servers[idx];
				ws.connected = server;
			}
			wssClientsArchive.delete(JSON.stringify(res));
			ws.send(JSON.stringify({tag: 'receiveSessionID', status: 1, data: {
				username: ws.username,
				sessionID: ws.sessionID
			}}));
			console.log(server.name);
			if (server) {
				if (server.gameData.gameState == '') {
					ws.send(JSON.stringify({tag: 'showLobby', status: 1, data: server}));
				} else {
					Utils.broadcastToConnected(game.gong_zhu.users, server, {tag: 'startedGame'});
					Utils.broadcastGameStateToConnected(game.gong_zhu.users, server, game.gong_zhu.obfuscateGameData);
				}
			}
		} else {
			let sessionIDs = new Set(Array.from(wss.clients.union(wssClientsArchive)).map(ws => ws.sessionID));
			let res = Utils.generateSessionID(sessionIDs);
			ws.sessionID = res;
			ws.send(JSON.stringify({tag: 'receiveSessionID', status: 0, data: {
				sessionID: res
			}}));
		}
	},
	'requestSessionID': (ws, data) => {
		let sessionIDs = new Set(Array.from(wss.clients.union(wssClientsArchive)).map(ws => ws.sessionID));
		let res = Utils.generateSessionID(sessionIDs);
		ws.sessionID = res;
		ws.send(JSON.stringify({tag: 'receiveSessionID', status: 0, data: {
			sessionID: res
		}}));
	},
	'requestUsername': (ws, data) => {
		let res = game.gong_zhu.addUser(data.data);
		if (res[0]) {
			ws.username = res[1];
			game.gong_zhu.users.set(res[1], ws);
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
		let res = game.gong_zhu.joinServer(ws, data.data);
		if (res[0] && server.gameData.gameState == '') {
			Utils.broadcastToConnected(game.gong_zhu.users, server, {tag: 'showLobby', status: res[0], data: res[1]});
		} else {
			ws.send(JSON.stringify({tag: 'showLobby', status: res[0], data: res[1]}));
			if (res[0] && server.gameData.gameState != '') {
				ws.send(JSON.stringify({tag: 'startedGame'}));
				Utils.broadcastGameState(ws, server, game.gong_zhu.obfuscateGameData);
			}
		}
	},
	'leaveLobby': (ws, data) => {
		if (ws.connected) {
			let server = ws.connected;
			let isPlayerOrHost = (ws.username == server.host) || server.gameData.turnOrder.includes(ws.username);
			let res = game.gong_zhu.leaveServer(ws, server);
			ws.send(JSON.stringify({tag: 'leftLobby', status: res[0], data: res[1]}));
			if (res[0]) {
				if (server.gameData.gameState == '') Utils.broadcastToConnected(game.gong_zhu.users, 
					server,
					{tag: 'showLobby', status: res[0], data: res[1]}
				);
				else if (isPlayerOrHost) Utils.broadcastToConnected(game.gong_zhu.users, 
					server,
					{tag: 'otherLeftLobby', status: res[0], data: res[1]}
				);
				delete ws.connected;
			}
		}
	},
	'updateLobbySettings': (ws, data) => {
		if (!ws.connected) return;

		let idx = game.gong_zhu.getServerIdx(ws.connected);
		let server = game.gong_zhu.servers[idx];
		if (ws.username == server.host) {
			let res = game.gong_zhu.updateServerSettings(server, data.data.settings);
			if (res[0]) {
				Utils.broadcastToConnected(game.gong_zhu.users, ws.connected, {tag: 'showLobby', status: res[0], data: res[1]});
			}
		}
	},
	'startGame': (ws, data) => {
		let idx = game.gong_zhu.getServerIdx(ws.connected);
		let server = game.gong_zhu.servers[idx];
		if (server.connected.length < server.gameData.minPlayers) {
			ws.send(JSON.stringify({
				tag: 'broadcastedMessage',
				data: `Insufficient players ${server.connected.length} < ${server.gameData.minPlayers}`
			}));
			return;
		}

		game.gong_zhu.initGame(server);

		let newTurnOrder = new Set(server.gameData.turnOrder);
		for (let user of server.connected) {
			game.gong_zhu.users.get(user.username).send(JSON.stringify({
				tag: 'broadcastedMessage',
				data: (newTurnOrder.has(user.username) ? 'You are now playing!' : 'You are now spectating!')
			}));
		}
		Utils.broadcastToConnected(game.gong_zhu.users, server, {tag: 'startedGame'});
	},
	'sendCommand': (ws, data) => {
		let idx = game.gong_zhu.getServerIdx(ws.connected);
		let server = game.gong_zhu.servers[idx];
		console.log(ws.username);

		let res = game.gong_zhu.processCommand(data.data, ws, server);
		let resToAll = structuredClone(res);
		resToAll.data = res.data.filter(e => e.toAll).map(e => e.msg);
		res.data = res.data.map(e => e.msg);

		ws.send(JSON.stringify(res));
		Utils.broadcastToConnected(game.gong_zhu.users, server, resToAll, ws.username);
	},
	'sendChat': (ws, data) => {
		let idx = game.gong_zhu.getServerIdx(ws.connected);
		let server = game.gong_zhu.servers[idx];
		let msgTime = Date.now();
		Utils.broadcastToConnected(game.gong_zhu.users, server, {tag: 'receiveChat', data: {
			username: ws.username,
			text: data.data,
			time: msgTime
		}});
	}
};

wss.on('listening', function() {
	setInterval(function() {
		let purged = Utils.purgeArchive(wssClientsArchive, 1 * 60 * 1000); // UPDATE 3 minute timeout

		purged.forEach(e => {
			let ws = JSON.parse(e);
			if (ws.username) game.gong_zhu.removeUser(ws.username);
			if (ws.connected) {
				let [status, server] = game.gong_zhu.leaveServer(ws, ws.connected);
				if (status && server) {
					if (server.gameData.gameState == '') Utils.broadcastToConnected(game.gong_zhu.users, 
						server,
						{tag: 'showLobby', status: 1, data: server}
					);
					else Utils.broadcastToConnected(game.gong_zhu.users, 
						server,
						{tag: 'otherLeftLobby', status: 1, data: server}
					);
				}
			}
			for (let ws of wss.clients) {
				ws.send(JSON.stringify({tag: 'updateUserCount', data: wss.clients.size}));
			}
		});

		console.table({
			servers: game.gong_zhu.servers.length,
			users: game.gong_zhu.users.size,
			purged: purged.size,
			archived: wssClientsArchive.size
		});
	}, 60 * 1000);
});

wss.on('connection', function(ws, req) {
	console.log('Connection:', req.socket.remoteAddress);
	console.log('Users:', wss.clients.size);

	let timeout = Math.max(wss._server.keepAliveTimeout - 1000, 1000);
	if (!ws.handlers) ws.handlers = new Map();
	ws.handlers.ping = setInterval(function() {
		ws.ping();
	}, timeout);

	ws.active = 1;

	for (let ws of wss.clients) {
		ws.send(JSON.stringify({tag: 'updateUserCount', data: wss.clients.size}));
	}

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

	ws.on('close', function() {
		clearInterval(this.handlers.ping);
		console.log('Closed', this.username, this.connected);

		let oldWs = {
			username: this.username,
			sessionID: this.sessionID,
			connected: this.connected ? {
				name: this.connected.name,
				time: this.connected.time,
				creator: this.connected.creator
			} : undefined,
			active: 0
		};

		if (this.username && this.connected) {
			wssClientsArchive.set(JSON.stringify(oldWs), Date.now());
		}
	});
});

server.listen(app_port);
