import * as Utils from '../utils/utils.js';

export class Game {
	constructor() {
		this.users = new Map();
		this.usernames = new Map();
		this.servers = [];
	}

	get defaultSettings() {
		throw new Error('defaultSettings must be implemented by subclass');
	}

	obfuscateGameData(gameData, turnOrderIdx) {
		throw new Error('obfuscateGameData must be implemented by subclass');
	}

	processCommand(data, ws, server) {
		throw new Error('processCommand must be implemented by subclass');
	}

	initGame(server) {
		throw new Error('initGame must be implemented by subclass');
	}

	addUser(username) {
		let count = this.usernames.get(username) || 0;
		if (count >= 1000) return [0, 'Username saturated'];

		while (true) {
			let t = Math.floor(Math.random() * 10000).toString(10).padStart(4, '0');
			let user = username + '#' + t;
			if (!this.users.has(user)) {
				this.usernames.set(username, count + 1);
				return [1, user];
			}
		}
	}

	removeUser(username) {
		if (!username) return;
		let base = username.split('#', 1)[0];
		let count = this.usernames.get(base) || 0;
		if (count <= 1) this.usernames.delete(base);
		else this.usernames.set(base, count - 1);
		this.users.delete(username);
	}

	addServer(serverData) {
		serverData.connected = [];
		serverData.commandQueue = [];
		serverData.processingCommand = false;
		serverData.gameData = structuredClone(this.defaultSettings);
		let idx = Utils.binaryInsert(this.servers, serverData, function(a, b) {
			if (a.time != b.time) return b.time - a.time;
			else if (a.name != b.name) return a.name.localeCompare(b.name);
			else return a.creator.localeCompare(b.creator);
		});

		return [idx == -1 ? 0 : 1, serverData];
	}

	getServerIdx(serverData) {
		return Utils.binarySearchIdx(this.servers, serverData, function(a, b) {
			if (a.time != b.time) return b.time - a.time;
			else if (a.name != b.name) return a.name.localeCompare(b.name);
			else return a.creator.localeCompare(b.creator);
		});
	}

	joinServer(ws, server, spectateOnly = false) {
		let idx = this.getServerIdx(server);

		if (idx == -1) return [0, 'Server does not exist'];
		else if (this.servers[idx].gameData.settings.spectatorPolicy == 'disallowed' && this.servers[idx].gameData.maxPlayers <= this.servers[idx].connected.length) return [0, 'Server full (Spectators disallowed)'];
		else {
			let lowestPriority = Utils.getLowestPriority(this.servers[idx]);
			if (lowestPriority == -Infinity) lowestPriority = -1;
			Utils.binaryInsert(this.servers[idx].connected, {username: ws.username, priority: lowestPriority + 1, spectateOnly: spectateOnly}, function(a, b) {
				return (a.username).localeCompare(b.username);
			});
			ws.connected = this.servers[idx];
			Utils.updatePriorities(this.servers[idx]);
			return [1, this.servers[idx]];
		}
	}

	leaveServer(ws, server) {
		let idx = this.getServerIdx(server);

		if (idx == -1) return [0, 'Server does not exist'];
		else {
			let idx2 = Utils.binarySearchIdx(this.servers[idx].connected, {username: ws.username}, function(a, b) {
				return (a.username).localeCompare(b.username);
			});
			if (idx2 != -1) this.servers[idx].connected.splice(idx2, 1);
			if (!this.servers[idx].connected.length) {
				this.servers.splice(idx, 1);
				return [1, null];
			}
			if (this.servers[idx].host == ws.username) {
				let hostIdx = Math.floor(Math.random() * this.servers[idx].connected.length);
				this.servers[idx].host = this.servers[idx].connected[hostIdx].username;
			}
			Utils.updatePriorities(this.servers[idx]);
			return [1, this.servers[idx]];
		}
	}

	purgeStaleServers() {
		for (let idx = this.servers.length - 1; idx >= 0; idx--) {
			let server = this.servers[idx];

			server.connected = server.connected.filter(member => {
				let ws = this.users.get(member.username);
				return ws !== undefined && ws.connected === server;
			});

			if (!server.connected.length) {
				this.servers.splice(idx, 1);
				continue;
			}

			if (!server.connected.some(member => member.username == server.host)) {
				let hostIdx = Math.floor(Math.random() * server.connected.length);
				server.host = server.connected[hostIdx].username;
			}
			Utils.updatePriorities(server);
		}
	}

	updateServerSettings(server, settings) {
		let idx = this.getServerIdx(server);
		if (idx == -1) return [0, 'Server does not exist'];
		else {
			for (let property in settings) {
				this.servers[idx].gameData.settings[property] = settings[property];
			}
			if (this.servers[idx].gameData.settings.spectatorPolicy == 'disallowed') this.removeAllSpectators(this.servers[idx]);
			return [1, this.servers[idx]];
		}
	}

	removeSpectator(user, server) {
		let ws = this.users.get(user.username);
		let res = this.leaveServer(ws, server);
		if (res[0]) {
			ws.send(Utils.JSONStringify({tag: 'broadcastedMessage', data: 'Spectators kicked', timestamp: Date.now()}));
			ws.send(Utils.JSONStringify({tag: 'leftLobby', status: res[0], data: res[1], timestamp: Date.now()}));
			delete ws.connected;
		}
	}

	removeAllSpectators(server) {
		Utils.updatePriorities(server);

		let toRemove = Utils.getUsersSortedByPriority(server);
		toRemove.filter(user => user.spectateOnly).forEach(user => this.removeSpectator(user, server));

		toRemove = Utils.getUsersSortedByPriority(server);
		let hostIdx = toRemove.findIndex(user => user.username == server.host);
		toRemove.splice(hostIdx, 1);
		toRemove.splice(0, server.gameData.maxPlayers - 1);

		toRemove.forEach(user => this.removeSpectator(user, server));

		server.connected.forEach(user => this.users.get(user.username).send(Utils.JSONStringify({tag: 'showLobby', status: 1, data: server, timestamp: Date.now()})));
	}

	rotateSpectators(server) {
		let connected = [];

		let prioritySort = function(a, b) {
			return a.priority - b.priority;
		};

		switch (server.gameData.settings.spectatorPolicy) {
			case 'round-robin': {
				let previousPlayers = new Set(server.gameData.turnOrder);

				let [spectateOnly, players] = structuredClone(server.connected).reduce(function([s, p], e) {
					return (e.spectateOnly ? [[...s, e], p] : [s, [...p, e]]);
				}, [[], []]);

				let [front, end] = players.reduce(function([p, f], e) {
					return (!previousPlayers.has(e.username) ? [[...p, e], f] : [p, [...f, e]]);
				}, [[], []]);

				front.sort(prioritySort);
				end.sort(prioritySort);

				connected = front.concat(end, spectateOnly);

				break;
			}
			case 'replace-losers': {
				let previousLosers = new Set(server.gameData.turnOrder.filter((e, i) => server.gameData.scores[i][0] <= server.gameData.settings.losingThreshold));
				let previousWinners = new Set(server.gameData.turnOrder.filter((e, i) => server.gameData.scores[i][0] > server.gameData.settings.losingThreshold));

				let [spectateOnly, players] = structuredClone(server.connected).reduce(function([s, p], e) {
					return (e.spectateOnly ? [[...s, e], p] : [s, [...p, e]]);
				}, [[], []]);

				let [front2mid, end] = players.reduce(function([p, f], e) {
					return (!previousLosers.has(e.username) ? [[...p, e], f] : [p, [...f, e]]);
				}, [[], []]);
				let [front, mid] = front2mid.reduce(function([p, f], e) {
					return (previousWinners.has(e.username) ? [[...p, e], f] : [p, [...f, e]]);
				}, [[], []]);

				front.sort(prioritySort);
				mid.sort(prioritySort);
				end.sort(prioritySort);

				connected = front.concat(mid, end, spectateOnly);

				break;
			}
			default: {
				let [spectateOnly, players] = structuredClone(server.connected).reduce(function([s, p], e) {
					return (e.spectateOnly ? [[...s, e], p] : [s, [...p, e]]);
				}, [[], []]);

				players.sort(prioritySort);
				connected = players.concat(spectateOnly);
			}
		}

		for (let i = 0; i < connected.length; i++) {
			connected[i].priority = i;
		}
		connected.sort(function(a, b) {
			return (a.username).localeCompare(b.username);
		});

		server.connected = connected;
	}

	generateTurnOrder(server) {
		let connected = structuredClone(server.connected).filter(user => !user.spectateOnly).sort(function(a, b) {
			return a.priority - b.priority;
		}).map(user => user.username);
		return Utils.shuffleArray(connected.slice(0, server.gameData.maxPlayers), server.dealRngFn || undefined);
	}

	enqueueCommand(data, ws, server) {
		server.commandQueue.push([data, ws]);
		this.processNextCommand(server);
	}

	processNextCommand(server) {
		if (server.processingCommand) return;

		server.processingCommand = true;

		while (server.commandQueue.length) {
			let [data, ws] = server.commandQueue.shift();

			if (data.currentFrame != server.gameData.currentFrame) {
				ws.send(Utils.JSONStringify({tag: 'commandNACK', data: {command: data.data, oldFrame: data.currentFrame, newFrame: server.gameData.currentFrame}, timestamp: Date.now()}));
				continue;
			}

			let res = this.processCommand(data.data, ws, server);
			let resToAll = structuredClone(res);
			resToAll.data = res.data.filter(e => e.toAll).map(e => e.msg);
			resToAll.timestamp = Date.now();
			res.data = res.data.map(e => e.msg);
			res.timestamp = Date.now();

			ws.send(Utils.JSONStringify({tag: 'commandACK', data: {command: data.data, oldFrame: data.currentFrame, newFrame: server.gameData.currentFrame}, timestamp: Date.now()}));

			ws.send(Utils.JSONStringify(res));
			Utils.broadcastToConnected(this.users, server, resToAll, ws.username);
		}
		server.processingCommand = false;
	}
}
