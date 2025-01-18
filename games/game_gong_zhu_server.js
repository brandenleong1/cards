// const Utils = require(path.resolve(__dirname, '../utils.js')).Utils;

import * as Utils from '../utils/utils.js';
import * as GameUtils from '../utils/game_utils.js';

export let users = new Map(); // Username => WS
let usernames = new Map();

export let servers = [];

export const defaultSettings = {
	gameState: '',
	decks: [],
	numDecks: 1,
	minPlayers: 4,
	maxPlayers: 4,
	turnOrder: [],
	hands: [],
	discard: [],
	losingThreshold: -1000,
	scores: [],
};


export function addUser(username) {
	if (usernames[username]) {
		if (usernames[username] == 1000) return [0, 'Username saturated'];
		else {
			while (true) {
				let t = Math.floor(Math.random() * 10000).toString(10).padStart(4, '0');
				let user = username + '#' + t;
				if (!users.has(user)) {
					usernames[username]++;
					return [1, user];
				}
			}
		}
	} else {
		let t = Math.floor(Math.random() * 10000).toString(10).padStart(4, '0');
		let user = username + '#' + t;
		usernames[username] = 1;
		return [1, user];
	}
}

export function removeUser(username) {
	if (!username) return;
	usernames[username.split('#', 1)[0]] -= 1;
	users.delete(username);
}

export function addServer(data) {
	data.connected = [];
	data.gameData = {};
	let idx = Utils.binaryInsert(servers, data, function(a, b) {
		if (a.time != b.time) return b.time - a.time;
		else if (a.name != b.name) return a.name.localeCompare(b.name);
		else return a.creator.localeCompare(b.creator);
	});

	// console.log(servers);

	return [idx == -1 ? 0 : 1, data];
}

export function getServerIdx(serverData) {
	return Utils.binarySearchIdx(servers, serverData, function(a, b) {
		if (a.time != b.time) return b.time - a.time;
		else if (a.name != b.name) return a.name.localeCompare(b.name);
		else return a.creator.localeCompare(b.creator);
	});
}

export function joinServer(ws, serverData) {
	// console.log(serverData);
	let idx = getServerIdx(serverData);

	if (idx == -1) return [0, 'Server does not exist'];
	else {
		Utils.binaryInsert(servers[idx].connected, ws.username, function(a, b) {
			return a.localeCompare(b);
		});
		// console.log(servers[idx].connected);
		// console.log('joinServer', ws, serverData);
		ws.connected = servers[idx];
		return [1, servers[idx]];
	}
}

export function leaveServer(ws, serverData) {
	let idx = getServerIdx(serverData);

	if (idx == -1) return [0, 'Server does not exist'];
	else {
		let idx2 = Utils.binarySearchIdx(servers[idx].connected, ws.username, function(a, b) {
			return a.localeCompare(b);
		});
		if (idx2 != -1) servers[idx].connected.splice(idx2, 1);
		if (!servers[idx].connected.length) {
			servers.splice(idx, 1);
			return [1, null];
		}
		if (servers[idx].host == ws.username) {
			let hostIdx = Math.floor(Math.random() * servers[idx].connected.length);
			servers[idx].host = servers[idx].connected[hostIdx];
		}
		return [1, servers[idx]];
	}
}

export function initGame(server) {
	let gameData = server.gameData;
	for (let i = 0; i < gameData.numDecks; i++) {
		gameData.decks.push(GameUtils.initDeck());
		gameData.decks[i] = Utils.shuffleArray(gameData.decks[i]);
	}

	let turnOrder = new Array(server.connected.length).fill(0).map((e, i) => i);
	for (let i = 0; i < server.connected.length; i++) {
		gameData.hands.push(new Array());
		gameData.scores.push(0);
	}
	gameData.turnOrder = Utils.shuffleArray(turnOrder);

	gameData.gameState = 'DEAL_3';
	gameOFL(server);
}

async function gameServerNSL(server) {
	if (server.gameData.gameState == 'DEAL_3') {
		await Utils.sleep(2000);
		server.gameData.gameState = 'SHOW_3';
	} else if (server.gameData.gameState == 'DEAL_ALL') {
		await Utils.sleep(2000);
		server.gameData.gameState = 'SHOW_ALL';
	} else if (server.gameData.gameState == 'SCORE') {
		await Utils.sleep(10000);
		if (server.gameData.scores.every(e => e > server.gameData.losingThreshold)) {
			server.gameData.gameState = 'DEAL_3';
		} else {
			server.gameData.gameState = 'LEADERBOARD';
		}
	}
	gameOFL(server);
}

export function gameClientNSL(server) {
	if (server.gameData.gameState == 'SHOW_3') {
		server.gameData.gameState = 'DEAL_ALL';
	} else if (server.gameData.gameState == 'SHOW_ALL') {
		server.gameData.gameState = 'PLAY_1';
	} else if (server.gameData.gameState == 'PLAY_1') {
		server.gameData.gameState = 'PLAY_2';
	} else if (server.gameData.gameState == 'PLAY_2') {
		server.gameData.gameState = 'PLAY_3';
	} else if (server.gameData.gameState == 'PLAY_3') {
		server.gameData.gameState = 'PLAY_4';
	} else if (server.gameData.gameState == 'PLAY_4') {
		if (server.gameData.hands.every(e => !e.length)) {
			server.gameData.gameState = 'SCORE';
		} else {
			server.gameData.gameState = 'PLAY_1';
		}
	} else if (server.gameData.gameState == 'LEADERBOARD') {
		server.gameData.gameState = 'DEAL_3';
	}
	gameOFL(server);
}

function gameOFL(server) { // TODO OFL
	let state = server.gameData.gameState;
	if (state == 'DEAL_3') {

	} else if (state == 'SHOW_3') {

	} else if (state == 'DEAL_ALL') {

	} else if (state == 'SHOW_ALL') {

	} else if (state == 'PLAY_1') {

	} else if (state == 'PLAY_2') {

	} else if (state == 'PLAY_3') {

	} else if (state == 'PLAY_4') {

	} else if (state == 'SCORE') {

	} else if (state == 'LEADERBOARD') {

	}
}