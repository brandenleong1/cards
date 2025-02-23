import * as Utils from '../../utils/utils.js';
import * as GameUtils from '../../utils/game_utils.js';
import * as CommandParse from '../../utils/command_parse.js';

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
	stacks: [],
	losingThreshold: -1000,
	scores: [],
	round: 0,
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
	data.gameData = structuredClone(defaultSettings);
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

	for (let i = 0; i < server.connected.length; i++) {
		gameData.hands.push(new Array());
		for (let j = 0; j < 2; j++) gameData.hands[i].push(new Array());
		gameData.scores.push(0);
	}
	gameData.turnOrder = Utils.shuffleArray(server.connected);

	gameData.gameState = 'LEADERBOARD';
	gameData.round = 1;
	// gameOFL(server);
}

export async function gameServerNSL(server) {
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
	// gameOFL(server);
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
		if (server.gameData.hands.every(e => !e.flat().length)) {
			server.gameData.gameState = 'SCORE';
		} else {
			server.gameData.gameState = 'PLAY_1';
		}
	} else if (server.gameData.gameState == 'LEADERBOARD') {
		server.gameData.gameState = 'DEAL_3';
	}
	// gameOFL(server);
}

function gameOFL(server) { // TODO OFL (may not even need this?)
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

export function processCommand(data, ws, server) {
	let command = CommandParse.parseCommand(data);
	console.log(command);
	let ret = [];
	let status = 1;

	switch (command.command[0].toLowerCase()) {
		case 'help':
			let str = '';
			str += 'HELP - display help menu\n';
			if (ws.username == server.host) str += 'EXIT - exit back to lobby\n';
			if (ws.username == server.host) str += 'DEAL - start round\n';
			str += 'SORT - sorts hand in the specified order\n';
				str += '\t- unspecified cards retain their order\n';
				str += '\tSORT [order]\n';
					str += '\t\te.g. SORT "1 2 7 3 0"\n';
			str += 'SWAP - swap two cards in your hand\n';
				str += '\tSWAP [idxA] [idxB]\n';
					str += '\t\te.g. SWAP 5 6\n';
			str += 'PLAY - play card(s)\n';
				str += '\t- can also be used in the "SHOW" phase to show cards\n';
				str += '\tPLAY [cards]\n';
				str += '\t\te.g. PLAY "4 1"\n';
			str += 'PASS - pass a play (in the "SHOW" phase)\n';
			str += 'DEBUG - show debug elements\n';
			ret.push(str.slice(0, -1));
			break;
		case 'exit':	// TODO
			break;
		case 'deal':	// TODO
			if (ws.username == server.host) {
				gameClientNSL(server);
				let serverInfo = structuredClone(server);
				let gameData = structuredClone(server.gameData); // TODO - obfuscate
				delete serverInfo.gameData;
				broadcastToConnected(server, {
					tag: 'updateGUI',
					data: {gameData: gameData, serverData: serverInfo}
				});
				break;
			}
		case 'sort':	// TODO
			break;
		case 'swap':	// TODO
			break;
		case 'play':	// TODO
			break;
		case 'pass':	// TODO
			break;
		case 'debug':
			ws.send(JSON.stringify({tag: 'toggleDebug', data: ret}));
			break;
		default:
			ret.push('Unknown command [' + command.command + ']');
			status = 0;
	}

	ws.send(JSON.stringify({tag: 'receiveCommand', status: status, data: ret}));
}

export function broadcastToConnected(server, data) {
	for (let username of server.connected) {
		let ws = users[username];
		ws.send(JSON.stringify(data));
	}
}
