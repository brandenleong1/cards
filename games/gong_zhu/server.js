import * as Utils from '../../utils/utils.js';
import * as GameUtils from '../../utils/game_utils.js';
import * as CommandParse from '../../utils/command_parse.js';

export let users = new Map(); // Username => WS
let usernames = new Map();

export let servers = [];

export const defaultSettings = {
	gameState: '',
	numDecks: 1,
	minPlayers: 4,
	maxPlayers: 4,
	losingThreshold: -1000,
	decks: [],
	turnOrder: [],
	needToAct: [],
	hands: [],					// [hidden, shown, played] * numPlayers
	stacks: [[], []],			// discard, [shown, val]
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

export function clearGameData(server) {
	let properties = [
		'gameState',
		'decks',
		'turnOrder',
		'needToAct',
		'hands',
		'stacks',
		'scores'
	];
	for (let property of properties) {
		server.gameData[property] = structuredClone(defaultSettings[property]);
	}
}

export function initGame(server) {
	clearGameData(server);
	let gameData = server.gameData;

	for (let i = 0; i < gameData.numDecks; i++) {
		gameData.decks.push(GameUtils.initDeck());
		gameData.decks[i] = Utils.shuffleArray(gameData.decks[i]);
	}

	for (let i = 0; i < server.connected.length; i++) {
		gameData.hands.push(new Array());
		for (let j = 0; j < 3; j++) gameData.hands[i].push(new Array());
		gameData.scores.push(0);
		gameData.needToAct.push(0);
	}
	gameData.turnOrder = Utils.shuffleArray(server.connected);

	gameData.gameState = 'LEADERBOARD';
	gameData.round = 1;
	gameOFL(server);
}

function gameNSL(server) {
	if (server.gameData.gameState == 'SHOW_3') {
		server.gameData.gameState = 'SHOW_ALL';
	} else if (server.gameData.gameState == 'SHOW_ALL') {
		server.gameData.gameState = 'PLAY_0';
	} else if (server.gameData.gameState == 'PLAY_0') {
		server.gameData.gameState = 'PLAY_1';
	} else if (server.gameData.gameState == 'PLAY_1') {
		server.gameData.gameState = 'PLAY_2';
	} else if (server.gameData.gameState == 'PLAY_2') {
		server.gameData.gameState = 'PLAY_3';
	} else if (server.gameData.gameState == 'PLAY_3') {
		if (server.gameData.hands.every(e => !e.flat().length)) {
			server.gameData.gameState = 'SCORE';
		} else {
			server.gameData.gameState = 'PLAY_0';
		}
	} else if (server.gameData.gameState == 'SCORE') {
		if (server.gameData.scores.every(e => e > server.gameData.losingThreshold)) {
			server.gameData.gameState = 'DEAL_3';
		} else {
			server.gameData.gameState = 'LEADERBOARD';
		}
	}  else if (server.gameData.gameState == 'LEADERBOARD') {
		server.gameData.gameState = 'SHOW_3';
	}

	gameOFL(server);
}

function gameOFL(server) { // TODO OFL (may not even need this?)
	let state = server.gameData.gameState;
	if (state == 'SHOW_3') {
		for (let i = 0; i < server.gameData.turnOrder.length; i++) {
			for (let j = 0; j < 3; j++) server.gameData.hands[i][0].push(server.gameData.decks[0].pop());
			server.gameData.needToAct[i] = 1;
		}
	} else if (state == 'SHOW_ALL') {

	} else if (state == 'PLAY_0') {

	} else if (state == 'PLAY_1') {

	} else if (state == 'PLAY_2') {

	} else if (state == 'PLAY_3') {

	} else if (state == 'SCORE') {

	} else if (state == 'LEADERBOARD') {

	}

	Utils.broadcastGameStateToConnected(users, server);
}

export function processCommand(data, ws, server) {
	let command = CommandParse.parseCommand(data);
	let commandUpper = command.command[0].toUpperCase();
	console.log(command);
	let ret = [];
	let status = 1;

	let myIdx = server.gameData.turnOrder.indexOf(ws.username);
	switch (command.command[0].toLowerCase()) {
		case 'help':
			if (command.command.length > 1) {
				ret.push(['Too many arguments for [' + commandUpper + '] (need 0)', 0]);
				status = 0;
				break;
			}
			let str = '';
			str += 'HELP - display help menu\n';
			str += 'EXIT - exit back to lobby\n';
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
			str += 'CLEAR - clears the console\n';
				str += '\talias CLR\n';
			str += 'DEBUG - show debug elements\n';
			ret.push([str.slice(0, -1), 0]);
			break;
		case 'exit':	// TODO
			if (command.command.length > 1) {
				ret.push(['Too many arguments for [' + commandUpper + '] (need 0)', 0]);
				status = 0;
			} else {
				Utils.broadcastToConnected(users, server,
					{tag: 'broadcastedMessage', data: '[' + ws.username + '] exited to lobby'}
				);
				Utils.broadcastToConnected(users, server,
					{tag: 'joinedLobby', status: 1, data: server}
				);
			}
			break;
		case 'deal':
			if (ws.username == server.host) {
				if (command.command.length > 1) {
					ret.push(['Too many arguments for [' + commandUpper + '] (need 0)', 0]);
					status = 0;
					break;
				}
				if (server.gameData.gameState == 'LEADERBOARD' || server.gameData.gameState == 'SCORE') {
					gameNSL(server);
					ret.push(['Started round ' + server.gameData.round, 1]);
				} else {
					ret.push(['Cannot issue command [' + commandUpper + '] in state [' + server.gameData.gameState + ']', 0]);
					status = 0;
				}
			} else {
				ret.push(['Unknown command [' + command.command[0] + ']', 0]);
				status = 0;
			}
			break;
		case 'sort':
			if (server.gameData.gameState == 'LEADERBOARD' || server.gameData.gameState == 'SCORE') {
				ret.push(['Cannot issue command [' + commandUpper + '] in state [' + server.gameData.gameState + ']', 0]);
				status = 0;
			} else if (command.command.length < 2) {
				ret.push(['Insufficient arguments for [' + commandUpper + '] (need 1)', 0]);
				status = 0;
			} else if (command.command.length > 2) {
				ret.push(['Too many arguments for [' + commandUpper + '] (need 1)', 0]);
				status = 0;
			} else {
				let arg1 = command.command[1].trim().split(/\s+/g).map(e => parseInt(e, 10));
				let invalidArg = arg1.find(e => isNaN(e) || e < 0 || e >= server.gameData.hands[myIdx].length);
				if (invalidArg != undefined) {
					ret.push(['Invalid argument at index 1 for [' + commandUpper + '] (subargument "' + invalidArg + '")', 0]);
					status = 0;
					break;
				} else if ((new Set(arg1)).size != arg1.length) {
					ret.push(['Invalid argument at index 1 for [' + commandUpper + '] (duplicate subarguments)', 0]);
					status = 0;
					break;
				}

				server.gameData.hands[myIdx][0] = Utils.sortArray(server.gameData.hands[myIdx][0], arg1);
				Utils.broadcastGameState(ws, server);
			}
			break;
		case 'swap':	// TODO
			break;
		case 'play':	// TODO
			if (
				server.gameData.gameState != 'SHOW_3' &&
				server.gameData.gameState != 'SHOW_ALL' &&
				server.gameData.gameState != ('PLAY_' + myIdx)
			) {
				ret.push(['Cannot issue command [' + commandUpper + '] in state [' + server.gameData.gameState + ']', 0]);
				status = 0;
			} else if (command.command.length < 2) {
				ret.push(['Insufficient arguments for [' + commandUpper + '] (need 1)', 0]);
				status = 0;
			} else if (command.command.length > 2) {
				ret.push(['Too many arguments for [' + commandUpper + '] (need 1)', 0]);
				status = 0;
			} else {
				let arg1 = command.command[1].trim().split(/\s+/g).map(e => parseInt(e, 10));
				let invalidArg = arg1.find(e => isNaN(e) || e < 0 || e >= server.gameData.hands[myIdx].length);
				if (invalidArg != undefined) {
					ret.push(['Invalid argument at index 1 for [' + commandUpper + '] (subargument "' + invalidArg + '")', 0]);
					status = 0;
					break;
				}

				if (server.gameData.gameState == 'SHOW_3' || server.gameData.gameState == 'SHOW_ALL') {
					let invalidArg = arg1.find(e => [11, 13, 36, 48].indexOf(server.gameData.hands[myIdx][0][e]) == -1 || server.gameData.stacks[1].findIndex(e1 => e1 == server.gameData.hands[myIdx][0][e]) != -1);
					if (invalidArg != undefined) {
						ret.push(['Invalid argument at index 1 for [' + commandUpper + '] (subargument "' + invalidArg + '")', 0]);
						status = 0;
						break;
					}

					let cards = arg1.map(e => server.gameData.hands[myIdx][0][e]);
					let val = server.gameData.gameState == 'SHOW_3' ? 4 : 2;
					for (let e of cards) {
						if (server.gameData.stacks[1].indexOf(e) == -1) {
							server.gameData.stacks[1].push([e, val]);
							server.gameData.hands[myIdx][1].push(e);
							ret.push(['Shown card [' + GameUtils.card2Str(e) + '] for x' + val + ' value', 0]);
						}
					}

					server.gameData.needToAct[myIdx] = 0;

					if (server.gameData.needToAct.every(e => e == 0)) {
						gameNSL(server);
						break;
					}
				}

				Utils.broadcastGameStateToConnected(users, server);
			}
			break;
		case 'pass':
			if (command.command.length > 1) {
				ret.push(['Too many arguments for [' + commandUpper + '] (need 0)', 0]);
				status = 0;
			}	else if (
				server.gameData.gameState != 'SHOW_3' &&
				server.gameData.gameState != 'SHOW_ALL' &&
				server.gameData.gameState != ('PLAY_' + myIdx)
			) {
				ret.push(['Cannot issue command [' + commandUpper + '] in state [' + server.gameData.gameState + ']', 0]);
				status = 0;
			} else {
				server.gameData.needToAct[myIdx] = 0;
				Utils.broadcastGameStateToConnected(users, server);
			}
			break;
		case 'clear':
		case 'clr':
			if (command.command.length > 1) {
				ret.push(['Too many arguments for [' + commandUpper + '] (need 0)', 0]);
				status = 0;
			} else {
				ws.send(JSON.stringify({tag: 'clearConsole'}));
			}
			break;
		case 'debug':
			ws.send(JSON.stringify({tag: 'toggleDebug', data: ret}));
			break;
		default:
			ret.push(['Unknown command [' + commandUpper + ']', 0]);
			status = 0;
	}

	return {tag: 'receiveCommand', status: status, data: ret};
}
