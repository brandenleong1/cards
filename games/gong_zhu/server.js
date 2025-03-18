import * as Utils from '../../utils/utils.js';
import * as GameUtils from '../../utils/game_utils.js';
import * as CommandParse from '../../utils/command_parse.js';

export let users = new Map();		// Username => WS
export let usernames = new Map();	// Username => Count

export let servers = [];

export const defaultSettings = {
	gameState: '',
	numDecks: 1,
	minPlayers: 4,
	maxPlayers: 4,
	decks: [],
	turnOrder: [],
	turnFirstIdx: 0,
	needToAct: [],
	hands: [],					// [[hidden + shown, shown, collected, played], * numPlayers]
	stacks: [[], []],			// [discard, [shown, val]]
	scores: [],
	round: 0,
	settings: {
		losingThreshold: -1000,
		expose3: false,
		zhuYangManJuan: false
	}
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
		usernames.set(username, 1);
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
	let idx = getServerIdx(serverData);

	if (idx == -1) return [0, 'Server does not exist'];
	else {
		Utils.binaryInsert(servers[idx].connected, ws.username, function(a, b) {
			return a.localeCompare(b);
		});
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

export function updateServerSettings(serverData, settings) {
	let idx = getServerIdx(serverData);
	if (idx == -1) return [0, 'Server does not exist'];
	else {
		for (let property in settings) {
			servers[idx].gameData.settings[property] = settings[property];
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

function resetRoundData(server) {
	let properties = [
		'decks',
		'hands',
		'stacks'
	];
	for (let property of properties) {
		server.gameData[property] = structuredClone(defaultSettings[property]);
	}

	for (let i = 0; i < server.gameData.numDecks; i++) {
		server.gameData.decks.push(GameUtils.initDeck());
		server.gameData.decks[i] = Utils.shuffleArray(server.gameData.decks[i]);
	}

	for (let i = 0; i < server.connected.length; i++) {
		server.gameData.hands.push(new Array());
		for (let j = 0; j < 4; j++) server.gameData.hands[i].push(new Array());
	}

	server.gameData.scores.forEach(e => e[1] = 0);
	server.gameData.stacks.forEach(e => e.length = 0);
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
		for (let j = 0; j < 4; j++) gameData.hands[i].push(new Array());
		gameData.scores.push(new Array());
		for (let j = 0; j < 2; j++) gameData.scores[i].push(0);
		gameData.needToAct.push(0);
	}
	gameData.turnOrder = Utils.shuffleArray(server.connected);

	gameData.gameState = 'LEADERBOARD';
	gameData.round = 0;
	gameData.turnFirstIdx = 0;

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
		if (server.gameData.hands.every(e => !e[0].length)) {
			if (server.gameData.scores.some(e => e[0] <= server.gameData.settings.losingThreshold)) {
				server.gameData.gameState = 'LEADERBOARD';
			} else {
				server.gameData.gameState = 'SCORE';
			}
		} else {
			server.gameData.gameState = 'PLAY_0';
		}
	} else if (server.gameData.gameState == 'SCORE') {
		if (server.gameData.settings.expose3) {
			server.gameData.gameState = 'SHOW_3';
		} else {
			server.gameData.gameState = 'SHOW_ALL';
		}
	}  else if (server.gameData.gameState == 'LEADERBOARD') {
		if (server.gameData.settings.expose3) {
			server.gameData.gameState = 'SHOW_3';
		} else {
			server.gameData.gameState = 'SHOW_ALL';
		}
	}

	return gameOFL(server);
}

function gameOFL(server) {
	let gameData = server.gameData;
	let state = gameData.gameState;
	let ret = [];

	if (state == 'SHOW_3') {
		gameData.round += 1;
		resetRoundData(server);

		for (let i = 0; i < gameData.turnOrder.length; i++) {
			for (let j = 0; j < 3; j++) gameData.hands[i][0].push(gameData.decks[0].pop());
			gameData.needToAct[i] = 1;
		}
	} else if (state == 'SHOW_ALL') {
		if (!gameData.settings.expose3) {
			gameData.round += 1;
			resetRoundData(server);
		}

		while (gameData.decks[0].length) {
			for (let i = 0; i < gameData.turnOrder.length; i++) {
				gameData.hands[i][0].push(gameData.decks[0].pop());
			}
		}

		for (let i = 0; i < gameData.turnOrder.length; i++) {
			gameData.needToAct[i] = 1;
		}
	} else if (state == 'PLAY_0') {
		if (!gameData.stacks[0].length && gameData.hands.every(e => !e[3].length)) {
			gameData.turnFirstIdx = gameData.hands.findIndex(e => e[0].includes(1));
		} else {
			let winnerIdx = gameData.turnFirstIdx;
			let winnerRank = gameData.hands[gameData.turnFirstIdx][3][0] % 13;
			let trickSuit = Math.floor(gameData.hands[gameData.turnFirstIdx][3][0] / 13);
			for (let i = 0; i < gameData.turnOrder.length; i++) {
				let mySuit = Math.floor(gameData.hands[i][3][0] / 13);
				let myRank = gameData.hands[i][3][0] % 13;

				if (mySuit == trickSuit) {
					if (myRank == 0 || (winnerRank != 0 && myRank > winnerRank)) {
						winnerIdx = i;
						winnerRank = myRank;
					}
				}
			}

			gameData.turnFirstIdx = winnerIdx;

			let played = new Set(gameData.hands.map(e => e[3]).flat());
			let important = played.intersection(new Set([11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 36, 48]));
			important.forEach(e => gameData.hands[winnerIdx][2].push(e));
			gameData.scores[winnerIdx][1] = scoreFromCards(gameData.hands[winnerIdx][2], server);

			ret.push(['Player [' + gameData.turnOrder[gameData.turnFirstIdx] + '] wins with [' + GameUtils.card2Str(gameData.hands[winnerIdx][3][0]) + '] and takes [' + [...important].map(e => GameUtils.card2Str(e)).join(', ') + ']', 1]);

			gameData.hands.forEach(e => gameData.stacks[0].push(e[3].pop()));

			gameData.hands.forEach(e => {
				for (let j = e[1].length - 1; j >= 0; j--) {
					if (Math.floor(e[1][j] / 13) == trickSuit) {
						e[1].splice(j, 1);
					}
				}
			});
		}
		ret.push(['Started Trick ' + (Math.round(gameData.stacks[0].length / 4) + 1) + '; Player [' + gameData.turnOrder[gameData.turnFirstIdx] + '] leads...', 1]);
	} else if (state == 'PLAY_1') {

	} else if (state == 'PLAY_2') {

	} else if (state == 'PLAY_3') {

	} else if (state == 'SCORE') {
		let winnerIdx = gameData.turnFirstIdx;
		let winnerRank = gameData.hands[gameData.turnFirstIdx][3][0] % 13;
		let trickSuit = Math.floor(gameData.hands[gameData.turnFirstIdx][3][0] / 13);
		for (let i = 0; i < gameData.turnOrder.length; i++) {
			let mySuit = Math.floor(gameData.hands[i][3][0] / 13);
			let myRank = gameData.hands[i][3][0] % 13;

			if (mySuit == trickSuit) {
				if (myRank == 0 || (winnerRank != 0 && myRank > winnerRank)) {
					winnerIdx = i;
					winnerRank = myRank;
				}
			}
		}

		gameData.turnFirstIdx = winnerIdx;

		let played = new Set(gameData.hands.map(e => e[3]).flat());
		let important = played.intersection(new Set([11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 36, 48]));
		important.forEach(e => gameData.hands[winnerIdx][2].push(e));
		gameData.scores[winnerIdx][1] = scoreFromCards(gameData.hands[winnerIdx][2], server);

		ret.push(['Player [' + gameData.turnOrder[gameData.turnFirstIdx] + '] wins with [' + GameUtils.card2Str(gameData.hands[winnerIdx][3][0]) + '] and takes [' + [...important].map(e => GameUtils.card2Str(e)).join(', ') + ']', 1]);

		gameData.hands.forEach(e => gameData.stacks[0].push(e[3].pop()));

		gameData.hands.forEach(e => {
			for (let j = e[1].length - 1; j >= 0; j--) {
				if (Math.floor(e[1][j] / 13) == trickSuit) {
					e[1].splice(j, 1);
				}
			}
		});

		gameData.scores.forEach((e, i) => {
			e[0] += e[1];
			ret.push(['Player [' + gameData.turnOrder[i] + '] receives ' + (e[1] > 0 ? '+' + e[1] : e[1]), 1]);
		});
	} else if (state == 'LEADERBOARD') {

	}

	Utils.broadcastGameStateToConnected(users, server, obfuscateGameData);
	return ret;
}

export function processCommand(data, ws, server) {
	let gameData = server.gameData;

	let command = CommandParse.parseCommand(data);
	let commandUpper = command.command[0].toUpperCase();
	console.log(command);
	let ret = [];
	let status = 1;

	let myIdx = gameData.turnOrder.indexOf(ws.username);
	let relativeIdx = ((myIdx - gameData.turnFirstIdx) % (gameData.turnOrder.length)) + (((myIdx - gameData.turnFirstIdx) % (gameData.turnOrder.length)) < 0 ? gameData.turnOrder.length : 0);
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
			str += 'SORT - sorts hand in the specified order, if given\n';
				str += '\t- unspecified cards retain their order\n';
				str += '\tSORT\n';
				str += '\t\talias SORT auto\n';
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
		case 'exit':
			if (command.command.length > 1) {
				ret.push(['Too many arguments for [' + commandUpper + '] (need 0)', 0]);
				status = 0;
			} else {
				server.gameData.gameState = '';
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
				if (gameData.gameState == 'LEADERBOARD' || gameData.gameState == 'SCORE') {
					ret.push(...gameNSL(server));
					ret.push(['Started Round ' + gameData.round, 1]);
				} else {
					ret.push(['Cannot issue command [' + commandUpper + '] in state [' + gameData.gameState + ']', 0]);
					status = 0;
				}
			} else {
				ret.push(['Unknown command [' + command.command[0] + ']', 0]);
				status = 0;
			}
			break;
		case 'sort':
			if (gameData.gameState == 'LEADERBOARD' || gameData.gameState == 'SCORE') {
				ret.push(['Cannot issue command [' + commandUpper + '] in state [' + gameData.gameState + ']', 0]);
				status = 0;
			} else if (command.command.length > 2) {
				ret.push(['Too many arguments for [' + commandUpper + '] (need max 1)', 0]);
				status = 0;
			} else {
				if (command.command.length == 1 || command.command[1].trim() == 'auto') {
					console.log(gameData.hands[myIdx][0].map(e => e + ' ' + GameUtils.card2Str(e)));
					gameData.hands[myIdx][0].sort((a, b) => {return a - b;});
					console.log(gameData.hands[myIdx][0].map(e => e + ' ' + GameUtils.card2Str(e)));
				} else {
					let arg1 = command.command[1].trim().split(/\s+/g).map(e => parseInt(e, 10));
					let invalidArgIdx = arg1.findIndex(e => isNaN(e) || e < 0 || e >= gameData.hands[myIdx][0].length);
					if (invalidArgIdx != -1) {
						ret.push(['Invalid argument at index 1 for [' + commandUpper + '] (subargument "' + arg1[invalidArgIdx] + '")', 0]);
						status = 0;
						break;
					} else if ((new Set(arg1)).size != arg1.length) {
						ret.push(['Invalid argument at index 1 for [' + commandUpper + '] (duplicate subarguments)', 0]);
						status = 0;
						break;
					}

					gameData.hands[myIdx][0] = Utils.sortArray(gameData.hands[myIdx][0], arg1);
				}

				Utils.broadcastGameState(ws, server, obfuscateGameData);
			}
			break;
		case 'swap':	// TODO swap
			break;
		case 'play':
			if (
				gameData.gameState != 'SHOW_3' &&
				gameData.gameState != 'SHOW_ALL' &&
				gameData.gameState != ('PLAY_' + relativeIdx)
			) {
				ret.push(['Cannot issue command [' + commandUpper + '] in state [' + gameData.gameState + ']', 0]);
				status = 0;
			} else if (command.command.length < 2) {
				ret.push(['Insufficient arguments for [' + commandUpper + '] (need 1)', 0]);
				status = 0;
			} else if (command.command.length > 2) {
				ret.push(['Too many arguments for [' + commandUpper + '] (need 1)', 0]);
				status = 0;
			} else {
				let arg1 = command.command[1].trim().split(/\s+/g).map(e => parseInt(e, 10));
				let invalidArgIdx = arg1.findIndex(e => isNaN(e) || e < 0 || e >= gameData.hands[myIdx][0].length);
				if (invalidArgIdx != -1) {
					ret.push(['Invalid argument at index 1 for [' + commandUpper + '] (subargument "' + arg1[invalidArgIdx] + '")', 0]);
					status = 0;
					break;
				}

				if (gameData.gameState == 'SHOW_3' || gameData.gameState == 'SHOW_ALL') {
					let invalidArgIdx = arg1.findIndex(e => [11, 13, 36, 48].indexOf(gameData.hands[myIdx][0][e]) == -1 || gameData.stacks[1].findIndex(e1 => e1 == gameData.hands[myIdx][0][e]) != -1);
					if (invalidArgIdx != -1) {
						ret.push(['Invalid argument at index 1 for [' + commandUpper + '] (subargument "' + arg1[invalidArgIdx] + '")', 0]);
						status = 0;
						break;
					}

					let cards = arg1.map(e => gameData.hands[myIdx][0][e]);
					let val = gameData.gameState == 'SHOW_3' ? 4 : 2;
					for (let e of cards) {
						if (gameData.stacks[1].indexOf(e) == -1) {
							gameData.stacks[1].push([e, val]);
							gameData.hands[myIdx][1].push(e);
							ret.push(['Shown card [' + GameUtils.card2Str(e) + '] for x' + val + ' value', 0]);
						}
					}
				} else {
					if (arg1.length != 1) {
						ret.push(['Invalid argument at index 1 for [' + commandUpper + '] (incompatible number of subarguments; need 1, got ' + arg1.length + ')', 0]);
						status = 0;
						break;
					}

					let playableCards = new Set();
					if (relativeIdx == 0) {
						gameData.hands[myIdx][0].filter(e => !gameData.hands[myIdx][1].includes(e)).forEach(e => playableCards.add(e));
						gameData.hands[myIdx][1].filter(e => GameUtils.filterBySuit(e, gameData.hands[myIdx][0]).length == 1).forEach(e => playableCards.add(e));
					} else {
						let filtered = GameUtils.filterBySuit(gameData.hands[gameData.turnFirstIdx][3][0], gameData.hands[myIdx][0]);
						if (filtered.length) {
							filtered.forEach(e => {
								if (!gameData.hands[myIdx][1].includes(e)) playableCards.add(e);
							});
						} else {
							gameData.hands[myIdx][0].forEach(e => playableCards.add(e));
						}
					}

					let invalidArgIdx = arg1.findIndex(e => !playableCards.has(gameData.hands[myIdx][0][e]));
					if (invalidArgIdx != -1) {
						ret.push(['Invalid argument at index 1 for [' + commandUpper + '] (subargument "' + arg1[invalidArgIdx] + '")', 0]);
						status = 0;
						break;
					}

					ret.push(['Player [' +  gameData.turnOrder[myIdx] + '] played card [' + GameUtils.card2Str(gameData.hands[myIdx][0][arg1[0]]) + ']', 1]);

					let shownIdx = gameData.hands[myIdx][1].findIndex(e => e == gameData.hands[myIdx][0][arg1[0]]);
					if (shownIdx != -1) gameData.hands[myIdx][1].splice(shownIdx, 1);
					gameData.hands[myIdx][3].push(...gameData.hands[myIdx][0].splice(arg1[0], 1));

					ret.push(...gameNSL(server));
				}

				Utils.broadcastGameStateToConnected(users, server, obfuscateGameData);
			}
			break;
		case 'pass':
			if (command.command.length > 1) {
				ret.push(['Too many arguments for [' + commandUpper + '] (need 0)', 0]);
				status = 0;
			} else if (
				gameData.gameState != 'SHOW_3' &&
				gameData.gameState != 'SHOW_ALL'
			) {
				ret.push(['Cannot issue command [' + commandUpper + '] in state [' + gameData.gameState + ']', 0]);
				status = 0;
			} else {
				gameData.needToAct[myIdx] = 0;
				if (gameData.needToAct.every(e => e == 0)) {
					ret.push(...gameNSL(server));
					break;
				}
				Utils.broadcastGameStateToConnected(users, server, obfuscateGameData);
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

function scoreFromCards(cardArr, server) {
	let c = 0;

	let cardSet = new Set(cardArr);
	let heartSet = new Set(new Array(13).fill(0).map((e, i) => i + 13));

	let modifiers = [11, 13, 36, 48].reduce((dict, e) => {
		let idx = server.gameData.stacks[1].findIndex(card => card[0] == e)
		dict[e] = (idx == -1 ? 1 : server.gameData.stacks[1][idx][1]);
		return dict;
	}, {});

	if (cardSet.isSupersetOf(heartSet)) {
		c += modifiers[13] * 200;

		if (server.gameData.settings.zhuYangManJuan) {
			if (cardSet.has(11) && cardSet.has(36)) c += modifiers[11] * 100;
		} else {
			if (cardSet.has(11)) c += modifiers[11] * 100;
		}
	} else {
		if (cardSet.has(11)) c += modifiers[11] * -100;
		if (cardSet.has(13)) c += modifiers[13] * -50;
		if (cardSet.has(25)) c += modifiers[13] * -40;
		if (cardSet.has(24)) c += modifiers[13] * -30;
		if (cardSet.has(23)) c += modifiers[13] * -20;
		for (let i = 17; i <= 22; i++) if (cardSet.has(i)) c += modifiers[13] * -10;
		if (cardSet.has(36)) c += modifiers[36] * 100;
	}


	if (cardSet.has(48)) {
		if (cardSet.size == 1) {
			c += modifiers[48] * 50;
		} else {
			c *= modifiers[48] * 2;
		}
	}

	return c;
}

export function obfuscateGameData(gameData, idx) {
	let gameDataCopy = structuredClone(gameData);

	Utils.nullify(gameDataCopy.decks);
	gameDataCopy.hands.forEach((hand, i) => {
		if (i != idx) Utils.nullify(hand[0]);
	});
	Utils.nullify(gameDataCopy.stacks[0]);
	gameDataCopy.stacks[1] = gameDataCopy.stacks[1].filter(e =>
		!((gameDataCopy.gameState == 'SHOW_3' && e[1] == 4) || (gameDataCopy.gameState == 'SHOW_ALL' && e[1] == 2))
	);

	return gameDataCopy;
}