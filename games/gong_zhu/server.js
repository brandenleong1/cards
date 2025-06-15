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
		spectatorPolicy: 'disallowed',
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

export function joinServer(ws, server) {
	let idx = getServerIdx(server);

	if (idx == -1) return [0, 'Server does not exist'];
	else if (servers[idx].gameData.settings.spectatorPolicy == 'disallowed' && servers[idx].gameData.maxPlayers <= servers[idx].connected.length) return [0, 'Server full (Spectators disallowed)'];
	else {
		let lowestPriority = Utils.getLowestPriority(servers[idx]);
		if (lowestPriority == -Infinity) lowestPriority = -1;
		Utils.binaryInsert(servers[idx].connected, {username: ws.username, priority: lowestPriority + 1}, function(a, b) {
			return (a.username).localeCompare(b.username);
		});
		ws.connected = servers[idx];
		return [1, servers[idx]];
	}
}

export function leaveServer(ws, server) {
	let idx = getServerIdx(server);

	if (idx == -1) return [0, 'Server does not exist'];
	else {
		let idx2 = Utils.binarySearchIdx(servers[idx].connected, {username: ws.username}, function(a, b) {
			return (a.username).localeCompare(b.username);
		});
		if (idx2 != -1) servers[idx].connected.splice(idx2, 1);
		if (!servers[idx].connected.length) {
			servers.splice(idx, 1);
			return [1, null];
		}
		if (servers[idx].host == ws.username) {
			let hostIdx = Math.floor(Math.random() * servers[idx].connected.length);
			servers[idx].host = servers[idx].connected[hostIdx].username;
		}
		Utils.updatePriorities(servers[idx]);
		return [1, servers[idx]];
	}
}

export function updateServerSettings(server, settings) {
	let idx = getServerIdx(server);
	if (idx == -1) return [0, 'Server does not exist'];
	else {
		for (let property in settings) {
			servers[idx].gameData.settings[property] = settings[property];
		}
		if (servers[idx].gameData.settings.spectatorPolicy == 'disallowed') removeAllSpectators(servers[idx]);
		return [1, servers[idx]];
	}
}

export function removeAllSpectators(server) {
	Utils.updatePriorities(server);
	let toRemove = Utils.getUsersSortedByPriority(server);
	let hostIdx = toRemove.findIndex(user => user.username == server.host);
	toRemove.splice(hostIdx, 1);
	toRemove.splice(0, server.gameData.maxPlayers - 1);

	console.log(toRemove);

	toRemove.forEach(function(user) {
		let ws = users.get(user.username);
		let res = leaveServer(ws, server);
		if (res[0]) {
			ws.send(JSON.stringify({tag: 'broadcastedMessage', data: 'Spectators kicked'}));
			ws.send(JSON.stringify({tag: 'leftLobby', status: res[0], data: res[1]}));
			delete ws.connected;
		}
	});

	server.connected.forEach(user => users.get(user.username).send(JSON.stringify({tag: 'showLobby', status: 1, data: server})));
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

	for (let i = 0; i < server.gameData.turnOrder.length; i++) {
		server.gameData.hands.push(new Array());
		for (let j = 0; j < 4; j++) server.gameData.hands[i].push(new Array());
	}

	server.gameData.scores.forEach(e => e[1] = 0);
	server.gameData.stacks.forEach(e => e.length = 0);
}

function rotateSpectators(server) {
	let connected = [];

	let prioritySort = function(a, b) {
		return a.priority - b.priority;
	};

	switch (server.gameData.settings.spectatorPolicy) {
		case 'round-robin': {
			let previousPlayers = new Set(server.gameData.turnOrder);

			let [front, end] = structuredClone(server.connected).reduce(function([p, f], e) {
				return (!previousPlayers.has(e.username) ? [[...p, e], f] : [p, [...f, e]]);
			}, [[], []]);

			front.sort(prioritySort);
			end.sort(prioritySort);

			connected = front.concat(end);

			break;
		}
		case 'replace-losers': {
			let previousLosers = new Set(server.gameData.turnOrder.filter((e, i) => server.gameData.scores[i][0] <= server.gameData.settings.losingThreshold));
			let previousWinners = new Set(server.gameData.turnOrder.filter((e, i) => server.gameDatas.scores[i][0] > server.gameData.settings.losingThreshold));

			let [front2mid, end] = structuredClone(server.connected).reduce(function([p, f], e) {
				return (!previousLosers.has(e.username) ? [[...p, e], f] : [p, [...f, e]]);
			}, [[], []]);
			let [front, mid] = structuredClone(front2mid).reduce(function([p, f], e) {
				return (previousWinners.has(e.username) ? [[...p, e], f] : [p, [...f, e]]);
			}, [[], []]);

			front.sort(prioritySort);
			mid.sort(prioritySort);
			end.sort(prioritySort);

			connected = front.concat(front, mid, end);

			break;
		}
		default: {
			connected = structuredClone(server.connected);
			connected.sort(prioritySort);
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

function generateTurnOrder(server) {
	let connected = structuredClone(server.connected).sort(function(a, b) {
		return a.priority - b.priority;
	}).map(user => user.username);
	return Utils.shuffleArray(connected.slice(0, server.gameData.maxPlayers));
}

export function initGame(server) {
	clearGameData(server);
	let gameData = server.gameData;

	for (let i = 0; i < gameData.numDecks; i++) {
		gameData.decks.push(GameUtils.initDeck());
		gameData.decks[i] = Utils.shuffleArray(gameData.decks[i]);
	}

	gameData.turnOrder = generateTurnOrder(server);

	for (let i = 0; i < server.gameData.turnOrder.length; i++) {
		gameData.hands.push(new Array());
		for (let j = 0; j < 4; j++) gameData.hands[i].push(new Array());
		gameData.scores.push(new Array());
		for (let j = 0; j < 2; j++) gameData.scores[i].push(0);
		gameData.needToAct.push(0);
	}

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
			// console.log(server.gameData.scores.map(e => (e[0] + e[1])));
			if (server.gameData.scores.some(e => (e[0] + e[1]) <= server.gameData.settings.losingThreshold)) {
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
		resetRoundData(server);
		gameData.round += 1;

		for (let i = 0; i < gameData.turnOrder.length; i++) {
			for (let j = 0; j < 3; j++) gameData.hands[i][0].push(gameData.decks[0].pop());
			gameData.needToAct[i] = 1;
		}
	} else if (state == 'SHOW_ALL') {
		if (!gameData.settings.expose3) {
			resetRoundData(server);
			gameData.round += 1;
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
		}
		ret.push({
			msg: 'Started Trick ' + (Math.round(gameData.stacks[0].length / 4) + 1) + '; Player [' + gameData.turnOrder[gameData.turnFirstIdx] + '] leads...',
			toAll: true
		});
	} else if (state == 'PLAY_1') {

	} else if (state == 'PLAY_2') {

	} else if (state == 'PLAY_3') {

	} else if (state == 'SCORE') {
		gameData.scores.forEach((e, i) => {
			ret.push({
				msg: 'Player [' + gameData.turnOrder[i] + '] receives ' + (e[1] > 0 ? '+' + e[1] : e[1]),
				toAll: true
			});
		});
	} else if (state == 'LEADERBOARD') {
		if (server.gameData.scores.some(e => e[0] <= server.gameData.settings.losingThreshold)) {
			for (let i = 0; i < server.gameData.turnOrder.length; i++) {
				ret.push({
					msg: 'Player [' + gameData.turnOrder[i] + '] ' + (server.gameData.scores[i][0] <= server.gameData.settings.losingThreshold ? 'loses' : 'survives') + ' ↦ ' + server.gameData.scores[i][0] + ' pts',
					toAll: true
				});
			}
		}
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
	let isSpectator = myIdx == -1;
	let relativeIdx = isSpectator ? -1 : ((myIdx - gameData.turnFirstIdx) % (gameData.turnOrder.length)) + (((myIdx - gameData.turnFirstIdx) % (gameData.turnOrder.length)) < 0 ? gameData.turnOrder.length : 0);

	switch (command.command[0].toLowerCase()) {
		case 'help': {
			if (command.command.length > 1) {
				ret.push({
					msg: 'Too many arguments for [' + commandUpper + '] (max 0)',
					toAll: false
				});
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
				str += '\tSORT [order]...\n';
					str += '\t\te.g. SORT 1 2 7 3 0\n';
			str += 'SWAP - swap cards in your hand\n';
				str += '\t- card at order[0] -> order[1], order[1] -> order[2], etc.';
				str += '\tSWAP [order]...\n';
					str += '\t\te.g. SWAP 5 6\n';
			str += 'PLAY - play card(s)\n';
				str += '\t- can also be used in the "SHOW" phase to show cards\n';
				str += '\tPLAY [cards]...\n';
				str += '\t\te.g. PLAY 4 1\n';
			str += 'PASS - pass a play (in the "SHOW" phase)\n';
			str += 'CLEAR - clears the console\n';
				str += '\talias CLR\n';
			str += 'DEBUG - show debug elements\n';
			ret.push({
				msg: str.slice(0, -1),
				toAll: false
			});
			break;
		}
		case 'exit': {
			if (command.command.length > 1) {
				ret.push({
					msg: 'Too many arguments for [' + commandUpper + '] (need 0)',
					toAll: false
				});
				status = 0;
			} else {
				if (!isSpectator || ws.username == server.host) {
					server.gameData.gameState = '';
					Utils.broadcastToConnected(users, server,
						{tag: 'broadcastedMessage', data: '[' + ws.username + '] exited to lobby'}
					);
					Utils.broadcastToConnected(users, server,
						{tag: 'showLobby', status: 1, data: server}
					);
				} else {
					let res = leaveServer(ws, server);
					ws.send(JSON.stringify({tag: 'leftLobby', status: res[0], data: res[1]}));

					if (res[0]) delete ws.connected;
				}
			}
			break;
		}
		case 'deal': {
			if (ws.username == server.host) {
				if (command.command.length > 1) {
					ret.push({
						msg: 'Too many arguments for [' + commandUpper + '] (max 0)',
						toAll: false
					});
					status = 0;
					break;
				}
				if (gameData.gameState == 'LEADERBOARD' || gameData.gameState == 'SCORE') {
					if (gameData.gameState == 'LEADERBOARD' && gameData.scores.some(e => e[0] <= gameData.settings.losingThreshold)) {
						let previousTurnOrder = new Set(gameData.turnOrder);

						rotateSpectators(server);
						gameData.turnOrder = generateTurnOrder(server);

						let newTurnOrder = new Set(gameData.turnOrder);

						for (let username of previousTurnOrder) {
							if (!newTurnOrder.has(username)) users.get(username).send(JSON.stringify({tag: 'broadcastedMessage', data: 'You are now spectating!'}));
						}
						for (let username of newTurnOrder) {
							if (!previousTurnOrder.has(username)) users.get(username).send(JSON.stringify({tag: 'broadcastedMessage', data: 'You are now playing!'}));
						}
					}

					console.log('deal', server);

					ret.push(...gameNSL(server));
					ret.push({
						msg: 'Started Round ' + gameData.round,
						toAll: true
					});
				} else {
					ret.push({
						msg: 'Cannot issue command [' + commandUpper + '] in state [' + gameData.gameState + ']',
						toAll: false
					});
					status = 0;
				}
			} else {
				ret.push({
					msg: 'Unknown command [' + command.command[0] + ']',
					toAll: false
				});
				status = 0;
			}
			break;
		}
		case 'sort': {
			if (isSpectator) {
				ret.push({
					msg: 'Cannot issue command [' + commandUpper + '] as a specator',
					toAll: false
				});
				status = 0;
			} else if (gameData.gameState == 'LEADERBOARD' || gameData.gameState == 'SCORE') {
				ret.push({
					msg: 'Cannot issue command [' + commandUpper + '] in state [' + gameData.gameState + ']',
					toAll: false
				});
				status = 0;
			} else {
				if (command.command.length == 1 || command.command[1].trim() == 'auto') {
					gameData.hands[myIdx][0].sort((a, b) => a - b);
				} else {
					let args = command.command.slice(1).map(e => parseInt(e, 10));
					let invalidArgIdx = args.findIndex(e => isNaN(e) || e < 0 || e >= gameData.hands[myIdx][0].length);
					let duplicateIdx = args.findIndex((e, i) => args.indexOf(e) != i);
					if (invalidArgIdx != -1) {
						ret.push({
							msg: 'Invalid argument at index [' + (invalidArgIdx + 1) + '] for [' + commandUpper + '] (argument "' + command.command[invalidArgIdx + 1] + '")',
							toAll: false
						});
						status = 0;
						break;
					} else if (duplicateIdx != -1) {
						ret.push({
							msg: 'Invalid argument at index [' + (duplicateIdx + 1) + '] for [' + commandUpper + '] (duplicate arguments)',
							toAll: false
						});
						status = 0;
						break;
					}

					gameData.hands[myIdx][0] = Utils.sortArrayFromIndices(gameData.hands[myIdx][0], args);
				}

				Utils.broadcastGameState(ws, server, obfuscateGameData);
			}
			break;
		}
		case 'swap': {
			if (isSpectator) {
				ret.push({
					msg: 'Cannot issue command [' + commandUpper + '] as a specator',
					toAll: false
				});
				status = 0;
			} else if (gameData.gameState == 'LEADERBOARD' || gameData.gameState == 'SCORE') {
				ret.push({
					msg: 'Cannot issue command [' + commandUpper + '] in state [' + gameData.gameState + ']',
					toAll: false
				});
				status = 0;
			} else if (command.command.length < 2) {
				ret.push({
					msg: 'Insufficient arguments for [' + commandUpper + '] (need 1)',
					toAll: false
				});
				status = 0;
			} else {
				let args = command.command.slice(1).map(e => parseInt(e, 10));
				let invalidArgIdx = args.findIndex(e => isNaN(e) || e < 0 || e >= gameData.hands[myIdx][0].length);
				let duplicateIdx = args.findIndex((e, i) => args.indexOf(e) != i);
				if (invalidArgIdx != -1) {
					ret.push({
						msg: 'Invalid argument at index [' + (invalidArgIdx + 1) + '] for [' + commandUpper + '] (argument "' + command.command[invalidArgIdx + 1] + '")',
						toAll: false
					});
					status = 0;
					break;
				} else if (duplicateIdx != -1) {
					ret.push({
						msg: 'Invalid argument at index [' + (duplicateIdx + 1) + '] for [' + commandUpper + '] (duplicate arguments)',
						toAll: false
					});
					status = 0;
					break;
				}

				let swap = (arr, i, j) => {[arr[i], arr[j]] = [arr[j], arr[i]];};
				swap(gameData.hands[myIdx][0], args[0], args[args.length - 1]);
				for (let i = args.length - 1; i >= 2; i--) {
					swap(gameData.hands[myIdx][0], args[i], args[i - 1]);
				}

				Utils.broadcastGameState(ws, server, obfuscateGameData);
			}
			break;
		}
		case 'play': {
			if (isSpectator) {
				ret.push({
					msg: 'Cannot issue command [' + commandUpper + '] as a specator',
					toAll: false
				});
				status = 0;
			} else if (
				gameData.gameState != 'SHOW_3' &&
				gameData.gameState != 'SHOW_ALL' &&
				gameData.gameState != ('PLAY_' + relativeIdx)
			) {
				ret.push({
					msg: 'Cannot issue command [' + commandUpper + '] in state [' + gameData.gameState + ']',
					toAll: false
				});
				status = 0;
			} else if (command.command.length < 2) {
				ret.push({
					msg: 'Insufficient arguments for [' + commandUpper + '] (need 1)',
					toAll: false
				});
				status = 0;
			} else {
				let args = command.command.slice(1).map(e => parseInt(e, 10));
				let invalidArgIdx = args.findIndex(e => isNaN(e) || e < 0 || e >= gameData.hands[myIdx][0].length);
				if (invalidArgIdx != -1) {
					ret.push({
						msg: 'Invalid argument at index [' + (invalidArgIdx + 1) + '] for [' + commandUpper + '] (argument "' + command.command[invalidArgIdx + 1] + '")',
						toAll: false
					});
					status = 0;
					break;
				}

				if (gameData.gameState == 'SHOW_3' || gameData.gameState == 'SHOW_ALL') {
					let invalidArgIdx = args.findIndex(e => [11, 13, 36, 48].indexOf(gameData.hands[myIdx][0][e]) == -1 || gameData.stacks[1].findIndex(e1 => e1 == gameData.hands[myIdx][0][e]) != -1);
					if (invalidArgIdx != -1) {
						ret.push({
							msg: 'Invalid argument at index [' + (invalidArgIdx + 1) + '] for [' + commandUpper + '] (argument "' + command.command[invalidArgIdx + 1] + '")',
							toAll: false
						});
						status = 0;
						break;
					}

					let cards = args.map(e => gameData.hands[myIdx][0][e]);
					let val = gameData.gameState == 'SHOW_3' ? 4 : 2;
					for (let e of cards) {
						if (gameData.hands[myIdx][1].indexOf(e) == -1) {
							gameData.stacks[1].push([e, val]);
							gameData.hands[myIdx][1].push(e);
							ret.push({
								msg: 'Shown card [' + GameUtils.card2Str(e) + '] for x' + val + ' value',
								toAll: false
							});
						}
					}
				} else {
					if (args.length != 1) {
						ret.push({
							msg: 'Too many arguments for [' + commandUpper + '] (max 1)',
							toAll: false
						});
						status = 0;
						break;
					}

					let playableCards = new Set();
					if (relativeIdx == 0) {
						gameData.hands[myIdx][0].filter(e => !gameData.hands[myIdx][1].includes(e)).forEach(e => playableCards.add(e));
						gameData.hands[myIdx][1].filter(e => GameUtils.filterBySuit(e, gameData.hands[myIdx][0]).length == 1).forEach(e => playableCards.add(e));
					} else {
						let filtered = GameUtils.filterBySuit(gameData.hands[gameData.turnFirstIdx][3][0], gameData.hands[myIdx][0]);
						if (filtered.length == 1) {
							filtered.forEach(e => playableCards.add(e));
						} else if (filtered.length) {
							filtered.forEach(e => {
								if (!gameData.hands[myIdx][1].includes(e)) playableCards.add(e);
							});
						} else {
							gameData.hands[myIdx][0].forEach(e => playableCards.add(e));
						}
					}

					let invalidArgIdx = args.findIndex(e => !playableCards.has(gameData.hands[myIdx][0][e]));
					if (invalidArgIdx != -1) {
						ret.push({
							msg: 'Invalid argument at index [' + (invalidArgIdx + 1) + '] for [' + commandUpper + '] (argument "' + command.command[invalidArgIdx + 1] + '")',
							toAll: false
						});
						status = 0;
						break;
					}

					ret.push({
						msg: 'Player [' +  gameData.turnOrder[myIdx] + '] played card [' + GameUtils.card2Str(gameData.hands[myIdx][0][args[0]]) + ']',
						toAll: true
					});

					let shownIdx = gameData.hands[myIdx][1].findIndex(e => e == gameData.hands[myIdx][0][args[0]]);
					if (shownIdx != -1) gameData.hands[myIdx][1].splice(shownIdx, 1);
					gameData.hands[myIdx][3].push(...gameData.hands[myIdx][0].splice(args[0], 1));

					if (gameData.gameState == 'PLAY_3') {
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

						ret.push({
							msg: 'Player [' + gameData.turnOrder[gameData.turnFirstIdx] + '] wins with [' + GameUtils.card2Str(gameData.hands[winnerIdx][3][0]) + '] and takes [' + [...important].map(e => GameUtils.card2Str(e)).join(', ') + ']',
							toAll: true
						});

						gameData.hands.forEach(e => gameData.stacks[0].push(e[3].pop()));

						gameData.hands.forEach(e => {
							for (let j = e[1].length - 1; j >= 0; j--) {
								if (Math.floor(e[1][j] / 13) == trickSuit) {
									e[1].splice(j, 1);
								}
							}
						});

						if (server.gameData.hands.every(e => !e[0].length)) {
							gameData.scores.forEach((e, i) => {
								e[0] += e[1];
							});
						}
					}

					ret.push(...gameNSL(server));
				}

				Utils.broadcastGameStateToConnected(users, server, obfuscateGameData);
			}
			break;
		}
		case 'pass': {
			if (isSpectator) {
				ret.push({
					msg: 'Cannot issue command [' + commandUpper + '] as a specator',
					toAll: false
				});
				status = 0;
			} else if (command.command.length > 1) {
				ret.push({
					msg: 'Too many arguments for [' + commandUpper + '] (need 0)',
					toAll: false
				});
				status = 0;
			} else if (
				gameData.gameState != 'SHOW_3' &&
				gameData.gameState != 'SHOW_ALL'
			) {
				ret.push({
					msg: 'Cannot issue command [' + commandUpper + '] in state [' + gameData.gameState + ']',
					toAll: false
				});
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
		}
		case 'clear': {}
		case 'clr': {
			if (command.command.length > 1) {
				ret.push({
					msg: 'Too many arguments for [' + commandUpper + '] (need 0)',
					toAll: false
				});
				status = 0;
			} else {
				ws.send(JSON.stringify({tag: 'clearConsole'}));
			}
			break;
		}
		case 'debug': {
			ws.send(JSON.stringify({tag: 'toggleDebug', data: ret}));
			break;
		}
		default: {
			ret.push({
				msg: 'Unknown command [' + commandUpper + ']',
				toAll: false
			});
			status = 0;
		}
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

export function obfuscateGameData(gameData, turnOrderIdx) {
	let gameDataCopy = structuredClone(gameData);

	Utils.nullify(gameDataCopy.decks);
	gameDataCopy.hands.forEach((hand, i) => {
		if (i != turnOrderIdx) Utils.nullify(hand[0]);
	});
	Utils.nullify(gameDataCopy.stacks[0]);
	gameDataCopy.stacks[1] = gameDataCopy.stacks[1].filter(e =>
		!((gameDataCopy.gameState == 'SHOW_3' && e[1] == 4) || (gameDataCopy.gameState == 'SHOW_ALL' && e[1] == 2))
	);

	return gameDataCopy;
}
