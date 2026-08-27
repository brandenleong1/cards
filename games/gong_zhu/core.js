import * as Utils from '../../utils/utils.js';
import * as GameUtils from '../../utils/game_utils.js';

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
		zhuYangManJuan: false,
		allowCustomSeed: false,
		customSeed: 0
	},
	currentFrame: -1n,
};

export function resetRoundData(gameData, rngFn = undefined) {
	let properties = [
		'decks',
		'hands',
		'stacks'
	];
	for (let property of properties) {
		gameData[property] = structuredClone(defaultSettings[property]);
	}

	for (let i = 0; i < gameData.numDecks; i++) {
		gameData.decks.push(GameUtils.initDeck());
		gameData.decks[i] = Utils.shuffleArray(gameData.decks[i], rngFn || undefined);
	}

	for (let i = 0; i < gameData.turnOrder.length; i++) {
		gameData.hands.push(new Array());
		for (let j = 0; j < 4; j++) gameData.hands[i].push(new Array());
	}

	gameData.scores.forEach(e => e[1] = 0);
	gameData.stacks.forEach(e => e.length = 0);
}

export function clearGameData(gameData) {
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
		gameData[property] = structuredClone(defaultSettings[property]);
	}
}

export function initGameData(gameData) {
	for (let i = 0; i < gameData.turnOrder.length; i++) {
		gameData.hands.push(new Array());
		for (let j = 0; j < 4; j++) gameData.hands[i].push(new Array());
		gameData.scores.push(new Array());
		for (let j = 0; j < 2; j++) gameData.scores[i].push(0);
		gameData.needToAct.push(0);
	}

	gameData.gameState = 'LEADERBOARD';
	gameData.round = 0;
	gameData.turnFirstIdx = 0;
}

export function initGame(gameData, turnOrder, rngFn = undefined) {
	for (let i = 0; i < gameData.numDecks; i++) {
		gameData.decks.push(GameUtils.initDeck());
		gameData.decks[i] = Utils.shuffleArray(gameData.decks[i], rngFn || undefined);
	}

	gameData.turnOrder = turnOrder;
	gameData.currentFrame = 0n;
	initGameData(gameData);

	return gameOFL(gameData, rngFn);
}

export function gameNSL(gameData, rngFn = undefined) {
	if (gameData.gameState == 'SHOW_3') {
		gameData.gameState = 'SHOW_ALL';
	} else if (gameData.gameState == 'SHOW_ALL') {
		gameData.gameState = 'PLAY_0';
	} else if (gameData.gameState == 'PLAY_0') {
		gameData.gameState = 'PLAY_1';
	} else if (gameData.gameState == 'PLAY_1') {
		gameData.gameState = 'PLAY_2';
	} else if (gameData.gameState == 'PLAY_2') {
		gameData.gameState = 'PLAY_3';
	} else if (gameData.gameState == 'PLAY_3') {
		if (gameData.hands.every(e => !e[0].length)) {
			if (gameData.scores.some(e => (e[0] + e[1]) <= gameData.settings.losingThreshold)) {
				gameData.gameState = 'LEADERBOARD';
			} else {
				gameData.gameState = 'SCORE';
			}
		} else {
			gameData.gameState = 'PLAY_0';
		}
	} else if (gameData.gameState == 'SCORE') {
		if (gameData.settings.expose3) {
			gameData.gameState = 'SHOW_3';
		} else {
			gameData.gameState = 'SHOW_ALL';
		}
	}  else if (gameData.gameState == 'LEADERBOARD') {
		if (gameData.settings.expose3) {
			gameData.gameState = 'SHOW_3';
		} else {
			gameData.gameState = 'SHOW_ALL';
		}
	}

	gameData.currentFrame++;
	return gameOFL(gameData, rngFn);
}

export function gameOFL(gameData, rngFn = undefined) {
	let state = gameData.gameState;
	let ret = [];

	if (state == 'SHOW_3') {
		resetRoundData(gameData, rngFn);
		gameData.round += 1;

		for (let i = 0; i < gameData.turnOrder.length; i++) {
			for (let j = 0; j < 3; j++) gameData.hands[i][0].push(gameData.decks[0].pop());
			gameData.needToAct[i] = 1;
		}
	} else if (state == 'SHOW_ALL') {
		if (!gameData.settings.expose3) {
			resetRoundData(gameData, rngFn);
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
		for (let i = 0; i < gameData.turnOrder.length; i++) gameData.needToAct[i] = 0;
		gameData.needToAct[(gameData.turnFirstIdx + 0) % gameData.turnOrder.length] = 1;

		ret.push({
			msg: 'Started Trick ' + (Math.round(gameData.stacks[0].length / 4) + 1) + '; Player [' + gameData.turnOrder[gameData.turnFirstIdx] + '] leads...',
			toAll: true
		});
	} else if (state == 'PLAY_1') {
		for (let i = 0; i < gameData.turnOrder.length; i++) gameData.needToAct[i] = 0;
		gameData.needToAct[(gameData.turnFirstIdx + 1) % gameData.turnOrder.length] = 1;

	} else if (state == 'PLAY_2') {
		for (let i = 0; i < gameData.turnOrder.length; i++) gameData.needToAct[i] = 0;
		gameData.needToAct[(gameData.turnFirstIdx + 2) % gameData.turnOrder.length] = 1;

	} else if (state == 'PLAY_3') {
		for (let i = 0; i < gameData.turnOrder.length; i++) gameData.needToAct[i] = 0;
		gameData.needToAct[(gameData.turnFirstIdx + 3) % gameData.turnOrder.length] = 1;

	} else if (state == 'SCORE') {
		gameData.scores.forEach((e, i) => {
			ret.push({
				msg: 'Player [' + gameData.turnOrder[i] + '] receives ' + (e[1] > 0 ? '+' + e[1] : e[1]),
				toAll: true
			});
		});
	} else if (state == 'LEADERBOARD') {
		if (gameData.scores.some(e => e[0] <= gameData.settings.losingThreshold)) {
			for (let i = 0; i < gameData.turnOrder.length; i++) {
				ret.push({
					msg: 'Player [' + gameData.turnOrder[i] + '] ' + (gameData.scores[i][0] <= gameData.settings.losingThreshold ? 'loses' : 'survives') + ' ↦ ' + gameData.scores[i][0] + ' pts',
					toAll: true
				});
			}
		}
	}

	return ret;
}

export function legalMoves(gameData, seat) {
	let n = gameData.turnOrder.length;
	let relativeIdx = (((seat - gameData.turnFirstIdx) % n) + n) % n;

	let playableCards = new Set();
	if (relativeIdx == 0) {
		gameData.hands[seat][0].filter(e => !gameData.hands[seat][1].includes(e)).forEach(e => playableCards.add(e));
		gameData.hands[seat][1].filter(e => GameUtils.filterBySuit(e, gameData.hands[seat][0]).length == 1).forEach(e => playableCards.add(e));
	} else {
		let filtered = GameUtils.filterBySuit(gameData.hands[gameData.turnFirstIdx][3][0], gameData.hands[seat][0]);
		if (filtered.length == 1) {
			filtered.forEach(e => playableCards.add(e));
		} else if (filtered.length) {
			filtered.forEach(e => {
				if (!gameData.hands[seat][1].includes(e)) playableCards.add(e);
			});
		} else {
			gameData.hands[seat][0].forEach(e => playableCards.add(e));
		}
	}

	return playableCards;
}

export function scoreFromCards(cardArr, gameData) {
	let c = 0;

	let cardSet = new Set(cardArr);
	let heartSet = new Set(new Array(13).fill(0).map((e, i) => i + 13));

	let modifiers = [11, 13, 36, 48].reduce((dict, e) => {
		let idx = gameData.stacks[1].findIndex(card => card[0] == e);
		dict[e] = (idx == -1 ? 1 : gameData.stacks[1][idx][1]);
		return dict;
	}, {});

	if (cardSet.isSupersetOf(heartSet)) {
		c += modifiers[13] * 200;

		if (gameData.settings.zhuYangManJuan) {
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

export function applyCommand(gameData, seat, command, rngFn = undefined, newTurnOrder = null) {
	let commandUpper = command.command[0].toUpperCase();
	let ret = [];
	let status = 1;

	let n = gameData.turnOrder.length;
	let relativeIdx = (((seat - gameData.turnFirstIdx) % n) + n) % n;

	switch (command.command[0].toLowerCase()) {
		case 'deal': {
			if (newTurnOrder) {
				clearGameData(gameData);
				gameData.turnOrder = newTurnOrder;
				initGameData(gameData);
			}
			if (gameData.gameState == 'LEADERBOARD' || gameData.gameState == 'SCORE') {
				ret.push(...gameNSL(gameData, rngFn));
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
			break;
		}
		case 'play': {
			if (
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
				let invalidArgIdx = args.findIndex(e => isNaN(e) || e < 0 || e >= gameData.hands[seat][0].length);
				if (invalidArgIdx != -1) {
					ret.push({
						msg: 'Invalid argument at index [' + (invalidArgIdx + 1) + '] for [' + commandUpper + '] (argument "' + command.command[invalidArgIdx + 1] + '")',
						toAll: false
					});
					status = 0;
				} else if (gameData.gameState == 'SHOW_3' || gameData.gameState == 'SHOW_ALL') {
					let invalidArgIdx = args.findIndex(e => [11, 13, 36, 48].indexOf(gameData.hands[seat][0][e]) == -1);
					if (invalidArgIdx != -1) {
						ret.push({
							msg: 'Invalid argument at index [' + (invalidArgIdx + 1) + '] for [' + commandUpper + '] (argument "' + command.command[invalidArgIdx + 1] + '")',
							toAll: false
						});
						status = 0;
					} else {
						let cards = args.map(e => gameData.hands[seat][0][e]);
						let val = gameData.gameState == 'SHOW_3' ? 4 : 2;
						for (let e of cards) {
							if (gameData.hands[seat][1].indexOf(e) == -1) {
								gameData.stacks[1].push([e, val]);
								gameData.hands[seat][1].push(e);
								ret.push({
									msg: 'Shown card [' + GameUtils.card2Str(e) + '] for x' + val + ' value',
									toAll: false
								});
							}
						}
					}
				} else if (args.length != 1) {
					ret.push({
						msg: 'Too many arguments for [' + commandUpper + '] (max 1)',
						toAll: false
					});
					status = 0;
				} else {
					let playableCards = legalMoves(gameData, seat);
					let invalidArgIdx = args.findIndex(e => !playableCards.has(gameData.hands[seat][0][e]));
					if (invalidArgIdx != -1) {
						ret.push({
							msg: 'Invalid argument at index [' + (invalidArgIdx + 1) + '] for [' + commandUpper + '] (argument "' + command.command[invalidArgIdx + 1] + '")',
							toAll: false
						});
						status = 0;
					} else {
						ret.push({
							msg: 'Player [' +  gameData.turnOrder[seat] + '] played card [' + GameUtils.card2Str(gameData.hands[seat][0][args[0]]) + ']',
							toAll: true
						});

						let shownIdx = gameData.hands[seat][1].findIndex(e => e == gameData.hands[seat][0][args[0]]);
						if (shownIdx != -1) gameData.hands[seat][1].splice(shownIdx, 1);
						gameData.hands[seat][3].push(...gameData.hands[seat][0].splice(args[0], 1));

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
							gameData.scores[winnerIdx][1] = scoreFromCards(gameData.hands[winnerIdx][2], gameData);

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

							if (gameData.hands.every(e => !e[0].length)) {
								gameData.scores.forEach((e, i) => {
									e[0] += e[1];
								});
							}
						}

						ret.push(...gameNSL(gameData, rngFn));
					}
				}
			}
			break;
		}
		case 'pass': {
			if (command.command.length > 1) {
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
				gameData.needToAct[seat] = 0;
				if (gameData.needToAct.every(e => e == 0)) {
					ret.push(...gameNSL(gameData, rngFn));
				}
			}
			break;
		}
	}

	return {status: status, events: ret};
}

export function obfuscateGameData(gameData, turnOrderIdx) {
	let gameDataCopy = structuredClone(gameData);

	Utils.nullify(gameDataCopy.decks);
	Utils.nullify(gameDataCopy.stacks[0]);
	gameDataCopy.stacks[1] = gameDataCopy.stacks[1].filter(e =>
		!((gameDataCopy.gameState == 'SHOW_3' && e[1] == 4) || (gameDataCopy.gameState == 'SHOW_ALL' && e[1] == 2))
	);

	let shownCards = new Set(gameDataCopy.stacks[1].map(e => e[0]));
	gameDataCopy.hands.forEach((hand, i) => {
		if (i != turnOrderIdx) {
			Utils.nullify(hand[0]);
			hand[1] = hand[1].filter(e => shownCards.has(e));
		}
	});

	return gameDataCopy;
}
