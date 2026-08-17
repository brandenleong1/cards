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
		let idx = gameData.stacks[1].findIndex(card => card[0] == e)
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
