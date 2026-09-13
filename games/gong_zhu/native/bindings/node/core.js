import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const addon = require('./gong_zhu.node');

function seedFrom(rngFn) {
	const r = (rngFn || Math.random)();
	return Math.floor(r * 0x100000000) >>> 0;
}

export const defaultSettings = addon.getDefaultSettings();

export function parseCommand(input) {
	return addon.parseCommand(input, {});
}

export function clearGameData(gameData) {
	Object.assign(gameData, addon.clearGameData(gameData));
}

export function initGame(gameData, turnOrder, rngFn = undefined) {
	Object.assign(gameData, addon.initGame(gameData, turnOrder, seedFrom(rngFn)));
}

export function applyCommand(gameData, seat, command, rngFn = undefined, newTurnOrder = null) {
	const willShuffle = command.command[0].toUpperCase() === 'DEAL';
	const seed = willShuffle ? seedFrom(rngFn) : 0;
	return addon.applyCommand(gameData, seat, command, seed, newTurnOrder ?? null);
}

export function obfuscateGameData(gameData, turnOrderIdx) {
	return addon.obfuscateGameData(gameData, turnOrderIdx);
}
