import * as Utils from '../../utils/utils.js';
import * as CommandParse from '../../utils/command_parse.js';
import * as Core from './core.js';
import { Game } from '../game.js';

export class GongZhu extends Game {
	get defaultSettings() {
		return Core.defaultSettings;
	}

	obfuscateGameData(gameData, turnOrderIdx) {
		return Core.obfuscateGameData(gameData, turnOrderIdx);
	}

	initGame(server) {
		Core.clearGameData(server.gameData);
		server.commandQueue = [];
		server.processingCommand = false;
		Core.initGame(server.gameData, this.generateTurnOrder(server), server.dealRngFn || undefined);
		Utils.broadcastGameStateToConnected(this.users, server, this.obfuscateGameData);
	}

	processCommand(data, ws, server) {
		let gameData = server.gameData;

		let command = CommandParse.parseCommand(data);
		let commandUpper = command.command[0].toUpperCase();
		let ret = [];
		let status = 1;

		let myIdx = gameData.turnOrder.indexOf(ws.username);
		let isSpectator = myIdx == -1;
		let newTurnOrder = null;

		let verb = command.command[0].toLowerCase();

		switch (verb) {
			// Gameplay Commands
			case 'deal': {
				if (ws.username != server.host) {
					ret.push({
						msg: 'Unknown command [' + command.command[0] + ']',
						toAll: false
					});
					return {tag: 'receiveCommand', status: 0, data: ret};
				}
				if (command.command.length > 1) {
					ret.push({
						msg: 'Too many arguments for [' + commandUpper + '] (max 0)',
						toAll: false
					});
					return {tag: 'receiveCommand', status: 0, data: ret};
				}
				if (gameData.gameState == 'LEADERBOARD' && gameData.scores.some(e => e[0] <= gameData.settings.losingThreshold)) {
					let previousTurnOrder = new Set(gameData.turnOrder);

					this.rotateSpectators(server);
					newTurnOrder = this.generateTurnOrder(server);

					let newTurnOrderSet = new Set(newTurnOrder);
					for (let username of previousTurnOrder) {
						if (!newTurnOrderSet.has(username)) this.users.get(username).send(Utils.JSONStringify({tag: 'broadcastedMessage', data: 'You are now spectating!', timestamp: Date.now()}));
					}
					for (let username of newTurnOrderSet) {
						if (!previousTurnOrder.has(username)) this.users.get(username).send(Utils.JSONStringify({tag: 'broadcastedMessage', data: 'You are now playing!', timestamp: Date.now()}));
					}
				}
			}
			case 'play': {}
			case 'pass': {
				if (isSpectator && verb != 'deal') {
					ret.push({
						msg: 'Cannot issue command [' + commandUpper + '] as a specator',
						toAll: false
					});
					return {tag: 'receiveCommand', status: 0, data: ret};
				}

				let res = Core.applyCommand(gameData, myIdx, command, server.dealRngFn || undefined, newTurnOrder);
				status = res.status;
				ret.push(...res.events);
				if (status) Utils.broadcastGameStateToConnected(this.users, server, this.obfuscateGameData);
				return {tag: 'receiveCommand', status: status, data: ret};
			}

			// Non-Gameplay Commands
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
						Utils.broadcastToConnected(this.users, server,
							{tag: 'broadcastedMessage', data: '[' + ws.username + '] exited to lobby', timestamp: Date.now()}
						);
						Utils.broadcastToConnected(this.users, server,
							{tag: 'showLobby', status: 1, data: server, timestamp: Date.now()}
						);
					} else {
						let res = this.leaveServer(ws, server);
						ws.send(Utils.JSONStringify({tag: 'leftLobby', status: res[0], data: res[1], timestamp: Date.now()}));

						if (res[0]) delete ws.connected;
					}
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

					Utils.broadcastGameState(ws, server, this.obfuscateGameData);
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

					Utils.broadcastGameState(ws, server, this.obfuscateGameData);
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
					ws.send(Utils.JSONStringify({tag: 'clearConsole', timestamp: Date.now()}));
				}
				break;
			}
			case 'debug': {
				ws.send(Utils.JSONStringify({tag: 'toggleDebug', data: ret, timestamp: Date.now()}));
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
}
