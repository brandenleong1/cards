import * as Utils from '../../utils/utils.js';
import * as GameUtils from '../../utils/game_utils.js';
import * as CommandParse from '../../utils/command_parse.js';
import * as Core from './core.js';
import { Game } from '../game.js';

export class GongZhu extends Game {
	get defaultSettings() {
		return Core.defaultSettings;
	}

	gameNSL(server) {
		let ret = Core.gameNSL(server.gameData, server.dealRngFn || undefined);
		Utils.broadcastGameStateToConnected(this.users, server, this.obfuscateGameData);
		return ret;
	}

	gameOFL(server) {
		let ret = Core.gameOFL(server.gameData, server.dealRngFn || undefined);
		Utils.broadcastGameStateToConnected(this.users, server, this.obfuscateGameData);
		return ret;
	}

	obfuscateGameData(gameData, turnOrderIdx) {
		return Core.obfuscateGameData(gameData, turnOrderIdx);
	}

	initGame(server) {
		this.clearGameData(server);
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

							this.rotateSpectators(server);
							this.clearGameData(server);

							gameData.turnOrder = this.generateTurnOrder(server);
							Core.initGameData(server.gameData);

							let newTurnOrder = new Set(gameData.turnOrder);

							for (let username of previousTurnOrder) {
								if (!newTurnOrder.has(username)) this.users.get(username).send(Utils.JSONStringify({tag: 'broadcastedMessage', data: 'You are now spectating!', timestamp: Date.now()}));
							}
							for (let username of newTurnOrder) {
								if (!previousTurnOrder.has(username)) this.users.get(username).send(Utils.JSONStringify({tag: 'broadcastedMessage', data: 'You are now playing!', timestamp: Date.now()}));
							}
						}

						ret.push(...this.gameNSL(server));
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

						let playableCards = Core.legalMoves(gameData, myIdx);

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
							gameData.scores[winnerIdx][1] = Core.scoreFromCards(gameData.hands[winnerIdx][2], gameData);

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

						ret.push(...this.gameNSL(server));
					}

					Utils.broadcastGameStateToConnected(this.users, server, this.obfuscateGameData);
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
						ret.push(...this.gameNSL(server));
						break;
					}
					Utils.broadcastGameStateToConnected(this.users, server, this.obfuscateGameData);
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
