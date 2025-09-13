let ws;

let username;
let gameDataOld, gameDataNew;
let animatingGUI = false;
let handlers = new Map();

let messageDecoder = {
	'broadcastedMessage':	(data) => {Popup.toastPopup(data.data);},
	'receiveSessionID':		(data) => {receiveSessionID(data);},
	'receiveUsername':		(data) => {receiveUsername(data);},
	'updateUserCount':		(data) => {updateUserCount(data);},
	'updateLobbies':		(data) => {updateLobbies(data);},
	'createdLobby':			(data) => {joinLobby(data);},
	'showLobby':			(data) => {showLobby(data);},
	'leftLobby':			(data) => {leftLobby(data);},
	'otherLeftLobby':		(data) => {otherLeftLobby(data);},
	'startedGame':			(data) => {startedGame(data);},
	'receiveCommand':		(data) => {receiveCommand(data);},
	'receiveChat':			(data) => {receiveChat(data);},
	'updateGUI':			(data) => {drawGUI(data);},
	'clearConsole':			(data) => {Utils.clearDiv(document.querySelector('#game-console-output'));},
	'toggleDebug':			(data) => {toggleDebug(data);}
};


function initWebSocket() {
	// ws = new WebSocket('http://localhost:8080');
	let urlParams = new URLSearchParams(window.location.search);
	let url = urlParams.get('url');

	if (!url) {
		url = prompt('Enter WebSocket URL:');
	}

	try {
		ws = new WebSocket(url);

		ws.addEventListener('open', function(e) {
			if (Cookies.getCookie('sessionID')) {
				console.log('Checking Session ID', Cookies.getCookie('sessionID'));
				ws.send(JSON.stringify({tag: 'checkSessionID', data: Cookies.getCookie('sessionID')}));
			} else {
				console.log('Requesting new Session ID');
				ws.send(JSON.stringify({tag: 'requestSessionID'}));
			}
		});

		ws.addEventListener('message', function(message) {
			let data = JSON.parse(message.data);
			let tags = data.tag.split('/');
			let func;
			for (let i = 0; i < tags.length; i++) {
				if (!i) func = messageDecoder[tags[i]];
				else func = func[tags[i]];
			}
			if (func) func(data);
		});

		ws.addEventListener('close', function(e) {
			console.log('WebSocket close:', e);
			Popup.toastPopup('WebSocket closed');

			[
				(document.querySelector('#submit-username-btn') ? document.querySelector('#submit-username-btn').parentElement : null),
				document.querySelector('#lobby-menu'),
				document.querySelector('#lobby'),
				document.querySelector('#game')
			].filter(e => e).forEach(e => e.remove());

			let div = document.createElement('div');
			div.classList.add('content-text');
			div.innerText = 'WebSocket connection closed, try again another time.';
			document.body.append(div);
		});

		ws.addEventListener('error', function(e) {
			console.log('WebSocket error:', e);
			Popup.toastPopup('WebSocket error');

			[
				(document.querySelector('#submit-username-btn') ? document.querySelector('#submit-username-btn').parentElement : null),
				document.querySelector('#lobby-menu'),
				document.querySelector('#lobby'),
				document.querySelector('#game')
			].filter(e => e).forEach(e => e.remove());

			let div = document.createElement('div');
			div.classList.add('content-text');
			div.innerText = 'WebSocket connection error, try again another time.';
			document.body.append(div);

			ws.close();
			setTimeout(location.reload, 1000);
		});
	} catch (e) {}
}

function receiveSessionID(data) {
	if (data.status) {
		username = data.data.username;
		document.querySelector('#username').style.display = null;

		document.querySelector('#username-input').style.display = 'none';
		Utils.clearDiv(document.querySelector('#username-input'));

		let div = document.createElement('div');
		div.classList.add('content-header-2');
		div.innerHTML = 'Playing as: <span style="color: var(--color_red);">' + data.data.username + '</span>';
		document.querySelector('#username').append(div);
	} else {
		document.querySelector('#username-input').style.display = null;
	}
	Cookies.setCookie('sessionID', data.data.sessionID, 30 * 60 * 1000);
	console.log('Session ID', Cookies.getCookie('sessionID'));

	handlers.set('sessionIDRefresh', setInterval(function() {
		refreshSessionIDCookie();
	}, 5000));
}

function refreshSessionIDCookie() {
	Cookies.setCookie('sessionID', Cookies.getCookie('sessionID'), 30 * 60 * 1000);
}

function submitUsername() {
	let btn = document.querySelector('#submit-username-btn');

	let username = btn.parentElement.querySelector('input').value.trim().toLowerCase();
	if (!username.length) Popup.toastPopup('Username cannot be blank');
	else if (!/^[a-z0-9]+$/i.test(username)) Popup.toastPopup('Username must only contain alphanumeric characters');
	else {
		btn.style.filter = 'brightness(0.5)';
		btn.style.cursor = 'not-allowed';

		ws.send(JSON.stringify({tag: 'requestUsername', data: username}));
	}
}

function receiveUsername(data) {
	let btn = document.querySelector('#submit-username-btn');

	if (!data.status) {
		Popup.toastPopup(data.data);
	} else {
		document.querySelector('#username-input').style.display = 'none';
		Utils.clearDiv(document.querySelector('#username-input'));

		let div = document.createElement('div');
		div.classList.add('content-header-2');
		div.innerHTML = 'Playing as: <span style="color: var(--color_red);">' + data.data + '</span>';
		document.querySelector('#username').append(div);

		document.querySelector('#username').style.display = null;
		username = data.data;

		getLobbies();
		handlers.set('lobbyRefresh', setInterval(getLobbies, 2000));

		document.querySelector('#lobby-menu').style.display = null;
	}
	btn.style.cursor = null;
	btn.style.filter = null;
}

function getLobbies() {
	ws.send(JSON.stringify({tag: 'getLobbies'}));
}

function updateUserCount(data) {
	document.querySelector('#player-count span').innerText = data.data;
}

function parseTime(timeMs, date = true, time = true) {
	let dateTime = new Date(timeMs);

	let year = dateTime.getFullYear().toString(10).substring(2);
	let month = (dateTime.getMonth() + 1).toString(10).padStart(2, '0');
	let day = dateTime.getDate().toString(10).padStart(2, '0');
	let hour = dateTime.getHours().toString(10).padStart(2, '0');
	let minutes = dateTime.getMinutes().toString(10).padStart(2, '0');

	let str = '';
	if (date) str += month + '/' + day + '/' + year;
	if (time) str += (date ? ' ' : '') + hour + ':' + minutes;

	return str;
}

async function showCreateLobbyPopup() {
	await Popup.popup(document.querySelector('#popup-create-lobby'));
	let btn = document.querySelector('#popup-create-lobby').querySelector('.button-positive-2');
	btn.onclick = function() {
		this.onclick = null;
		createLobby();
	};
}

function updateLobbies(data) {
	let lobbySelect = document.querySelector('#lobby-select');
	Utils.clearDiv(lobbySelect);

	if (!data.data.length) {
		let div = document.createElement('div');
		div.classList.add('content-text');
		div.innerText = 'No lobbies available. Create your own!';
		lobbySelect.append(div);
	}

	for (let server of data.data) {
		let container = document.createElement('div');
		container.classList.add('lobby-tile', 'content-container-vertical');

		let div1 = document.createElement('div');
		div1.classList.add('content-header-2');
		div1.innerText = server.name;

		let div2 = document.createElement('div');
		div2.classList.add('content-text');
		if (server.connected.length > server.gameData.maxPlayers) {
			div2.innerText = 'Players: ' + server.gameData.maxPlayers + ' + ' + (server.connected.length - server.gameData.maxPlayers) + ' \u{1F441}';
		} else {
			div2.innerText = 'Players: ' + server.connected.length;
		}

		let div3 = document.createElement('div');
		div3.classList.add('content-break-vertical');

		let div4 = document.createElement('div');
		div4.classList.add('content-text');
		div4.innerText = 'Created: ' + parseTime(server.time);

		let div5 = document.createElement('div');
		div5.classList.add('content-text');
		div5.innerText = 'By: ' + server.creator;

		let div6 = document.createElement('div');
		div6.classList.add('content-text');
		div6.innerText = 'Host: ' + server.host;

		[div1, div2, div3, div4, div5, div6].forEach(e => container.append(e));

		lobbySelect.append(container);
		container.data = server;

		let popup = document.querySelector('#popup-load-lobby');
		container.onclick = async function() {
			document.querySelector('#load-lobby-name').innerText = server.name;
			document.querySelector('#load-lobby-creation').innerText = 'Created: ' + parseTime(server.time) + '\nBy: ' + server.creator;
			document.querySelector('#load-lobby-host').innerText = 'Host: ' + server.host;
			if (server.gameData.settings.spectatorPolicy != 'disallowed') {
				document.querySelector('#load-lobby-users').innerText = 'Players: ' + Math.min(server.gameData.maxPlayers, server.connected.length) + ' + ' + Math.max(0, server.connected.length - server.gameData.maxPlayers) + ' \u{1F441}';
			} else {
				document.querySelector('#load-lobby-users').innerText = 'Players: ' + server.connected.length;
			}

			let list = document.querySelector('#load-lobby-user-list');
			Utils.clearDiv(list);
			for (let i = 0; i < server.connected.length; i++) {
				let div1 = document.createElement('div');
				div1.style.gridRow = (i + 1) + ' / ' + (i + 2);

				let user = server.connected[i];
				for (let j = 0; j < 3; j++) {
					let div2 = document.createElement('div');
					div2.style.gridColumn = (j + 1) + ' / ' + (j + 2);
					div2.style.alignSelf = 'center';
					div2.style.lineHeight = '1em';

					if (j == 0) {
						div2.innerText = (user.username == server.host ? '\u{1F732} ' : '');
					} else if (j == 1) {
						div2.innerText = (user.priority >= server.gameData.maxPlayers ? '\u{1F441}' : '');
					} else if (j == 2) {
						div2.innerText = user.username;
					}

					if (user.priority >= server.gameData.maxPlayers) div2.style.filter = 'opacity(60%)';

					div1.append(div2);
				}

				list.append(div1);
			}

			await Popup.popup(popup);

			popup.querySelector('.button-positive-2').onclick = function() {
				this.onclick = null;
				this.parentNode.parentNode.click();
				ws.send(JSON.stringify({tag: 'joinLobby', data: server}));
			};
		};
	}
}

function createLobby() {
	let vals = Array.from(document.querySelectorAll('#popup-create-lobby input')).map(e => e.value.trim());

	for (let val of vals) {
		if (!val.length) {
			Popup.toastPopup('One or more fields blank');
			return;
		} else if (!/^[a-z0-9]+$/i.test(val)) {
			Popup.toastPopup('Fields must only contain alphanumeric characters');
			return;
		}
	}

	ws.send(JSON.stringify({tag: 'createLobby', data: {name: vals[0], time: Date.now(), creator: username, host: username}}));
}

function joinLobby(data) {
	if (!data.status) {
		Popup.toastPopup(data.data);
		return;
	}
	ws.send(JSON.stringify({tag: 'joinLobby', data: data.data}));
}

function showLobby(data) {
	if (!data.status) {
		Popup.toastPopup(data.data);
		return;
	}

	document.querySelector('#popup-create-lobby').parentNode.click();
	document.querySelector('#popup-load-lobby').parentNode.click();

	document.querySelector('#username').style.display = 'none';
	document.querySelector('#lobby-menu').style.display = 'none';
	document.querySelector('#lobby').style.display = null;
	document.querySelector('#game').style.display = 'none';

	let server = data.data;
	document.querySelector('#lobby-name').innerText = 'Lobby [' + server.name + ']';
	document.querySelector('#lobby-creation').innerText = 'Created: ' + parseTime(server.time) + '\nBy: ' + server.creator;
	document.querySelector('#lobby-host').innerText = 'Host: ' + server.host;
	if (server.connected.length > server.gameData.maxPlayers) {
		document.querySelector('#lobby-users').innerText = 'Players: ' + server.gameData.maxPlayers + ' + ' + (server.connected.length - server.gameData.maxPlayers) + ' \u{1F441}';
	} else {
		document.querySelector('#lobby-users').innerText = 'Players: ' + server.connected.length;
	}

	document.querySelector('#btn-start-game').style.display = (server.host == username) ? null : 'none';

	console.log(server);

	let list = document.querySelector('#lobby-user-list');
	Utils.clearDiv(list);
	for (let i = 0; i < server.connected.length; i++) {
		let div1 = document.createElement('div');
		div1.style.gridRow = (i + 1) + ' / ' + (i + 2);

		let user = server.connected[i];
		for (let j = 0; j < 4; j++) {
			let div2 = document.createElement('div');
			div2.style.gridColumn = (j + 1) + ' / ' + (j + 2);
			div2.style.alignSelf = 'center';
			div2.style.lineHeight = '1em';

			if (j == 0) {
				div2.innerText = (user.username == server.host ? '\u{1F732} ' : '');
			} else if (j == 1) {
				div2.innerText = (user.priority >= server.gameData.maxPlayers ? '\u{1F441}' : '');
			} else if (j == 2) {
				div2.innerText = user.username + (user.username == username ? ' <= You' : '');
			} else if (j == 3) {
				div2.innerText = '(' + user.priority + ')';
				div2.style.textAlign = 'center';
				div2.style.fontSize = '0.8em';
				div2.style.color = 'var(--color_txt2)';
			}

			if (user.priority >= server.gameData.maxPlayers) div2.style.filter = 'opacity(60%)';

			div1.append(div2);
		}

		list.append(div1);
	}

document.querySelector('#lobby-settings-spectator-policy').value = server.gameData.settings.spectatorPolicy;
document.querySelector('#lobby-settings-losing-threshold').value = server.gameData.settings.losingThreshold;
document.querySelector('#lobby-settings-expose-3').checked = server.gameData.settings.expose3;
document.querySelector('#lobby-settings-zhu-yang-man-juan').checked = server.gameData.settings.zhuYangManJuan;

for (let e of document.querySelectorAll('#lobby-settings input, #lobby-settings select')) {
	if (username == server.host) {
		e.onchange = function() {
			ws.send(JSON.stringify({tag: 'updateLobbySettings', data: {settings: {
				spectatorPolicy: document.querySelector('#lobby-settings-spectator-policy').value,
				losingThreshold: parseInt(document.querySelector('#lobby-settings-losing-threshold').value, 10),
				expose3: document.querySelector('#lobby-settings-expose-3').checked,
				zhuYangManJuan: document.querySelector('#lobby-settings-zhu-yang-man-juan').checked
				}}}));
			};
			e.disabled = false;
		} else {
			e.onchange = null;
			e.disabled = true;
		}
	}

	if (handlers.get('lobbyRefresh')) clearInterval(handlers.get('lobbyRefresh'));
}

function leaveLobby() {
	ws.send(JSON.stringify({tag: 'leaveLobby'}));
}

function leftLobby(data) {
	if (!data.status) {
		Popup.toastPopup(data.data);
		return;
	}

	getLobbies();
	handlers.set('lobbyRefresh', setInterval(getLobbies, 2000));

	document.querySelector('#title').style.display = null;
	document.querySelector('#username').style.display = null;
	document.querySelector('#lobby-menu').style.display = null;
	document.querySelector('#lobby').style.display = 'none';
	document.querySelector('#game').style.display = 'none';
}

function otherLeftLobby(data) {
	document.querySelector('#title').style.display = null;
	document.querySelector('#lobby').style.display = null;
	document.querySelector('#game').style.display = 'none';
	Popup.toastPopup('Player disconnected, returning to lobby...');
	showLobby(data);
}

function startGame() {
	ws.send(JSON.stringify({tag: 'startGame'}));
}

function startedGame(data) {
	document.querySelector('#username').style.display = 'none';
	document.querySelector('#title').style.display = 'none';
	document.querySelector('#lobby').style.display = 'none';
	document.querySelector('#game').style.display = null;

	Utils.clearDiv(document.querySelector('#game-console-output'));
	Utils.clearDiv(document.querySelector('#game-chat-output'));
}

function sendCommand() {
	let data = document.querySelector('#game-console-input').value.trim();
	if (data) {
		let code = document.createElement('code');
		code.innerText = '>> ' + data;
		document.querySelector('#game-console-output').append(code);
		ws.send(JSON.stringify({tag: 'sendCommand', data: data}));
		document.querySelector('#game-console-input').value = '';
		code.scrollIntoView({behavior: 'smooth', block: 'end'});
	}
}

function receiveCommand(data) {
	let output = document.querySelector('#game-console-output');
	console.log('receiveCommand', data);
	for (let log of data.data) {
		let code = document.createElement('code');
		if (!data.status) code.style.color = 'var(--color_red)';
		code.textContent = log;
		output.append(code);
		code.scrollIntoView({behavior: 'smooth', block: 'end'});
	}
}

function sendChat() {
	let data = document.querySelector('#game-chat-input').value.trim();
	if (data) {
		ws.send(JSON.stringify({tag: 'sendChat', data: data}));
		document.querySelector('#game-chat-input').value = '';
	}
}

function receiveChat(data) {
	let output = document.querySelector('#game-chat-output');
	let isAtBottom = !Math.floor(output.scrollHeight - output.scrollTop - output.clientHeight);

	let div = document.createElement('div');
	let span1 = document.createElement('span');

	let span1Username = document.createElement('b');
	span1Username.classList.add('username');
	span1Username.innerText = data.data.username;
	span1.append(span1Username);

	let span1Text = document.createElement('span');
	span1Text.textContent = '\u00A0\u00A0' + data.data.text;
	span1.append(span1Text);

	let span2 = document.createElement('span');
	span2.innerText = parseTime(data.data.time, date = false);

	div.append(span1, span2);
	output.append(div);
	if (isAtBottom) div.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

async function drawGUI(data) { // TODO animation
	console.log('drawGUI', data);

	let gameData = data.data.gameData;
	let server = data.data.serverData;

	let drawCard = document.querySelector('#symbolic-cards-checkbox').checked ? Cards.card2Str : Cards.card2Unicode;
	let drawCardSpacing = document.querySelector('#symbolic-cards-checkbox').checked ? ' ' : '';

	// if (animatingGUI) {
	// 	gameDataNew = gameData;
	// 	return;
	// }

	// if (gameDataOld && gameDataOld.gameState != gameData.gameState) {
	// 	animatingGUI = true;
	// }

	// Clear GUI
	let leaderboard = document.querySelector('#game-gui-leaderboard');
	let handsSelf = document.querySelector('#game-gui-hands-self');
	let hands = document.querySelector('#game-gui-hands');
	let shownCards = document.querySelector('#game-gui-shown-cards');

	Utils.clearDiv(leaderboard);
	Utils.clearDiv(handsSelf);
	Utils.clearDiv(hands);
	Utils.clearDiv(shownCards);

	// Show proper frame
	if (
		gameData.gameState == 'LEADERBOARD' || 
		gameData.gameState == 'SCORE'
	) {
		document.querySelector('#game-gui-frame-game').style.display = 'none';
		document.querySelector('#game-gui-frame-leaderboard').style.display = null;

		if (gameData.gameState == 'LEADERBOARD') {
			document.querySelector('#game-gui-help-next-round').style.display = 'none';
			document.querySelector('#game-gui-help-next-game').style.display = null;
		} else {
			document.querySelector('#game-gui-help-next-round').style.display = null;
			document.querySelector('#game-gui-help-next-game').style.display = 'none';
		}
	} else {
		document.querySelector('#game-gui-frame-game').style.display = null;
		document.querySelector('#game-gui-frame-leaderboard').style.display = 'none';
	}

	// Initialize values
	document.querySelector('#game-gui-round-count').innerText = gameData.round;
	document.querySelector('#game-gui-limit-count').innerText = gameData.settings.losingThreshold;
	document.querySelector('#game-gui-trick-count').innerText = Math.round(gameData.stacks[0].length / gameData.turnOrder.length) + 1;
	document.querySelector('#game-gui-game-state').innerText = gameData.gameState;

	let absMaxScore = Math.abs(gameData.scores[0][0]);
	for (let i of gameData.scores) {
		if (Math.abs(i[0]) > absMaxScore) absMaxScore = Math.abs(i[0]);
	}

	let myIdx = gameData.turnOrder.findIndex(e => e == username);
	let isSpectator = myIdx == -1;

	let consoleInput = document.querySelector('#game-console-input');
	if (isSpectator) {
		consoleInput.placeholder = 'You are a spectator!';
	} else {
		consoleInput.placeholder = 'Enter command';
	}
	consoleInput.placeholder += ' (HELP for command list)';

	let sortedOrder = gameData.scores.map((e, i) => [e[0], i]).toSorted((a, b) => {
		if (a[0] != b[0]) return b[0] - a[0];
		else return a[1] - b[1];
	});

	for (let i = 0; i < sortedOrder.length; i++) {
		let div1 = document.createElement('div');
		div1.classList.add('player-' + i);
		div1.style.gridRow = (i + 1) + ' / ' + (i + 2);

		for (let j = 0; j < 4; j++) {
			let div2 = document.createElement('div');
			div2.classList.add('content-container-text');
			div2.style.gridColumn = (j + 1) + ' / ' + (j + 2);
			div2.style.alignSelf = 'center';
			div2.style.lineHeight = '1em';

			if (j == 0) {
				if (username == gameData.turnOrder[sortedOrder[i][1]]) {
					div2.innerText = '\u{1F464}';
				}
			} else if (j == 1) {
				div2.innerText = gameData.turnOrder[sortedOrder[i][1]];
			} else if (j == 2) {
				div2.innerText = sortedOrder[i][0];
				let percent = Math.abs(sortedOrder[i][0]) / absMaxScore * 100;
				div2.style.color = 'color-mix(in srgb, var(--color_' + ((sortedOrder[i][0] < 0) ? 'red' : 'green') + ') ' + percent + '%, var(--color_black))';
			} else if (j == 3) {
				if (gameData.scores[sortedOrder[i][1]][1]) {
					div2.innerText = (gameData.scores[sortedOrder[i][1]][1] > 0 ? '+' : '') + gameData.scores[sortedOrder[i][1]][1];
					div2.style.color = gameData.scores[sortedOrder[i][1]][1] > 0 ? 'var(--color_green)' : 'var(--color_red)';
				}
			}
			div1.append(div2);
		}
		leaderboard.append(div1);
	}

	if (!isSpectator) {
		let playableCards = new Set();
		switch (gameData.gameState) {
			case 'SHOW_3': {}
			case 'SHOW_ALL': {
				[11, 13, 36, 48].filter(e => gameData.stacks[1].findIndex(e1 => e1[0] == e) == -1).forEach(e => playableCards.add(e));
				break;
			}
			case 'PLAY_0': {
				if (username == gameData.turnOrder[gameData.turnFirstIdx]) {
					gameData.hands[myIdx][0].filter(e => !gameData.hands[myIdx][1].includes(e)).forEach(e => playableCards.add(e));
					gameData.hands[myIdx][1].filter(e => Cards.filterBySuit(e, gameData.hands[myIdx][0]).length == 1).forEach(e => playableCards.add(e));
				}
				break;
			}
			case 'PLAY_1': {
				if (username == gameData.turnOrder[(gameData.turnFirstIdx + 1) % gameData.turnOrder.length]) {
					let filtered = Cards.filterBySuit(gameData.hands[gameData.turnFirstIdx][3][0], gameData.hands[myIdx][0]);
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
				break;
			}
			case 'PLAY_2': {
				if (username == gameData.turnOrder[(gameData.turnFirstIdx + 2) % gameData.turnOrder.length]) {
					let filtered = Cards.filterBySuit(gameData.hands[gameData.turnFirstIdx][3][0], gameData.hands[myIdx][0]);
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
				break;
			}
			case 'PLAY_3': {
				if (username == gameData.turnOrder[(gameData.turnFirstIdx + 3) % gameData.turnOrder.length]) {
					let filtered = Cards.filterBySuit(gameData.hands[gameData.turnFirstIdx][3][0], gameData.hands[myIdx][0]);
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
				break;
			}
		}
		for (let i = 0; i < gameData.hands[myIdx][0].length; i++) {
			let card = gameData.hands[myIdx][0][i];

			let div1 = document.createElement('div');

			let div2 = document.createElement('div');
			div2.innerText = i;
			if (gameData.hands[myIdx][1].indexOf(card) != -1) div2.style.color = 'var(--color_red)';
			let div3 = document.createElement('div');
			div3.innerText = drawCard(card);

			if (!playableCards.has(card)) div1.classList.add('unplayable');

			div1.append(div2, div3);
			handsSelf.append(div1);
		}
	}

	for (let i = 0; i < gameData.turnOrder.length; i++) {
		let div1 = document.createElement('div');
		div1.classList.add('player-' + i);
		div1.style.gridRow = (i + 1) + ' / ' + (i + 2);

		for (let j = 0; j < 8; j++) {
			let div2 = document.createElement('div');
			div2.classList.add('content-container-text');
			div2.style.gridColumn = (j + 1) + ' / ' + (j + 2);
			div2.style.alignSelf = 'center';
			div2.style.lineHeight = '1em';

			if (j == 0) {
				let relativeTurn = ((i - gameData.turnFirstIdx) % (gameData.turnOrder.length)) + (((i - gameData.turnFirstIdx) % (gameData.turnOrder.length)) < 0 ? gameData.turnOrder.length : 0);
				if (gameData.gameState == 'SHOW_3' || gameData.gameState == 'SHOW_ALL') {
					if (gameData.needToAct[i]) {
						div2.innerText = '✘';
						div2.style.color = 'var(--color_red)';
					} else {
						div2.innerText = '✔';
						div2.style.color = 'var(--color_green)';
					}
				} else if (gameData.gameState == 'PLAY_' + relativeTurn) {
					div2.innerText = '>>';
				}
			} else if (j == 1) {
				if (username == gameData.turnOrder[i]) {
					div2.innerText = '\u{1F464}';
					myIdx = i;
				}
			} else if (j == 2) {
				div2.innerText = gameData.turnOrder[i];
			} else if (j == 3){
				div2.innerText = '🂠' + gameData.hands[i][0].length;
			} else if (j == 4) {
				div2.style.fontSize = '1.6em';
				if (gameData.gameState == 'SHOW_ALL') {
					div2.innerText = gameData.hands[i][1].filter(e => gameData.stacks[1].filter(e1 => e1[1] == 4).map(e1 => e1[0]).includes(e)).map(e => Cards.card2Unicode(e)).join('');
				} else if (gameData.gameState.startsWith('PLAY_')) {
					div2.innerText = gameData.hands[i][1].map(e => drawCard(e)).join(drawCardSpacing);
				}
			} else if (j == 5) {
				div2.style.fontSize = '0.8em';
				if (gameData.scores[i][1]) {
					div2.innerText = '(' + (gameData.scores[i][1] > 0 ? '+' : '') + gameData.scores[i][1] + ')';
					div2.style.color = gameData.scores[i][1] > 0 ? 'var(--color_green)' : 'var(--color_red)';
				}
			} else if (j == 6) {
				div2.style.fontSize = '1.6em';
				div2.innerText = gameData.hands[i][2].map(e => drawCard(e)).join(drawCardSpacing);
			} else if (j == 7) {
				div2.style.fontSize = '2.4em';
				div2.innerText = gameData.hands[i][3].map(e => drawCard(e)).join(drawCardSpacing);
			}

			if (div2.innerText.length) div2.classList.add('not-empty');
			div1.append(div2);
		}
		hands.append(div1);
	}

	for (let e of gameData.stacks[1]) {
		if ((gameData.gameState == 'SHOW_ALL' && e[1] == 4) || (gameData.gameState.startsWith('PLAY_'))) {
			let div1 = document.createElement('div');

			let div2 = document.createElement('div');
			div2.innerText = drawCard(e[0]);
			let div3 = document.createElement('div');
			div3.innerText = '(x' + e[1] + ')';

			div1.append(div2, div3);

			shownCards.append(div1);
		}
	}

	for (let e of document.querySelectorAll('.game-gui-server-owner')) {
		e.innerText = server.host;
		if (server.host == username) e.style.color = 'var(--color_red)';
	}

	// if (animatingGUI) {
	// 	animatingGUI = false;
	// 	drawGUI({data: {gameData: gameDataNew, serverData: server}});
	// }

	gameDataOld = data;
}

function toggleDebug(data) {
	let debugElements = document.querySelectorAll('.debug');
	for (let e of debugElements) {
		e.classList.toggle('shown');
	}
}
