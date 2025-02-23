let ws;
// let ws = new WebSocket('ws' + window.location.href.substring(window.location.href.indexOf(':')));

let username;
let handlers = {};

let messageDecoder = {
	'broadcastedMessage':	(data) => {Popup.toastPopup(data.data);},
	'receiveUsername':		(data) => {receiveUsername(data);},
	'updateUserCount':		(data) => {updateUserCount(data);},
	'updateLobbies':		(data) => {updateLobbies(data);},
	'createdLobby':			(data) => {joinLobby(data);},
	'joinedLobby':			(data) => {showLobby(data);},
	'leftLobby':			(data) => {leftLobby(data);},
	'otherLeftLobby':		(data) => {otherLeftLobby(data);},
	'startedGame':			(data) => {startedGame(data);},
	'receiveCommand':		(data) => {receiveCommand(data);},
	'receiveChat':			(data) => {receiveChat(data);},
	'updateGUI':			(data) => {drawGUI(data);},
	'toggleDebug':			(data) => {toggleDebug(data);}
};


function initWebSocket() {
	ws = new WebSocket('http://localhost:8080');
	// ws = new WebSocket('https://fxvw5vx2-8080.usw3.devtunnels.ms/');

	ws.addEventListener('message', function(message) {
		// console.log(message);

		let data = JSON.parse(message.data);
		let tags = data.tag.split('/');
		let func;
		for (let i = 0; i < tags.length; i++) {
			if (!i) func = messageDecoder[tags[i]];
			else func = func[tags[i]];
		}
		if (func) func(data);
	});

	ws.addEventListener('error', function(e) {
		console.log('WebSocket error:', e);
		Popup.toastPopup('WebSocket error');

		[
			document.querySelector('#submit-username-btn').parentElement,
			document.querySelector('#lobby-menu')
		].forEach(e => e.remove());

		let div = document.createElement('div');
		div.classList.add('content-text');
		div.innerText = 'WebSocket connection error, try again another time.';
		document.body.appendChild(div);
	});
}

function submitUsername() {
	let btn = document.querySelector('#submit-username-btn');

	let username = btn.parentElement.querySelector('input').value.trim().toLowerCase();
	if (!username.length) Popup.toastPopup('Username cannot be blank');
	else if (!/^[a-z0-9]+$/i.test(username)) Popup.toastPopup('Username must only contain alphanumeric characters');
	else {
		btn.style.filter = 'brightness(0.5)';
		btn.style.cursor = 'not-allowed';

		if (ws) ws.send(JSON.stringify({tag: 'requestUsername', data: username}));
	}
}

function receiveUsername(data) {
	let btn = document.querySelector('#submit-username-btn');

	if (!data.status) {
		Popup.toastPopup(data.data);
	} else {
		let parent = document.querySelector('#username-div');
		Utils.clearDiv(parent);
		let div = document.createElement('div');
		div.classList.add('content-header-2');
		div.innerHTML = 'Playing as: <span style="color: var(--color_red);">' + data.data + '</span>';
		parent.appendChild(div);
		username = data.data;

		getLobbies();
		handlers['lobbyRefresh'] = setInterval(getLobbies, 2000);

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

function updateLobbies(data) {
	let lobbySelect = document.querySelector('#lobby-select');
	Utils.clearDiv(lobbySelect);

	if (!data.data.length) {
		let div = document.createElement('div');
		div.classList.add('content-text');
		div.innerText = 'No lobbies available. Create your own!';
		lobbySelect.appendChild(div);
	}

	for (let server of data.data) {
		// console.log(server);
		let container = document.createElement('div');
		container.classList.add('lobby-tile', 'content-container-vertical');

		let div1 = document.createElement('div');
		div1.classList.add('content-header-2');
		div1.innerText = server.name;

		let div2 = document.createElement('div');
		div2.classList.add('content-text');
		div2.innerText = 'Players: ' + server.connected.length;

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

		[div1, div2, div3, div4, div5, div6].forEach(e => container.appendChild(e));

		lobbySelect.appendChild(container);
		container.data = server;

		let popup = document.querySelector('#popup-load-lobby');
		container.onclick = function() {
			document.querySelector('#load-lobby-name').innerText = server.name;
			document.querySelector('#load-lobby-creation').innerText = 'Created: ' + parseTime(server.time) + '\nBy: ' + server.creator;
			document.querySelector('#load-lobby-host').innerText = 'Host: ' + server.host;
			document.querySelector('#load-lobby-users').innerText = 'Players: ' + server.connected.length;

			let list = document.querySelector('#load-lobby-user-list');
			Utils.clearDiv(list);
			for (let user of server.connected) {
				let div = document.createElement('div');
				div.classList.add('content-text');
				div.innerText = (user == server.host ? '⭐ ' : '') + user;
				list.appendChild(div);
			}

			popup.querySelector('.button-positive-2').onclick = function() {
				ws.send(JSON.stringify({tag: 'joinLobby', data: server}));
			};

			Popup.popup(popup);
		};
	}

	// TODO update lobbies onscreen
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

	document.querySelector('#username-div').style.display = 'none';
	document.querySelector('#lobby-menu').style.display = 'none';
	document.querySelector('#lobby').style.display = null;

	// console.log('Joined Lobby', data);
	let server = data.data;
	document.querySelector('#lobby-name').innerText = 'Lobby [' + server.name + ']';
	document.querySelector('#lobby-creation').innerText = 'Created: ' + parseTime(server.time) + '\nBy: ' + server.creator;
	document.querySelector('#lobby-host').innerText = 'Host: ' + server.host;
	document.querySelector('#lobby-users').innerText = 'Players: ' + server.connected.length;

	document.querySelector('#btn-start-game').style.display = (server.host == username) ? null : 'none';

	let list = document.querySelector('#lobby-user-list');
	Utils.clearDiv(list);
	for (let user of server.connected) {
		let div = document.createElement('div');
		div.classList.add('content-text');
		div.innerText = (user == server.host ? '⭐ ' : '') + user + (user == username ? ' ⇐ You' : '');
		list.appendChild(div);
	}

	if (handlers['lobbyRefresh']) clearInterval(handlers['lobbyRefresh']);
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
	handlers['lobbyRefresh'] = setInterval(getLobbies, 2000);

	document.querySelector('#username-div').style.display = null;
	document.querySelector('#lobby-menu').style.display = null;
	document.querySelector('#lobby').style.display = 'none';
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
	document.querySelector('#title').style.display = 'none';
	document.querySelector('#lobby').style.display = 'none';
	document.querySelector('#game').style.display = null;
	drawGUI(data);
}

function sendCommand() {
	let data = document.querySelector('#game-console-input').value.trim();
	if (data) {
		let code = document.createElement('code');
		code.innerText = '>> ' + data;
		document.querySelector('#game-console-output').appendChild(code);
		ws.send(JSON.stringify({tag: 'sendCommand', data: data}));
		document.querySelector('#game-console-input').value = '';
	}
}

function receiveCommand(data) {
	let output = document.querySelector('#game-console-output');
	console.log(data.data);
	for (let log of data.data) {
		let code = document.createElement('code');
		if (!data.status) code.style.color = 'var(--color_red)';
		code.textContent = log;
		output.appendChild(code);
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

	let div = document.createElement('div');
	let span1 = document.createElement('span');

	let span1Username = document.createElement('b');
	span1Username.classList.add('username');
	span1Username.innerText = data.data.username;
	span1.appendChild(span1Username);

	let span1Text = document.createElement('span');
	span1Text.textContent = '\u00A0\u00A0' + data.data.text;
	span1.appendChild(span1Text);
	div.appendChild(span1);

	let span2 = document.createElement('span');
	span2.innerText = parseTime(data.data.time, date = false);
	div.appendChild(span2);
	output.appendChild(div);
	if (!Math.floor(output.scrollHeight - output.scrollTop - output.clientHeight)) div.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function drawGUI(data) { // TODO
	console.log('drawGUI', data);
	
	let gameData = data.data.gameData;
	let server = data.data.serverData;

	// Clear GUI
	let leaderboard = document.querySelector('#game-gui-leaderboard');

	Utils.clearDiv(document.querySelector('#game-gui-hands-self'));
	Utils.clearDiv(document.querySelector('#game-gui-hands'));
	Utils.clearDiv(leaderboard);

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

	// Init values
	document.querySelector('#game-gui-round-count').innerText = gameData.round;
	document.querySelector('#game-gui-limit-count').innerText = gameData.losingThreshold;
	document.querySelector('#game-gui-game-state').innerText = gameData.gameState;

	let absMaxScore = Math.abs(gameData.scores[0]);
	for (let i of gameData.scores) {
		if (Math.abs(i) > absMaxScore) absMaxScore = Math.abs(i);
	}

	for (let i = 0; i < gameData.turnOrder.length; i++) {
		let div1 = document.createElement('div');
		div1.classList.add('player-' + i);
		div1.style.gridRow = (i + 1) + ' / ' + (i + 2);

		for (let j = 0; j < 3; j++) {
			let div2 = document.createElement('div');
			div2.classList.add('content-container-text');
			div2.style.gridColumn = (j + 1) + ' / ' + (j + 2);

			if (j == 0) {
				if (username == gameData.turnOrder[i]) div2.innerText = '⭐';
			} else if (j == 1) {
				div2.innerText = gameData.turnOrder[i];
			} else {
				div2.innerText = gameData.scores[i];
				let percent = Math.abs(gameData.scores[i]) / absMaxScore * 100;
				div2.style.color = 'color-mix(in srgb, var(--color_' + ((gameData.scores[i] < 0) ? 'red' : 'green') + ') ' + percent + '%, var(--color_black))';
			}
			div1.appendChild(div2);
		}
		leaderboard.appendChild(div1);
	}

	for (let e of document.querySelectorAll('.game-gui-server-owner')) e.innerText = server.host;
}

function toggleDebug(data) {
	let debugElements = document.querySelectorAll('.debug');
	for (let e of debugElements) {
		e.classList.toggle('shown');
	}
}
