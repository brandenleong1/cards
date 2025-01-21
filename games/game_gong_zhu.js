let ws;
// let ws = new WebSocket('ws' + window.location.href.substring(window.location.href.indexOf(':')));

let username;
let server;
let handlers = {};

let messageDecoder = {
	'broadcastedMessage':	(data) => {Popup.toastPopup(data.data);},
	'receiveUsername':		(data) => {receiveUsername(data);},
	'updateUserCount':		(data) => {updateUserCount(data);},
	'updateLobbies':		(data) => {updateLobbies(data);},
	'createdLobby':			(data) => {joinLobby(data);},
	'joinedLobby':			(data) => {showLobby(data);},
	'leftLobby':			(data) => {leftLobby(data);}
};


function initWebSocket() {
	ws = new WebSocket('http://localhost:8080');

	ws.addEventListener('message', function(message) {
		console.log(message);
	
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
		let parent = btn.parentElement;
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

function parseTime(time) {
	let date = new Date(time);

	let year = date.getFullYear().toString(10).substring(2);
	let month = (date.getMonth() + 1).toString(10).padStart(2, '0');
	let day = date.getDate().toString(10).padStart(2, '0');
	let hour = date.getHours().toString(10).padStart(2, '0');
	let minutes = date.getMinutes().toString(10).padStart(2, '0');

	return month + '/' + day + '/' + year + ' ' + hour + ':' + minutes;
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
		console.log(server);
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

	document.querySelector('#lobby-menu').style.display = 'none';
	document.querySelector('#lobby').style.display = null;

	// console.log('Joined Lobby', data);
	server = data.data;
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

	document.querySelector('#lobby-menu').style.display = null;
	document.querySelector('#lobby').style.display = 'none';
}

function startGame() {
	ws.send(JSON.stringify({tag: 'startGame'}));
}

function startedGame(data) { // TODO Init Game
	document.querySelector('#title').style.display = 'none';
	document.querySelector('#lobby').style.display = 'none';
	document.querySelector('#game').style.display = null;
}