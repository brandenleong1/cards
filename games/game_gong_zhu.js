let ws;
// let ws = new WebSocket('ws' + window.location.href.substring(window.location.href.indexOf(':')));

let username;

function initWebSocket() {
	ws = new WebSocket('http://localhost:8080');

	ws.addEventListener('message', function(message) {
		console.log(message);
	
		let data = JSON.parse(message.data);
		switch (data.tag) {
			case 'receiveUsername':
				receiveUsername(data);
				break;
			case 'updateUserCount':
				updateUserCount(data);
				break;
			case 'updateLobbies':
				updateLobbies(data);
				break;
			case 'createdLobby':
				joinLobby(data);
				break;
			case 'joinedLobby':
				showLobby(data);
				break;
		}
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
		div.innerText = 'WebSocket connection error, please reload.';
		document.body.appendChild(div);
	});
}

document.querySelector('#submit-username-btn').onclick = function() {
	let username = this.parentElement.querySelector('input').value.trim();
	if (!username.length) Popup.toastPopup('Username cannot be blank');
	else if (!/^[a-z0-9]+$/i.test(username)) Popup.toastPopup('Username must only contain alphanumeric characters');
	else {
		this.style.filter = 'brightness(0.5)';
		this.style.cursor = 'not-allowed';

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

		[div1, div2, div3, div4, div5].forEach(e => container.appendChild(e));

		lobbySelect.appendChild(container);
	}

	// TODO update lobbies onscreen
}

function createLobby() {
	let vals = Array.from(document.querySelectorAll('#popup-create-lobby input')).map(e => e.value.trim());

	for (let val of vals) {
		if (!val.length) {
			Popup.toastPopup('One or more fields blank');
			return;
		}
	}

	ws.send(JSON.stringify({tag: 'createLobby', data: {name: vals[0], time: Date.now(), creator: username}}));
}

function joinLobby(data) {
	// TODO check for closed lobby while on popup screen

	if (!data.status) Popup.toastPopup(data.data);
	ws.send(JSON.stringify({tag: 'joinLobby', data: data.data}));
}

function showLobby(data) {
	if (!data.status) {
		Popup.toastPopup(data.data);
		return;
	}

	console.log('Joined Lobby', data);
}