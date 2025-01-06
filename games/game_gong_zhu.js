const ws = new WebSocket('http://localhost:8080');
// const ws = new WebSocket('ws' + window.location.href.substring(window.location.href.indexOf(':')));

let username;

document.querySelector('#submit-username-btn').onclick = function() {
	let username = this.parentElement.querySelector('input').value.trim();
	if (!username.length) Popup.toastPopup('Username cannot be blank');
	else if (!/^[a-z0-9]+$/i.test(username)) Popup.toastPopup('Username must only contain alphanumeric characters');
	else {
		this.style.filter = 'brightness(0.5)';
		this.style.cursor = 'not-allowed';

		console.log('sending', JSON.stringify({tag: 'requestUsername', data: username}));
		ws.send(JSON.stringify({tag: 'requestUsername', data: username}));
	}
}


// ws.addEventListener('open', function() {
// 	ws.send('something');
// });

ws.addEventListener('message', function(message) {
	console.log(message);

	let data = JSON.parse(message.data);
	switch (data.tag) {
		case 'receiveUsername':
			receiveUsername(data);
			break;
		case 'receiveLobbies':
			updateLobbies(data);
			break;
	}
});

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
	}
	btn.style.cursor = null;
	btn.style.filter = null;
}

function refreshLobby() {
	ws.send(JSON.stringify({tag: 'refreshLobby'}));
}

function updateLobbies(data) {
	document.querySelector('#player-count span').innerText = data.data.userCount;

	// TODO update lobbies onscreen
}