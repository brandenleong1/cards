let ws;

let messageDecoder = {
	'createConsole':	(data) => {createConsole(data);},
	'receiveCommand':	(data) => {receiveCommand(data);},
	'removeConsole':	(data) => {removeConsole(data);}
}

function initWebSocket() {
	let urlParams = new URLSearchParams(window.location.search);
	let url = urlParams.get('url');

	if (!url) {
		url = prompt('Enter WebSocket URL:');
	}

	try {
		ws = new WebSocket(url);

		ws.addEventListener('open', function(e) {
			console.log('Connected to [' + url + ']');
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
	} catch (e) {}
}

function sendCommand(id) {
	let consoleDiv = document.querySelector('#' + id);
	let msg = consoleDiv.querySelector('input[type=text]').value.trim();
	if (msg) {
		let code = document.createElement('code');
		code.innerText = '>> ' + msg;
		consoleDiv.querySelector('.console-output').append(code);
		ws.send(JSON.stringify({tag: 'sendCommand', id: id, data: msg}));
		consoleDiv.querySelector('input[type=text]').value = '';
		code.scrollIntoView({behavior: 'smooth', block: 'end'});
	}
}

function createConsole(data) {
	// data.data = {id: str}

	if (document.querySelector('#' + data.data.id)) {
		Popup.toastPopup('Could not create console with id [' + data.data.id + '] (already exists)');
		return;
	}

	let consoleDiv = document.createElement('div');
	consoleDiv.classList.add('content-container-vertical', 'console');
	consoleDiv.id = data.data.id;

	let headerDiv = document.createElement('div');
	headerDiv.classList.add('content-header');
	headerDiv.innerText = data.data.id;

	let outputDiv = document.createElement('div');
	outputDiv.classList.add('content-container-vertical', 'console-output');

	let inputLabel = document.createElement('label');

	let inputDiv = document.createElement('input');
	inputDiv.setAttribute('type', 'text');
	inputDiv.setAttribute('placeholder', 'Enter command...');
	inputDiv.addEventListener('keypress', function(e) {
		if (e.key == 'Enter') {
			e.preventDefault();
			sendCommand(data.data.id);
		}
	});

	inputLabel.append(inputDiv);

	consoleDiv.append(headerDiv, outputDiv, inputLabel);

	document.querySelector('#consoles-bounding').append(consoleDiv);
}

function receiveCommand(data) {
	// data.data = {id: str, msg: [str, ...], status: bool}

	let consoleDiv = document.querySelector('#' + data.data.id);
	if (!consoleDiv) {
		Popup.toastPopup('Message sent but could not receive to id [' + data.data.id + ']');
		return;
	}

	let output = consoleDiv.querySelector('.console-output')
	for (let log of data.data.msg) {
		let code = document.createElement('code');
		if (!data.data.status) code.style.color = 'var(--color_red)';
		code.textContent = log;
		output.append(code);
		code.scrollIntoView({behavior: 'smooth', block: 'end'});
	}
}

function removeConsole(data) {
	//data.data = {id: str}

	let consoleDiv = document.querySelector('#' + data.data.id);
	if (consoleDiv) {
		consoleDiv.remove();
	}
}
