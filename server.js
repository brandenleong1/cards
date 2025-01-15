const express = require('express');
const path = require('path');
const http = require('http');
const ws = require('ws');

const game = {
	gong_zhu: require(path.resolve( __dirname, './games/game_gong_zhu_server.js'))
};

const app = express();
const app_port = process.env.PORT || 8080;

const server = http.createServer(app);
const wss = new ws.Server({server});

wss.on('connection', function(ws, req) {
	console.log('Connection:', req.url);
	console.log('Users:', wss.clients.size);

	ws.on('message', function(data, isBinary) {
		data = isBinary ? data : data.toString();
		data = JSON.parse(data);
		// console.log(data);

		let res;
		switch (data.tag) {	// TODO differentiate between independent game signals
			case 'requestUsername':
				res = game.gong_zhu.addUser(data.data);
				if (res[0]) {
					ws.username = res[1];
					game.gong_zhu.users[res[1]] = ws;
				}
				ws.send(JSON.stringify({tag: 'receiveUsername', status: res[0], data: res[1]}));
				break;
			case 'getLobbies':
				ws.send(JSON.stringify({tag: 'updateLobbies', data: game.gong_zhu.servers}));
				break;
			case 'createLobby':
				res = game.gong_zhu.addServer(data.data);
				ws.send(JSON.stringify({tag: 'createdLobby', status: res[0], data: res[1]}));
				break;
			case 'joinLobby':
				res = game.gong_zhu.joinServer(ws, data.data);
				if (res[0]) {
					for (let username of res[1].connected) {
						game.gong_zhu.users[username].send(JSON.stringify({tag: 'joinedLobby', status: res[0], data: res[1]}));
					}
					console.log('joinServer', res[1].connected);
					// ws.connected = data.data;
					// console.log('joinServer', ws.connected, data.data.connected);
				} else {
					ws.send(JSON.stringify({tag: 'joinedLobby', status: res[0], data: res[1]}));
				}
				break;
			case 'leaveLobby':
				if (ws.connected) {
					game.gong_zhu.leaveServer(ws, ws.connected);
					for (let username of ws.connected.connected) {
						let ws1 = game.gong_zhu.users[username];
						ws1.send(JSON.stringify({tag: 'joinedLobby', status: 1, data: ws.connected}));
					}
				}
				break;
		}
	});

	for (let ws of wss.clients) {
		ws.send(JSON.stringify({tag: 'updateUserCount', data: wss.clients.size}));
	}

	ws.on('close', function() {
		// TODO when host disconnects, pass to someone else
		console.log('Closed', this.username, this.connected);
		console.log(game.gong_zhu.servers);
		if (this.username) game.gong_zhu.removeUser(this.username);
		if (this.connected) {
			game.gong_zhu.leaveServer(this, this.connected);
			for (let username of this.connected.connected) {
				let ws = game.gong_zhu.users[username];
				ws.send(JSON.stringify({tag: 'joinedLobby', status: 1, data: this.connected}));
			}
		}
		console.log(game.gong_zhu.servers);
		for (let ws of wss.clients) {
			ws.send(JSON.stringify({tag: 'updateUserCount', data: wss.clients.size}));
		}
	});
});

server.listen(app_port);






// app.use(express.static(__dirname));
// app.use(express.json());

// app.listen(app_port, () => {
// 	console.log('Server started at http://localhost:' + app_port);
// });

// app.get('/', function(req, res) {
// 	console.log(req, req.body);
// 	// res.sendFile(path.join(__dirname, '/index.html'));
// });

// // app.get("/msg", (req, res, next) => {
// // 	res.json({"message": "Hello, World!"});
// // });

// // app.post("/msg", (req, res, next) => {
// // 	const message = req.body.message;
// // 	console.log(req, message);
// // 	res.json({"receivedMessage": message});
// // });

// app.get('/api', function(req, res) {
// 	res.send((new Date()).toLocaleTimeString());
// });

// app.get('/game/gong_zhu', function(req, res) {
// 	res.sendFile(path.join(__dirname, '/games/game_gong_zhu.html'));
// });