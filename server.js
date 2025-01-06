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

		switch (data.tag) {
			case 'requestUsername':
				let res = game.gong_zhu.addUser(data.data);
				if (res[0]) ws.username = res[1];
				ws.send(JSON.stringify({tag: 'receiveUsername', status: res[0], data: res[1]}));
				break;
		}
	});

	ws.send(JSON.stringify({tag: 'receiveLobbies', data: {userCount: wss.clients.size}}));
});

server.listen(app_port);






// app.use(express.static(__dirname));
// app.use(express.json());

// app.listen(app_port, () => {
// 	console.log('Server started at http://localhost:' + app_port);
// });

// app.get('/', function(req, res) {
// 	res.sendFile(path.join(__dirname, '/index.html'));
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