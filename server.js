const express = require('express');
const path = require('path');
const http = require('http');
const ws = require('ws');

const app = express();
const port = process.env.PORT || 8080;

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });


wss.on('connection', function connection(ws) {
	ws.on('message', function incoming(data) {
		wss.clients.forEach(function each(client) {
			if (client !== ws && client.readyState === WebSocket.OPEN) {
				client.send(data);
			}
		});
	});
});

server.listen(port, function() {
	console.log(`Server is listening on ${port}!`);
})






app.use(express.static(__dirname));
app.use(express.json());

app.listen(PORT, () => {
	console.log('Server started at http://localhost:' + PORT);
});

app.get('/', function(req, res) {
	res.sendFile(path.join(__dirname, '/index.html'));
});

// app.get("/msg", (req, res, next) => {
// 	res.json({"message": "Hello, World!"});
// });

// app.post("/msg", (req, res, next) => {
// 	const message = req.body.message;
// 	console.log(req, message);
// 	res.json({"receivedMessage": message});
// });

app.get('/api', function(req, res) {
	res.send((new Date()).toLocaleTimeString());
});

app.get('/game/gong_zhu', function(req, res) {
	res.sendFile(path.join(__dirname, '/games/game_gong_zhu.html'));
});