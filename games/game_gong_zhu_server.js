let users = new Set();
let usernames = new Map();

let servers = [];

const Utils = {
	binarySearchIdx : function(list, item, compareFn = (a, b) => a - b) {
		let left = 0, right = list.length - 1;

		while (left <= right) {
			let mid = Math.floor((left + right) / 2);
			let compRes = compareFn(list[mid], item);
			if (isNaN(compRes)) {
				break;
			}

			if (compRes > 0) {
				right = mid - 1;
			} else if (compRes < 0) {
				left = mid + 1;
			} else {
				if (mid > 0 && compareFn(list[mid - 1], item) == 0) {
					right = mid - 1;
				} else {
					return mid;
				}
			}
		}

		return -1;
	},

	binaryInsert : function(list, item, compareFn = (a, b) => a - b) {
		let left = 0, right = list.length;
		let mid = 0;
		let compRes = 0;

		while (left < right) {
			mid = Math.floor((left + right) / 2);
			compRes = compareFn(list[mid], item);
			if (compRes > 0) {
				right = mid;
			} else if (compRes < 0) {
				left = mid + 1;
			} else {
				return -1;
			}
		}
		if (compRes < 0) mid++;

		list.splice(mid, 0, item);
		return mid;
	}
}

function addUser(username) {
	if (usernames.has(username)) {
		if (usernames[username] == 1000) return [0, 'Username saturated'];
		else {
			while (true) {
				let t = Math.floor(Math.random() * 10000).toString(10).padStart(4, '0');
				let user = username + '#' + t;
				if (!users.has(user)) {
					return [1, user];
				}
			}
		}
	} else {
		let t = Math.floor(Math.random() * 10000).toString(10).padStart(4, '0');
		let user = username + '#' + t;
		return [1, user];
	}
}

function removeUser(username) {
	if (!username) return;
	usernames[username.split('#', 1)[0]] -= 1;
	users.delete(username);
}

function addServer(data) {
	data.connected = [];
	let idx = Utils.binaryInsert(servers, data, function(a, b) {
		if (a.time != b.time) return b.time - a.time;
		else if (a.name != b.name) return a.name.localeCompare(b.name);
		else return a.creator.localeCompare(b.creator);
	});

	// console.log(servers);

	return [idx == -1 ? 0 : 1, data];
}

function getServer(serverData) {
	let idx = Utils.binarySearchIdx(servers, serverData, function(a, b) {
		if (a.time != b.time) return b.time - a.time;
		else if (a.name != b.name) return a.name.localeCompare(b.name);
		else return a.creator.localeCompare(b.creator);
	});

	if (idx == -1) return null;
	else return servers[idx];
}

function joinServer(ws, serverData) {
	// console.log(serverData);
	let idx = Utils.binarySearchIdx(servers, serverData, function(a, b) {
		if (a.time != b.time) return b.time - a.time;
		else if (a.name != b.name) return a.name.localeCompare(b.name);
		else return a.creator.localeCompare(b.creator);
	});

	if (idx == -1) return [0, 'Server does not exist'];
	else {
		Utils.binaryInsert(servers[idx].connected, ws, function(a, b) {
			return a.username.localeCompare(b.username);
		});
		ws.connected = serverData;
		return [1, servers[idx]];
	}
}

function leaveServer(ws, serverData) {
	let idx = Utils.binarySearchIdx(servers, serverData, function(a, b) {
		if (a.time != b.time) return b.time - a.time;
		else if (a.name != b.name) return a.name.localeCompare(b.name);
		else return a.creator.localeCompare(b.creator);
	});

	if (idx == -1) return [0, 'Server does not exist'];
	else {
		let idx2 = Utils.binarySearchIdx(servers[idx].connected, ws, function(a, b) {
			return a.username.localeCompare(b.username);
		});
		if (idx2 != -1) servers[idx].connected.splice(idx2, 1);
		if (!servers[idx].connected.length) {
			servers.splice(idx, 1);
			return [1, null];
		}
		return [1, servers[idx]];
	}
}

module.exports = {
	addUser, removeUser,
	servers, addServer, joinServer, leaveServer
};