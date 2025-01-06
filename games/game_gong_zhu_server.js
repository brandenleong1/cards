let users = new Set();
let usernames = new Map();

function addUser(username) {
	if (usernames.has(username)) {
		if (usernames[username] == 1000) return [0, 'Username saturated'];
		else {
			while (true) {
				let t = Math.floor(Math.random() * 10000).toString().padStart(4, '0');
				let user = username + '#' + t;
				if (!users.has(user)) {
					return [1, user];
				}
			}
		}
	} else {
		let t = Math.floor(Math.random() * 10000).toString().padStart(4, '0');
		let user = username + '#' + t;
		return [1, user];
	}
}

module.exports = {addUser};