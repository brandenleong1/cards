import * as crypto from 'crypto';

// ms: number
export function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

// type: string
export function storageAvailable(type) {
    let storage;
    try {
        storage = window[type];
        const x = '__storage_test__';
        storage.setItem(x, x);
        storage.removeItem(x);
        return true;
    }
    catch (e) {
        return e instanceof DOMException && (
            e.code === 22 ||
            e.code === 1014 ||
            e.name === 'QuotaExceededError' ||
            e.name === 'NS_ERROR_DOM_QUOTA_REACHED') &&
            (storage && storage.length !== 0);
    }
}

// e: any, ...class_constructors: [Class]
export function isClass(e, ...class_constructors) {
	if (typeof e == 'object') {
		for (let class_constructor of class_constructors) {
			if (e.constructor == class_constructor) {
				return true;
			}
		}
		return false;
	}
	return false;
}

// e: HTMLDivElement
export function clearDiv(e) {
	while (e.firstChild) {
    	e.removeChild(e.firstChild);
	}
}

// text: string, fileName: string
export function saveFile(text, fileName) {
	let file = new Blob([text], {type: 'text/plain; charset=utf-8,'});
	if (window.navigator.msSaveOrOpenBlob) {
		window.navigator.msSaveOrOpenBlob(file, fileName);
	} else {
		let a = document.createElement('a');
		let url = URL.createObjectURL(file);
		a.href = url;
		document.body.append(a);
		a.setAttribute('download', fileName);
		a.click();
		setTimeout(function() {
			document.body.removeChild(a);
			window.URL.revokeObjectURL(url);
		}, 0);
	}
}

// list: Array[T], item: T, compareFn: function(a: T, b: T): number
export function binarySearchIdx(list, item, compareFn = (a, b) => a - b) {
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
}

// list: Array[T], item: T, compareFn: function(a: T, b: T): number
export function binaryInsert(list, item, compareFn = (a, b) => a - b) {
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

// arr: Array[any]
export function shuffleArray(arr) {
	let arrN = structuredClone(arr);
	for (let i = arrN.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[arrN[i], arrN[j]] = [arrN[j], arrN[i]];
	}
	return arrN;
}

// arr: Array[Array[any]]
export function transpose2DArray(arr) {
	return arr[0].map((_, i) => arr.map((e) => e[i]));
}

// a: number, b: number, t: number
export function lerp(a, b, t) {
	return a + t * (b - a);
}

// arr: Array[any], indices: Array[int]
export function sortArray(arr, indices) {
	let newArr = [];
	let visited = (new Array(arr.length)).fill(0);
	for (let i of indices) {
		visited[i] = 1;
		newArr.push(arr[i]);
	}
	for (let i = 0; i < visited.length; i++) {
		if (!visited[i]) newArr.push(arr[i]);
	}
	return newArr;
}

export function nullify(arr) {
	for (let i = 0; i < arr.length; i++) {
		if (Array.isArray(arr[i])) {
			nullify(arr[i]);
		} else {
			arr[i] = null;
		}
	}
}

export function broadcastToConnected(users, server, data, ...ignoredUsernames) {
	for (let username of server.connected) {
		let ws = users.get(username);
		if (!ignoredUsernames.includes(username) && ws.active) ws.send(JSON.stringify(data));
	}
}

export function broadcastGameStateToConnected(users, server, obfuscateFunc = null) {
	let serverInfo = structuredClone(server);
	let gameData = structuredClone(server.gameData);
	delete serverInfo.gameData;

	for (let i = 0; i < server.connected.length; i++) {
		let username = server.connected[i];
		let ws = users.get(username);

		if (ws.active) {
			if (obfuscateFunc) {
				ws.send(JSON.stringify({
					tag: 'updateGUI',
					data: {
						gameData: obfuscateFunc(gameData, gameData.turnOrder.findIndex(e => e == username)),
						serverData: serverInfo
					}
				}));
			} else {
				ws.send(JSON.stringify({
					tag: 'updateGUI',
					data: {
						gameData: gameData,
						serverData: serverInfo
					}
				}));
			}
		}
	}
}

export function broadcastGameState(ws, server, obfuscateFunc = null) {
	let serverInfo = structuredClone(server);
	let gameData = structuredClone(server.gameData);
	delete serverInfo.gameData;

	let username = ws.username;

	if (obfuscateFunc) {
		ws.send(JSON.stringify({
			tag: 'updateGUI',
			data: {
				gameData: obfuscateFunc(gameData, gameData.turnOrder.findIndex(e => e == username)),
				serverData: serverInfo
			}
		}));
	} else {
		ws.send(JSON.stringify({
			tag: 'updateGUI',
			data: {
				gameData: gameData,
				serverData: serverInfo
			}
		}));
	}
}

export function generateSessionID(sessionIDs) {
	let sessionID = crypto.randomBytes(16).toString('hex');
	while (sessionIDs.has(sessionID)) sessionID = crypto.randomBytes(16).toString('hex');
	return sessionID;
}

export function purgeArchive(archive, olderThan = 60 * 1000) {
	let now = Date.now();
	let purged = new Set();
	archive.forEach((val, key) => {
		if (now - val > olderThan) {
			purged.add(key);
			archive.delete(key);
		}
	});

	return purged;
}