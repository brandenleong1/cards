function reqListener (data) {
	console.log(data);
}

document.querySelector('#game-gong-zhu-btn').onclick = function() {
	window.location = './games/game_gong_zhu.html';
	// console.log(window.location);

	// let xhr = new XMLHttpRequest();
	// xhr.addEventListener('load', reqListener);
	// xhr.open('GET', '/game/gong_zhu');
	// xhr.send();
};