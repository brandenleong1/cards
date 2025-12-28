async function init() {
	await initTheme(parseInt(Cookies.getCookie('themeID'), 10) || 0);

	let link = document.createElement('link');
	link.href = 'https://cdn.jsdelivr.net/gh/brandenleong1/utils@latest/themes/transition.css';
	link.rel = 'stylesheet';
	link.type = 'text/css';
	document.head.append(link);

	for (let e of document.querySelectorAll('[data-anim]')) {
		if ([Animate.fadeIn].includes(eval(e.dataset.anim))) await Animate.remove(e);
	}

	document.querySelector('#help').onclick = () => {Popup.popup(document.querySelector('#popup-help'))};
	document.querySelector('#settings').onclick = () => {Popup.popup(document.querySelector('#popup-settings'))};
	initWebSocket();
}

function initTheme(id = 0) {
	Themes.createThemeCSS(id);
	document.querySelector('#theme-btn').onclick = changeTheme;
	document.querySelector('#theme-btn').themeId = id;
	document.querySelector('#theme-btn').themeLabel1Shown = false;
	document.querySelector('#theme-label-2').innerText = Themes.themes[id][0];
	document.querySelector('#theme-css').setAttribute('href', Themes.themes[id][1]);
	Cookies.setCookie('themeID', id, 5 * 365 * 24 * 60 * 60 * 1000);
}

async function changeTheme() {
	let btn = document.querySelector('#theme-btn');
	btn.onclick = null;
	btn.themeId = (btn.themeId + 1) % Themes.themes.length;

	let labels = btn.themeLabel1Shown ? [document.querySelector('#theme-label-1'), document.querySelector('#theme-label-2')] : [document.querySelector('#theme-label-2'), document.querySelector('#theme-label-1')];
	labels[1].innerText = Themes.themes[btn.themeId][0];
	await Animate.animateGroup([
		[labels[0], Animate.fadeOut, {shiftTo: UP}],
		[labels[1], Animate.fadeIn, {shiftFrom: DOWN}]
	]);
	document.querySelector('#theme-css').setAttribute('href', Themes.themes[btn.themeId][1]);
	Cookies.setCookie('themeID', btn.themeId, 5 * 365 * 24 * 60 * 60 * 1000);

	btn.themeLabel1Shown = !btn.themeLabel1Shown;
	await Animate.wait(1500);
	btn.onclick = changeTheme;
}

window.addEventListener('load', async function() {
	await Utils.sleep(100);
	document.querySelector('#loading-css').remove();
	document.querySelector('#loading').style.display = 'none';
	await init();
});