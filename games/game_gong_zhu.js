let username;

document.querySelector('#submit-username-btn').onclick = function() {
	username = this.parentElement.querySelector('input').value.trim().toUpperCase();
	if (!username.length) Popup.toastPopup('Username cannot be blank');
	else {
		let parent = this.parentElement;
		Utils.clearDiv(parent);
		let div = document.createElement('div');
		div.classList.add('content-header-2');
		div.innerHTML = 'Playing as: <span style="color: var(--color_red);">' + username + '</span>';
		parent.appendChild(div);
	}
}