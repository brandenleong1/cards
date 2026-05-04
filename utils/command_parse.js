export function parseCommand(command, tagArgCounts = {}) {
	let split = command.trim().match(/[\""].+?[\""]|[^\s]+/g);
	let parsed = [];
	let tags = {};

	let tag = null;
	let tagArgs = [];
	for (let word of split) {
		if (!tag) {
			if (word.startsWith('-')) {
				tag = word;
			} else {
				if (word.startsWith('"') && word.endsWith('"')) {
					parsed.push(word.substring(1, word.length - 1));
				} else {
					parsed.push(word);
				}
			}
		} else {
			if (word.startsWith('"') && word.endsWith('"')) {
				tagArgs.push(word.substring(1, word.length - 1));
			} else {
				tagArgs.push(word);
			}
		}

		if (tag && tagArgs.length == (tagArgCounts[tag] | 0)) {
			tags[tag] = tagArgs;
			tag = null;
			tagArgs = [];
		}
	}
	if (tag) {
		while (tagArgs.length < (tagArgCounts[tag] | 0)) tagArgs.push('');
		tags[tag] = tagArgs;
	}

	return {command: parsed, tags: tags};
}