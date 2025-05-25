const card2Rank = 'A 2 3 4 5 6 7 8 9 10 J Q K'.split(' ');
const card2Suit = '♠♥♦♣'.split('');

export function initDeck() {
    return new Array(52).fill(0).map((e, i) => i);
}

export function card2Str(card) {
    if (card == 52) return '-★';
    if (card == 53) return '-☆';

    let rank = card2Rank[card % 13];
    let suit = card2Suit[Math.floor(card / 13)];

    return rank + suit;
}

export function card2Unicode(card) {
    if (card == 52) return String.fromCodePoint(0x1f0bf);
    if (card == 53) return String.fromCodePoint(0x1f0df);

    const blockStart = 0x1f0a1;

    let rank = card % 13;
    let suit = Math.floor(card / 13);

	if (rank >= 11) rank++;

    return String.fromCodePoint(blockStart + rank + (suit * 16));
}

export function filterByRank(card, arr) {
	if (card >= 52) return arr.filter(e => e >= 52);
	return arr.filter(e => e < 52 && (e % 13 == card % 13));
}

export function filterBySuit(card, arr) {
	if (card >= 52) return arr.filter(e => e >= 52);
	return arr.filter(e => e < 52 && (Math.floor(e / 13) == Math.floor(card / 13)));
}
