const card2Rank = 'A23456789JQK'.split('');
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

    return String.fromCodePoint(blockStart + rank + (suit * 16));
}