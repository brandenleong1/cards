#include <type_traits>

#include "common/cards.h"


std::string Card::toString() const noexcept {
	if (this->isHidden) {
		return "??";
	}

	switch (this->cardId) {
		case 52: {
			return "-★";
		}
		case 53: {
			return "-☆";
		}
		default: {
			const std::string rank = Card::card2Rank[cardId % 13];
			const std::string suit = Card::card2Suit[cardId / 13];

			return rank + suit;
		}
	}
}

CardRank Card::getCardRank() const noexcept {
	switch (this->cardId) {
		case 52: {
			return CardRank::rBlackJoker;
		}
		case 53: {
			return CardRank::rColoredJoker;
		}
		default: {
			return static_cast<CardRank>(this->cardId % 13);
		}
	}
}

CardSuit Card::getCardSuit() const noexcept {
	switch (this->cardId) {
		case 52: {
			[[ fallthrough ]];
		}
		case 53: {
			return CardSuit::Joker;
		}
		default: {
			return static_cast<CardSuit>(this->cardId / 13);
		}
	}
}

bool Card::operator==(const Card& other) const noexcept {
	return this->cardId == other.cardId;
}

bool Card::operator<(const Card& other) const noexcept {
	return this->cardId < other.cardId;
}

bool Card::operator>(const Card& other) const noexcept {
	return this->cardId > other.cardId;
}
