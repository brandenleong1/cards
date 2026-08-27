#include <cassert>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#pragma once


namespace cards {

enum class CardRank : uint8_t {
	rA,
	r2,
	r3,
	r4,
	r5,
	r6,
	r7,
	r8,
	r9,
	r10,
	rJ,
	rQ,
	rK,
	rBlackJoker,
	rColoredJoker
};

enum class CardSuit : uint8_t {
	Spades,
	Hearts,
	Diamonds,
	Clubs,
	Joker
};

class Card {
private:
	inline static const std::vector<std::string> card2Rank = {
		"A", "2", "3", "4", "5", "6", "7",
		"8", "9", "10", "J", "Q", "K"
	};
	inline static const std::vector<std::string> card2Suit = {
		"♠", "♥", "♦", "♣"
	};

private:
	uint8_t cardId;
	bool isHidden = false;

public:
	constexpr Card(const uint8_t cardId) : cardId(cardId) {
		assert(this->cardId <= 53 && "[Card::Card] cardId must be <= 53");
	}
	constexpr Card(const CardRank& rank, const CardSuit& suit) :
		cardId(static_cast<uint8_t>(static_cast<std::underlying_type_t<CardRank>>(rank) + (static_cast<std::underlying_type_t<CardSuit>>(suit) * 13))) {
	}


	std::string toString() const noexcept;

	inline uint8_t getCardId() const noexcept {
		return this->cardId;
	}
	void setHidden(const bool hidden) noexcept {
		this->isHidden = hidden;
	}
	inline bool getIsHidden() const noexcept {
		return this->isHidden;
	}
	CardRank getCardRank() const noexcept;
	CardSuit getCardSuit() const noexcept;


	bool operator==(const Card& other) const noexcept;
	bool operator<(const Card& other) const noexcept;
	bool operator>(const Card& other) const noexcept;
};

} // namespace cards

namespace std {

template <>
struct hash<cards::Card> {
	size_t operator()(const cards::Card& card) const noexcept {
		const size_t h1 = std::hash<uint8_t>{}(card.getCardId());

		return h1;
	}
};

} // namespace std
