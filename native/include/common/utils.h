#include <algorithm>
#include <execution>
#include <type_traits>
#include <vector>

#include "common/cards.h"

#pragma once

[[ nodiscard ]] std::vector<Card> initDeck();

[[ nodiscard ]] std::vector<Card> filterByRank(const Card& card, const std::vector<Card>& arr);
[[ nodiscard ]] std::vector<Card> filterBySuit(const Card& card, const std::vector<Card>& arr);
[[ nodiscard ]] std::vector<Card> shuffleArray(const std::vector<Card>& arr);

template <typename T>
inline void hideAllCards(std::vector<T>& arr) {
	std::for_each(std::execution::par, arr.begin(), arr.end(), [](T& subArr) {
		hideAllCards(subArr);
	});
}

inline void hideAllCards(Card& card) {
	card.setHidden(true);
}

