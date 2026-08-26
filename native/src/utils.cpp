#include <random>

#include "cards/utils.h"


namespace cards {

[[ nodiscard ]] std::vector<Card> initDeck() {
	constexpr size_t numCards = 52;

	std::vector<Card> result;
	result.reserve(numCards);

	for (size_t i = 0; i < numCards; i++) {
		result.emplace_back(i);
	}

	return result;
}

[[ nodiscard ]] std::vector<Card> filterByRank(const Card& card, const std::vector<Card>& arr) {
	std::vector<Card> result;
	result.reserve(arr.size());

	if (card.getCardId() >= 52) {
		std::copy_if(arr.begin(), arr.end(), std::back_inserter(result), [](const Card& c) -> bool {
			return c.getCardId() >= 52;
		});
	} else {
		std::copy_if(arr.begin(), arr.end(), std::back_inserter(result), [&card](const Card& c) -> bool {
			return (c.getCardId() < 52) && (card.getCardId() % 13 == c.getCardId() % 13);
		});
	}

	return result;
}

[[ nodiscard ]] std::vector<Card> filterBySuit(const Card& card, const std::vector<Card>& arr) {
	std::vector<Card> result;
	result.reserve(arr.size());

	if (card.getCardId() >= 52) {
		std::copy_if(arr.begin(), arr.end(), std::back_inserter(result), [](const Card& c) -> bool {
			return c.getCardId() >= 52;
		});
	} else {
		std::copy_if(arr.begin(), arr.end(), std::back_inserter(result), [&card](const Card& c) -> bool {
			return (c.getCardId() < 52) && (card.getCardId() / 13 == c.getCardId() / 13);
		});
	}

	return result;
}

namespace {

// TODO: get rid of static generator
std::random_device rd;
std::mt19937 rng(rd());

} // namespace

[[ nodiscard ]] std::vector<Card> shuffleArray(const std::vector<Card>& arr) {
	std::vector<Card> result(arr);

	std::shuffle(result.begin(), result.end(), rng);

	return result;
}

} // namespace cards
