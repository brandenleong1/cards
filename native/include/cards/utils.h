#include <algorithm>
#include <cctype>
#include <execution>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

#include "cards/cards.h"
#include "cards/rng.h"

#pragma once


namespace cards {

[[ nodiscard ]] std::vector<Card> initDeck();

[[ nodiscard ]] std::vector<Card> filterByRank(const Card& card, const std::vector<Card>& arr);
[[ nodiscard ]] std::vector<Card> filterBySuit(const Card& card, const std::vector<Card>& arr);
[[ nodiscard ]] std::vector<Card> shuffleArray(const std::vector<Card>& arr, Shuffler& shuffler);

template <typename T>
inline void hideAllCards(std::vector<T>& arr) {
	std::for_each(std::execution::par, arr.begin(), arr.end(), [](T& subArr) {
		hideAllCards(subArr);
	});
}

inline void hideAllCards(Card& card) {
	card.setHidden(true);
}

inline std::string toLower(const std::string_view s) {
	std::string res;
	res.reserve(s.size());

	std::transform(s.begin(), s.end(), std::back_inserter(res), [](const unsigned char c) -> unsigned char {
		return static_cast<unsigned char>(std::tolower(static_cast<int>(c)));
	});

	return res;
}

inline std::string toUpper(const std::string_view s) {
	std::string res;
	res.reserve(s.size());

	std::transform(s.begin(), s.end(), std::back_inserter(res), [](const unsigned char c) -> unsigned char {
		return static_cast<unsigned char>(std::toupper(static_cast<int>(c)));
	});

	return res;
}

inline std::optional<int64_t> toInt(const std::string& s) {
	if (s.empty()) {
		return std::nullopt;
	}
	
	try {
		const int64_t v = std::stoll(s);
		return v;
	} catch (...) {
		return std::nullopt;
	}
}

} // namespace cards
