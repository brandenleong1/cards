#include <optional>
#include <vector>

#include "common/cards.h"

#pragma once


namespace gong_zhu {

struct Hand {
	std::vector<Card>   toPlay; // hidden + shown
	std::vector<Card>   shown;
	std::vector<Card>   collected;
	std::optional<Card> played;
};

} // namespace gong_zhu
