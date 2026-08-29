#include <cstdint>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>

#include "cards/cards.h"
#include "cards/message.h"
#include "cards/player.h"
#include "gong_zhu/game_data.h"
#include "gong_zhu/hand.h"

#pragma once


namespace py = pybind11;

namespace cards {
namespace gong_zhu {

py::object toPy(const Card& card);
py::object toPy(const Player& player);
py::object toPy(const Message& message);
py::object toPy(const std::vector<bool>& v);
py::object toPy(const std::tuple<int64_t, int64_t>& score);
py::object toPy(const std::tuple<Card, uint8_t>& shown);
py::object toPy(const Hand& hand);
py::object toPy(const GameData& gd);

template <typename T>
py::object toPy(const std::vector<T>& v) {
	py::list out;
	for (const T& e : v) {
		out.append(toPy(e));
	}
	return out;
}

} // namespace gong_zhu
} // namespace cards
