#include <cstdint>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>

#include "cards/cards.h"
#include "cards/message.h"
#include "cards/player.h"

#pragma once


namespace py = pybind11;

namespace cards {

py::object toPy(const Card& card);
py::object toPy(const Player& player);
py::object toPy(const Message& message);
py::object toPy(const std::vector<bool>& v);
py::object toPy(const std::tuple<int64_t, int64_t>& pair);
py::object toPy(const std::tuple<Card, uint8_t>& pair);

template <typename T>
py::object toPy(const std::vector<T>& v) {
	py::list ret;
	for (const T& e : v) {
		ret.append(toPy(e));
	}
	return ret;
}

} // namespace cards
