#include <cstdint>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>

#include "cards/cards.h"
#include "gong_zhu/game_data.h"
#include "gong_zhu/hand.h"

#include "cards/serializer.h"

#pragma once


namespace py = pybind11;

namespace cards {
namespace gong_zhu {

using cards::toPy;

py::object toPy(const Hand& hand);
py::object toPy(const GameData& gd);

} // namespace gong_zhu
} // namespace cards
