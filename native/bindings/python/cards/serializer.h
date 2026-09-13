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

} // namespace cards
