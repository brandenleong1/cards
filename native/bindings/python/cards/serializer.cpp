#include <cstdint>
#include <tuple>

#include "cards/serializer.h"


namespace cards {

py::object toPy(const Card& card) {
	if (card.getIsHidden()) {
		return py::none();
	}
	return py::cast(static_cast<int64_t>(card.getCardId()));
}

py::object toPy(const std::vector<bool>& v) {
	py::list ret;
	for (bool b : v) {
		ret.append(b ? 1 : 0);
	}
	return ret;
}

py::object toPy(const std::tuple<int64_t, int64_t>& pair) {
	py::list ret;
	ret.append(std::get<0>(pair));
	ret.append(std::get<1>(pair));
	return ret;
}

py::object toPy(const std::tuple<Card, uint8_t>& pair) {
	py::list ret;
	ret.append(static_cast<int64_t>(std::get<0>(pair).getCardId()));
	ret.append(static_cast<int64_t>(std::get<1>(pair)));
	return ret;
}

py::object toPy(const Player& player) {
	return py::cast(player.getName());
}

py::object toPy(const Message& message) {
	py::dict event;
	event["msg"] = message.content;
	event["toAll"] = message.toAll;
	return event;
}

} // namespace cards
