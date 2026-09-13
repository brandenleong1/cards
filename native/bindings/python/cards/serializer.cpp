#include <cstdint>

#include "cards/serializer.h"


namespace cards {

py::object toPy(const Card& card) {
	if (card.getIsHidden()) {
		return py::none();
	}
	return py::cast(static_cast<int64_t>(card.getCardId()));
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
