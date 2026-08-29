#include "serializer.h"


namespace cards {
namespace gong_zhu {

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

py::object toPy(const std::vector<bool>& v) {
	py::list out;
	for (bool b : v) {
		out.append(b ? 1 : 0);
	}
	return out;
}

py::object toPy(const std::tuple<int64_t, int64_t>& score) {
	py::list pair;
	pair.append(std::get<0>(score));
	pair.append(std::get<1>(score));
	return pair;
}

py::object toPy(const std::tuple<Card, uint8_t>& shown) {
	py::list pair;
	pair.append(static_cast<int64_t>(std::get<0>(shown).getCardId()));
	pair.append(static_cast<int64_t>(std::get<1>(shown)));
	return pair;
}

py::object toPy(const Hand& hand) {
	py::list played;
	if (hand.played.has_value()) {
		played.append(toPy(hand.played.value()));
	}
	py::list entry;
	entry.append(toPy(hand.toPlay));
	entry.append(toPy(hand.shown));
	entry.append(toPy(hand.collected));
	entry.append(played);
	return entry;
}

py::object toPy(const GameData& gd) {
	py::list stacks;
	stacks.append(toPy(std::get<0>(gd.stacks)));   // discard: hidden -> [None, ...]
	stacks.append(toPy(std::get<1>(gd.stacks)));   // shown: [[id, val], ...]

	py::dict state;
	state["gameState"] = to_string(gd.gameState);
	state["hands"] = toPy(gd.hands);
	state["scores"] = toPy(gd.scores);
	state["stacks"] = stacks;
	state["needToAct"] = toPy(gd.needToAct);
	state["turnOrder"] = toPy(gd.turnOrder);
	state["turnFirstIdx"] = static_cast<int64_t>(gd.turnFirstIdx);
	state["currentFrame"] = gd.currentFrame;
	state["numDecks"] = static_cast<int64_t>(gd.numDecks);
	return state;
}

} // namespace gong_zhu
} // namespace cards
