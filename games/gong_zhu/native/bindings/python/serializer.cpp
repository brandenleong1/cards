#include "gong_zhu/game_state.h"

#include "serializer.h"


namespace cards {
namespace gong_zhu {

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
