#include "gong_zhu/game_state.h"


namespace cards {
namespace gong_zhu {

std::string to_string(const GameState& gameState) noexcept {
	switch (gameState) {
#define GAME_STATE(e) \
		case GameState::e: \
			return #e;
#include "gong_zhu/game_state.def"
#undef GAME_STATE
		default:
			return "";
	}
}

GameState gameStateFromString(const std::string& s) noexcept {
#define GAME_STATE(e) if (s == #e) return GameState::e;
#include "gong_zhu/game_state.def"
#undef GAME_STATE
	return GameState::UNDEFINED;
}

} // namespace gong_zhu
} // namespace cards
