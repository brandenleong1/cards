#include "gong_zhu/game_state.h"


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

} // namespace gong_zhu
