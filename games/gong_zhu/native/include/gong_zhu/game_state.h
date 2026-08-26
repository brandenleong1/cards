#include <cstdint>
#include <string>

#pragma once


namespace cards {
namespace gong_zhu {

struct GameSettings {
	std::string spectatorPolicy = "disallowed";
	int64_t     losingThreshold = -1000;
	bool        expose3 = false;
	bool        zhuYangManJuan = false;
	bool        allowCustomSeed = false;
	int64_t     customSeed = 0;
};

enum class GameState : uint8_t {
#define GAME_STATE(e) e,
#include "gong_zhu/game_state.def"
#undef GAME_STATE
};

std::string to_string(const GameState& gameState) noexcept;

} // namespace gong_zhu
} // namespace cards
