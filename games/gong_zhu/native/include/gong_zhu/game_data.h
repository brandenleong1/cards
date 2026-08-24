#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_set>

#include "common/cards.h"
#include "common/message.h"
#include "common/player.h"
#include "gong_zhu/game_state.h"
#include "gong_zhu/hand.h"

#pragma once


namespace gong_zhu {

struct GameData {
	GameState                           gameState = GameState::UNDEFINED;
	uint8_t                             numDecks = 1;
	uint16_t                            minPlayers = 4;
	uint16_t                            maxPlayers = 4;
	std::vector<std::vector<Card>>      decks;
	std::vector<Player>                 turnOrder;
	size_t                              turnFirstIdx = 0;
	std::vector<bool>                   needToAct;
	std::vector<Hand>                   hands;
	std::tuple<
		std::vector<Card>,              // discard
		std::vector<std::tuple<
			Card,                           // card
			uint8_t                         // value
		>>                              // shown
	>                                   stacks;
	std::vector<std::tuple<
		int64_t,                        // total
		int64_t                         // round delta
	>>                                  scores;
	uint64_t                            round = 0;
	GameSettings                        settings;
	int64_t                             currentFrame = -1;

public:
	void resetRoundData();
	void clearGameData();
	void initGameData();
	std::vector<Message> initGame(const std::vector<Player>& turnOrder);
	std::vector<Message> gameNSL();
	std::vector<Message> gameOFL();
	std::unordered_set<Card> getLegalMoves(const size_t turnOrderIdx) const;
	int64_t getScoreFromCards(const std::vector<Card>& cards) const;
	std::tuple<bool, std::vector<Message>> applyCommand(
		const size_t turnOrderIdx,
		const std::string& command,
		std::vector<Player>* newTurnOrder = nullptr
	);
	GameData obfuscateGameData(const size_t turnOrderIdx) const;
};

} // namespace gong_zhu
