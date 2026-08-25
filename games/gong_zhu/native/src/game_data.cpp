#include <algorithm>
#include <array>
#include <cassert>
#include <execution>
#include <iterator>
#include <set>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "common/utils.h"
#include "gong_zhu/game_data.h"


namespace gong_zhu {

static const GameData defaultState;

void GameData::resetRoundData() {
	this->decks = defaultState.decks;
	this->hands = defaultState.hands;
	this->stacks = defaultState.stacks;

	this->decks.assign(this->numDecks, initDeck());
	this->hands.resize(this->turnOrder.size());

	for (uint8_t i = 0; i < this->numDecks; i++) {
		this->decks[i] = shuffleArray(this->decks[i]);
	}

	for (size_t i = 0; i < this->scores.size(); i++) {
		std::get<1>(this->scores[i]) = 0;
	}

	std::apply([](auto&&... arg) -> void {
		(arg.clear(), ...);
	}, this->stacks);
}

void GameData::clearGameData() {
	this->gameState = defaultState.gameState;
	this->decks = defaultState.decks;
	this->turnOrder = defaultState.turnOrder;
	this->needToAct = defaultState.needToAct;
	this->hands = defaultState.hands;
	this->stacks = defaultState.stacks;
	this->scores = defaultState.scores;
}

void GameData::initGameData() {
	this->hands.assign(this->turnOrder.size(), Hand());
	this->scores.assign(this->turnOrder.size(), {0, 0});
	this->needToAct.assign(this->turnOrder.size(), false);

	this->gameState = GameState::LEADERBOARD;
	this->round = 0;
	this->turnFirstIdx = 0;
}

std::vector<Message> GameData::initGame(const std::vector<Player>& newTurnOrder) {
	this->turnOrder = newTurnOrder;

	this->resetRoundData();
	this->initGameData();
	
	this->currentFrame = 0;

	return this->gameOFL();
}

std::vector<Message> GameData::gameNSL() {
	GameState& state = this->gameState;

	if (state == GameState::SHOW_3) {
		state = GameState::SHOW_ALL;
	} else if (state == GameState::SHOW_ALL) {
		state = GameState::PLAY_0;
	} else if (state == GameState::PLAY_0) {
		state = GameState::PLAY_1;
	} else if (state == GameState::PLAY_1) {
		state = GameState::PLAY_2;
	} else if (state == GameState::PLAY_2) {
		state = GameState::PLAY_3;
	} else if (state == GameState::PLAY_3) {
		const bool allFinishedPlaying = std::all_of(this->hands.begin(), this->hands.end(),
			[](const Hand& hand) -> bool {
				return hand.toPlay.size() == 0;
			}
		);
		if (allFinishedPlaying) {
			const bool anyLost = std::any_of(this->scores.begin(), this->scores.end(),
				[this](const std::tuple<int64_t, int64_t>& score) -> bool {
					return (std::get<0>(score) + std::get<1>(score)) <= this->settings.losingThreshold;
				}
			);
			if (anyLost) {
				state = GameState::LEADERBOARD;
			} else {
				state = GameState::SCORE;
			}
		} else {
			state = GameState::PLAY_0;
		}
	} else if (state == GameState::SCORE) {
		if (this->settings.expose3) {
			state = GameState::SHOW_3;
		} else {
			state = GameState::SHOW_ALL;
		}
	} else if (state == GameState::LEADERBOARD) {
		if (this->settings.expose3) {
			state = GameState::SHOW_3;
		} else {
			state = GameState::SHOW_ALL;
		}
	}

	this->currentFrame++;
	return this->gameOFL();
}

std::vector<Message> GameData::gameOFL() {
	const size_t numPlayers = this->turnOrder.size();
	const GameState& state = this->gameState;
	std::vector<Message> ret;

	if (state == GameState::SHOW_3) {
		this->resetRoundData();
		this->round++;

		for (size_t i = 0; i < numPlayers; i++) {
			for (size_t j = 0; j < 3; j++) {
				this->hands[i].toPlay.push_back(std::move(this->decks[0].back()));
				this->decks[0].pop_back();
			}

			this->needToAct[i] = true;
		}
	} else if (state == GameState::SHOW_ALL) {
		if (!this->settings.expose3) {
			this->resetRoundData();
			this->round++;
		}

		while (this->decks[0].size()) {
			for (size_t i = 0; i < numPlayers; i++) {
				this->hands[i].toPlay.push_back(std::move(this->decks[0].back()));
				this->decks[0].pop_back();
			}
		}

		this->needToAct.assign(numPlayers, true);
	} else if (state == GameState::PLAY_0) {
		const auto isHandNotPlayed = [](const Hand& hand) constexpr -> bool {
			return !hand.played.has_value();
		};
		if (std::get<0>(this->stacks).size() == 0 && std::all_of(this->hands.begin(), this->hands.end(), isHandNotPlayed)) {
			const auto it = std::find_if(this->hands.begin(), this->hands.end(), [](const Hand& hand) -> bool {
				const bool has2Clubs = std::find(hand.toPlay.begin(), hand.toPlay.end(), 1) != hand.toPlay.end();
				return has2Clubs;
			});
			assert(it != this->hands.end() && "[gong_zhu::GameData::gameOFL] Cannot find 2 of Spades to start game");
			this->turnFirstIdx = static_cast<size_t>(std::distance(this->hands.begin(), it));
		}
		this->needToAct.assign(numPlayers, false);
		this->needToAct[this->turnFirstIdx] = true;

		ret.emplace_back(
			/* content = */ "Started Trick " + \
				std::to_string((std::get<0>(this->stacks).size() / 4) + 1) + \
				"; Player [" + this->turnOrder[this->turnFirstIdx].getName() + \
				"] leads...",
			/* toAll = */ true
		);
	} else if (state == GameState::PLAY_1) {
		this->needToAct.assign(numPlayers, false);
		this->needToAct[(this->turnFirstIdx + 1) % numPlayers] = true;
	} else if (state == GameState::PLAY_2) {
		this->needToAct.assign(numPlayers, false);
		this->needToAct[(this->turnFirstIdx + 2) % numPlayers] = true;
	} else if (state == GameState::PLAY_3) {
		this->needToAct.assign(numPlayers, false);
		this->needToAct[(this->turnFirstIdx + 3) % numPlayers] = true;
	} else if (state == GameState::SCORE) {
		for (size_t i = 0; i < this->scores.size(); i++) {
			ret.emplace_back(
				/* content = */ "Player [" + \
					this->turnOrder[i].getName() + \
					"] receives " + \
					std::string(std::get<1>(this->scores[i]) > 0 ? "+" : "") + \
					std::to_string(std::get<1>(this->scores[i])),
				/* toAll = */ true
			);
		}
	} else if (state == GameState::LEADERBOARD) {
		const bool anyLost = std::any_of(this->scores.begin(), this->scores.end(), [this](const std::tuple<int64_t, int64_t>& score) -> bool {
			return std::get<0>(score) <= this->settings.losingThreshold;
		});
		if (anyLost) {
			for (size_t i = 0; i < numPlayers; i++) {
				ret.emplace_back(
					/* content = */ "Player [" + \
						this->turnOrder[i].getName() + "] " + \
						std::string(std::get<0>(this->scores[i]) <= this->settings.losingThreshold ? "loses" : "survives") + " ↦ " + \
						std::to_string(std::get<0>(this->scores[i])) + " pts",
					/* toAll = */ true
				);
			}
		}
	}

	return ret;
}

std::unordered_set<Card> GameData::getLegalMoves(const size_t turnOrderIdx) const {
	const int64_t numPlayers = static_cast<int64_t>(this->turnOrder.size());
	const size_t relativeIdx = static_cast<size_t>((((static_cast<int64_t>(turnOrderIdx) - static_cast<int64_t>(this->turnFirstIdx)) % numPlayers) + numPlayers) % numPlayers);

	assert(turnOrderIdx < this->hands.size() && "[gong_zhu::GameData::getLegalMoves] turnOrderIdx must be less than or equal to this->hands.size()");

	std::unordered_set<Card> playableCards;
	std::unordered_set<Card> shownCards(this->hands[turnOrderIdx].shown.begin(), this->hands[turnOrderIdx].shown.end());

	if (relativeIdx == 0) {
		std::unordered_map<CardSuit, size_t> cardSuitCounts;
		for (const Card& card : this->hands[turnOrderIdx].toPlay) {
			cardSuitCounts[card.getCardSuit()]++;
		}

		for (const Card& card : this->hands[turnOrderIdx].toPlay) {
			if (shownCards.count(card) == 0 || cardSuitCounts[card.getCardSuit()] == 1) {
				playableCards.insert(card);
			}
		}
	} else {
		assert(this->hands[this->turnFirstIdx].played.has_value() && "[gong_zhu::GameData::getLegalMoves] Cannot get legal moves if the trick leader has not played");
		const std::vector<Card> filtered = filterBySuit(this->hands[this->turnFirstIdx].played.value(), this->hands[turnOrderIdx].toPlay);
		if (filtered.size() == 1) {
			// Even if it's shown, it's the only card playable -- must play
			playableCards.insert(filtered[0]);
		} else if (filtered.size() != 0) {
			// Play any non-shown card that matches the trick suit
			for (const Card& card : filtered) {
				if (shownCards.count(card) == 0) {
					playableCards.insert(card);
				}
			}
		} else {
			// Play anything -- no matching cards
			for (const Card& card : this->hands[turnOrderIdx].toPlay) {
				playableCards.insert(card);
			}
		}
	}

	return playableCards;
}

int64_t GameData::getScoreFromCards(const std::vector<Card>& cards) const {
	int64_t score = 0;

	const std::set<Card> cardSet(cards.begin(), cards.end());
	std::set<Card> heartSet;

	for (uint8_t i = 0; i < 13; i++) {
		heartSet.insert(heartSet.end(), Card(static_cast<CardRank>(i), CardSuit::Hearts));
	}

	std::unordered_map<Card, uint8_t> modifiers;

	for (const Card showableCard : {Card(CardRank::rQ, CardSuit::Spades), Card(CardRank::rA, CardSuit::Hearts), Card(CardRank::rJ, CardSuit::Diamonds), Card(CardRank::r10, CardSuit::Clubs)}) {
		const std::vector<std::tuple<Card, uint8_t>>& shownCards = std::get<1>(this->stacks);
		const auto it = std::find_if(shownCards.begin(), shownCards.end(), [&showableCard](const std::tuple<Card, uint8_t>& card) -> bool {
			return std::get<0>(card) == showableCard;
		});

		modifiers[showableCard] = (it == shownCards.end()) ? 1 : std::get<1>(*it);
	}

	if (std::includes(cardSet.begin(), cardSet.end(), heartSet.begin(), heartSet.end())) {
		score += modifiers[Card(CardRank::rA, CardSuit::Hearts)] * 200;

		if (this->settings.zhuYangManJuan) {
			if (cardSet.count(Card(CardRank::rQ, CardSuit::Spades)) && cardSet.count(Card(CardRank::rJ, CardSuit::Diamonds))) {
				score += modifiers[Card(CardRank::rQ, CardSuit::Spades)] * 100;
			}
		} else {
			if (cardSet.count(Card(CardRank::rQ, CardSuit::Spades))) {
				score += modifiers[Card(CardRank::rQ, CardSuit::Spades)] * 100;
			}
		}
	} else {
		if (cardSet.count(Card(CardRank::rQ, CardSuit::Spades))) {
			score += modifiers[Card(CardRank::rQ, CardSuit::Spades)] * -100;
		}
		if (cardSet.count(Card(CardRank::rA, CardSuit::Hearts))) {
			score += modifiers[Card(CardRank::rA, CardSuit::Hearts)] * -50;
		}
		if (cardSet.count(Card(CardRank::rK, CardSuit::Hearts))) {
			score += modifiers[Card(CardRank::rA, CardSuit::Hearts)] * -40;
		}
		if (cardSet.count(Card(CardRank::rQ, CardSuit::Hearts))) {
			score += modifiers[Card(CardRank::rA, CardSuit::Hearts)] * -30;
		}
		if (cardSet.count(Card(CardRank::rJ, CardSuit::Hearts))) {
			score += modifiers[Card(CardRank::rA, CardSuit::Hearts)] * -20;
		}
		for (uint8_t i = 5; i <= 10; i++) {
			if (cardSet.count(Card(static_cast<CardRank>(i - 1), CardSuit::Hearts))) {
				score += modifiers[Card(CardRank::rA, CardSuit::Hearts)] * -10;
			}
		}
		if (cardSet.count(Card(CardRank::rJ, CardSuit::Diamonds))) {
			score += modifiers[Card(CardRank::rJ, CardSuit::Diamonds)] * 100;
		}
	}

	if (cardSet.count(Card(CardRank::r10, CardSuit::Clubs))) {
		if (cardSet.size() == 1) {
			score += modifiers[Card(CardRank::r10, CardSuit::Clubs)] * 50;
		} else {
			score *= modifiers[Card(CardRank::r10, CardSuit::Clubs)] * 2;
		}
	}

	return score;
}

// TODO
std::tuple<bool, std::vector<Message>> GameData::applyCommand(
	[[maybe_unused]] const size_t turnOrderIdx,
	[[maybe_unused]] const std::string& command,
	[[maybe_unused]] std::vector<Player>* newTurnOrder
) {
	std::tuple<bool, std::vector<Message>> ret;

	return ret;
}

GameData GameData::obfuscateGameData(const size_t turnOrderIdx) const {
	GameData gameData(*this);

	hideAllCards(gameData.decks);
	hideAllCards(std::get<0>(gameData.stacks));

	std::vector<std::tuple<Card, uint8_t>>& shownCards = std::get<1>(gameData.stacks);
	shownCards.erase(std::remove_if(
		shownCards.begin(),
		shownCards.end(),
		[&gameData](const std::tuple<Card, uint8_t>& card) -> bool {
			return (
				(gameData.gameState == GameState::SHOW_3 && std::get<1>(card) == 4) ||
				(gameData.gameState == GameState::SHOW_ALL && std::get<1>(card) == 2)
			);
		}
	), shownCards.end());

	std::unordered_set<Card> shownCardsSet;
	std::transform(
		shownCards.begin(),
		shownCards.end(),
		std::inserter(shownCardsSet, shownCardsSet.end()),
		[](const std::tuple<Card, uint8_t>& card) -> Card {
			return std::get<0>(card);
		}
	);

	for (size_t i = 0; i < gameData.hands.size(); i++) {
		Hand& hand = gameData.hands[i];

		if (i != turnOrderIdx) {
			hideAllCards(hand.toPlay);
			
			std::vector<Card>& shown = hand.shown;
			shown.erase(std::remove_if(shown.begin(), shown.end(), [&shownCardsSet](const Card& card) -> bool {
				return shownCardsSet.count(card) == 0;
			}), shown.end());
		}
	}

	return gameData;
}

} // namespace gong_zhu
