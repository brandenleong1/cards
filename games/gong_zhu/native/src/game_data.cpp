#include <algorithm>
#include <array>
#include <cassert>
#include <execution>
#include <iterator>
#include <optional>
#include <set>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "cards/command.h"
#include "cards/utils.h"
#include "gong_zhu/game_data.h"


namespace cards {
namespace gong_zhu {

static const GameData defaultState;

void GameData::resetRoundData(Shuffler& shuffler) {
	this->decks = defaultState.decks;
	this->hands = defaultState.hands;
	this->stacks = defaultState.stacks;

	this->decks.assign(this->numDecks, initDeck());
	this->hands.resize(this->turnOrder.size());

	for (uint8_t i = 0; i < this->numDecks; i++) {
		this->decks[i] = shuffleArray(this->decks[i], shuffler);
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

std::vector<Message> GameData::initGame(
	const std::vector<Player>& newTurnOrder,
	Shuffler& shuffler
) {
	this->turnOrder = newTurnOrder;

	this->resetRoundData(shuffler);
	this->initGameData();

	this->currentFrame = 0;

	return this->gameOFL(shuffler);
}

std::vector<Message> GameData::gameNSL(Shuffler& shuffler) {
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
	return this->gameOFL(shuffler);
}

std::vector<Message> GameData::gameOFL(Shuffler& shuffler) {
	const size_t numPlayers = this->turnOrder.size();
	const GameState& state = this->gameState;
	std::vector<Message> ret;

	if (state == GameState::SHOW_3) {
		this->resetRoundData(shuffler);
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
			this->resetRoundData(shuffler);
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
				const bool has2Spades = std::find(hand.toPlay.begin(), hand.toPlay.end(), 1) != hand.toPlay.end();
				return has2Spades;
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

namespace {

inline bool isScoringCard(const uint8_t cardId) {
	return cardId == 11 || (cardId >= 13 && cardId <= 25) || cardId == 36 || cardId == 48;
}

} // namespace

std::tuple<int8_t, std::vector<Message>> GameData::applyCommand(
	const size_t turnOrderIdx,
	const std::string& command,
	Shuffler& shuffler,
	const std::vector<Player>* const newTurnOrder
) {
	std::vector<Message> ret;
	int8_t status = 1;

	ParsedCommand parsedCommand = parseCommand(command);
	if (parsedCommand.command.empty()) {
		return {false, ret};
	}

	const std::string commandUpper = toUpper(parsedCommand.command[0]);

	const int64_t numPlayers = static_cast<int64_t>(this->turnOrder.size());
	const size_t relativeIdx = static_cast<size_t>((((static_cast<int64_t>(turnOrderIdx) - static_cast<int64_t>(this->turnFirstIdx)) % numPlayers) + numPlayers) % numPlayers);

	const auto gotBadCommandInState = [&ret, &status, &commandUpper, this]() -> void {
		ret.emplace_back(
			/* content = */ "Cannot issue command [" + commandUpper + "] in state [" + to_string(this->gameState) + "]",
			/* toAll */ false
		);
		status = 0;
	};

	if (commandUpper == "DEAL") {
		if (newTurnOrder != nullptr) {
			this->clearGameData();
			this->turnOrder = *newTurnOrder;
			this->initGameData();
		}
		if (
			this->gameState == GameState::LEADERBOARD ||
			this->gameState == GameState::SCORE
		) {
			const std::vector<Message> messages = this->gameNSL(shuffler);
			ret.insert(ret.end(), messages.begin(), messages.end());
			ret.emplace_back(
				/* content = */ "Started Round " + std::to_string(this->round),
				/* toAll = */ true
			);
		} else {
			gotBadCommandInState();
		}
	} else if (commandUpper == "PLAY") {
		if (
			this->gameState != GameState::SHOW_3 &&
			this->gameState != GameState::SHOW_ALL &&
			this->gameState != static_cast<GameState>(static_cast<std::underlying_type_t<GameState>>(GameState::PLAY_0) + relativeIdx)
		) {
			gotBadCommandInState();
		} else if (parsedCommand.command.size() < 2) {
			ret.emplace_back(
				/* content = */ "Insufficient arguments for [" + commandUpper + "] (need 1)",
				/* toAll = */ false
			);
			status = 0;
		} else {
			Hand& myHand = this->hands[turnOrderIdx];
			std::vector<size_t> args;
			int64_t invalidArgIdx = -1;
			for (size_t i = 1; i < parsedCommand.command.size(); i++) {
				const std::optional<int64_t> val = toInt(parsedCommand.command[i]);
				if (!val.has_value() || val.value() < 0 || val.value() >= static_cast<int64_t>(myHand.toPlay.size())) {
					invalidArgIdx= static_cast<int64_t>(i);
					break;
				}
				args.push_back(static_cast<size_t>(val.value()));
			}

			if (invalidArgIdx != -1) {
				ret.emplace_back(
					/* content = */ "Invalid argument at index [" + std::to_string(invalidArgIdx) + "] for [" + commandUpper + "] (argument \"" + parsedCommand.command[static_cast<size_t>(invalidArgIdx)] + "\")",
					/* toAll = */ false
				);
				status = 0;
			} else if (
				this->gameState == GameState::SHOW_3 ||
				this->gameState == GameState::SHOW_ALL
			) {
				int64_t invalidShowArgIdx = -1;
				for (size_t i = 0; i < args.size(); i++) {
					const Card& cardToShow = myHand.toPlay[args[i]];
					if (
						!isScoringCard(cardToShow.getCardId()) ||
						(cardToShow.getCardId() >= 14 && cardToShow.getCardId() <= 25)
					) {
						invalidShowArgIdx = static_cast<int64_t>(i + 1);
						break;
					}
				}

				if (invalidShowArgIdx != -1) {
					ret.emplace_back(
						/* content = */ "Invalid argument at index [" + std::to_string(invalidShowArgIdx) + "] for [" + commandUpper + "] (argument \"" + parsedCommand.command[static_cast<size_t>(invalidShowArgIdx)] + "\")",
						/* toAll = */ false
					);
					status = 0;
				} else {
					const uint8_t multiplier = (this->gameState == GameState::SHOW_3) ? 4 : 2;
					for (const size_t arg : args) {
						const Card& card = myHand.toPlay[arg];
						if (std::find(myHand.shown.begin(), myHand.shown.end(), card) == myHand.shown.end()) {
							std::get<1>(this->stacks).emplace_back(card, multiplier);
							myHand.shown.push_back(card);
							ret.emplace_back(
								/* content = */ "Shown card [" + card.toString() + "] for x" + std::to_string(multiplier) + " value",
								/* toAll = */ false
							);
						}
					}
				}
			} else if (args.size() != 1) {
				ret.emplace_back(
					/* content = */ "Too many arguments for [" + commandUpper + "] (max 1)",
					/* toAll = */ false
				);
				status = 0;
			} else {
				const std::unordered_set<Card> legalMoves = this->getLegalMoves(turnOrderIdx);
				int64_t invalidPlayArgIdx = -1;
				for (size_t i = 0; i < args.size(); i++) {
					if (legalMoves.count(myHand.toPlay[args[i]]) == 0) {
						invalidPlayArgIdx = static_cast<int64_t>(i + 1);
						break;
					}
				}

				if (invalidPlayArgIdx != -1) {
					ret.emplace_back(
						/* content = */ "Invalid argument at index [" + std::to_string(invalidPlayArgIdx) + "] for [" + commandUpper + "] (argument \"" + parsedCommand.command[static_cast<size_t>(invalidPlayArgIdx)] + "\")",
						/* toAll = */ false
					);
					status = 0;
				} else {
					const Card cardToPlay = myHand.toPlay[args[0]];
					ret.emplace_back(
						/* content = */ "Player [" + this->turnOrder[turnOrderIdx].getName() + "] played card [" + cardToPlay.toString() + "]",
						/* toAll = */ true
					);

					const auto it = std::find(myHand.shown.begin(), myHand.shown.end(), cardToPlay);
					if (it != myHand.shown.end()) {
						myHand.shown.erase(it);
					}
					myHand.toPlay.erase(myHand.toPlay.begin() + static_cast<int64_t>(args[0]));
					myHand.played = cardToPlay;

					if (this->gameState == GameState::PLAY_3) {
						assert(this->hands[this->turnFirstIdx].played.has_value() && "[gong_zhu::GameData::applyCommand] Could not find played card during collection");

						const Card& playedLeadingCard = this->hands[this->turnFirstIdx].played.value();
						const CardSuit trickSuit = playedLeadingCard.getCardSuit();

						size_t winnerIdx = this->turnFirstIdx;
						CardRank winnerRank = playedLeadingCard.getCardRank();

						for (size_t i = 0; i < this->turnOrder.size(); i++) {
							assert(this->hands[i].played.has_value() && "[gong_zhu::GameData::applyCommand] Could not find played card during collection");

							const Card& playedCard = this->hands[i].played.value();

							const CardRank myRank = playedCard.getCardRank();
							const CardSuit mySuit = playedCard.getCardSuit();

							if (mySuit == trickSuit) {
								if (myRank == CardRank::rA || (winnerRank != CardRank::rA && myRank > winnerRank)) {
									winnerIdx = i;
									winnerRank = myRank;
								}
							}
						}

						this->turnFirstIdx = winnerIdx;

						std::vector<Card> collected;
						for (size_t i = 0; i < this->turnOrder.size(); i++) {
							assert(this->hands[i].played.has_value() && "[gong_zhu::GameData::applyCommand] Could not find played card during collection");

							const Card& playedCard = this->hands[i].played.value();

							if (isScoringCard(playedCard.getCardId())) {
								this->hands[winnerIdx].collected.push_back(playedCard);
								collected.push_back(playedCard);
							}
						}

						std::get<1>(this->scores[winnerIdx]) = this->getScoreFromCards(this->hands[winnerIdx].collected);

						std::string collectedStr;
						for (size_t i = 0; i < collected.size(); i++) {
							collectedStr += (i != 0) ? ", " : "";
							collectedStr += collected[i].toString();
						}

						ret.emplace_back(
							/* content = */ "Player [" + this->turnOrder[winnerIdx].getName() + "] wins with [" + this->hands[winnerIdx].played.value().toString() + "] and takes [" + collectedStr + "]",
							/* toAll = */ true
						);

						for (Hand& hand : this->hands) {
							std::get<0>(this->stacks).push_back(hand.played.value());
							hand.played.reset();
						}

						for (Hand& hand : this->hands) {
							hand.shown.erase(std::remove_if(
								hand.shown.begin(), hand.shown.end(),
								[&trickSuit](const Card& card) -> bool {
									return card.getCardSuit() == trickSuit;
								}
							), hand.shown.end());
						}

						const bool allEmpty = std::all_of(
							this->hands.begin(), this->hands.end(),
							[](const Hand& hand) -> bool {
								return hand.toPlay.empty();
							}
						);

						if (allEmpty) {
							for (std::tuple<int64_t, int64_t>& score : this->scores) {
								std::get<0>(score) += std::get<1>(score);
							}
						}
					}

					const std::vector<Message> messages = this->gameNSL(shuffler);
					ret.insert(ret.end(), messages.begin(), messages.end());
				}
			}
		}
	} else if (commandUpper == "PASS") {
		if (parsedCommand.command.size() > 1) {
			ret.emplace_back(
				/* content = */ "Too many arguments for [" + commandUpper + "] (need 0)",
				/* toAll = */ false
			);
			status = 0;
		} else if (
			this->gameState != GameState::SHOW_3 &&
			this->gameState != GameState::SHOW_ALL
		) {
			gotBadCommandInState();
		} else {
			this->needToAct[turnOrderIdx] = false;
			if (std::none_of(this->needToAct.begin(), this->needToAct.end(), [](const bool needsToAct) -> bool {
				return needsToAct;
			})) {
				const std::vector<Message> messages = this->gameNSL(shuffler);
				ret.insert(ret.end(), messages.begin(), messages.end());
			}
		}
	}

	return {status, ret};
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
} // namespace cards
