#include <cstdint>
#include <tuple>
#include <vector>

#include "gong_zhu/game_state.h"

#include "serializer.h"


namespace cards {
namespace gong_zhu {

Napi::Object toJs(Napi::Env env, const GameData& gameData) {
	Napi::Object ret = Napi::Object::New(env);

	ret.Set("gameState", Napi::String::New(env, gameData.gameState == GameState::UNDEFINED ? std::string() : to_string(gameData.gameState)));
	ret.Set("numDecks", Napi::Number::New(env, gameData.numDecks));
	ret.Set("minPlayers", Napi::Number::New(env, gameData.minPlayers));
	ret.Set("maxPlayers", Napi::Number::New(env, gameData.maxPlayers));

	ret.Set("decks", toJs(env, gameData.decks));
	ret.Set("turnOrder", toJs(env, gameData.turnOrder));
	ret.Set("turnFirstIdx", Napi::Number::New(env, static_cast<double>(gameData.turnFirstIdx)));

	ret.Set("needToAct", toJs(env, gameData.needToAct));

	Napi::Array hands = Napi::Array::New(env, gameData.hands.size());
	for (uint32_t i = 0; i < gameData.hands.size(); i++) {
		const Hand& hand = gameData.hands[i];
		Napi::Array played = Napi::Array::New(env, hand.played.has_value() ? 1 : 0);
		if (hand.played.has_value()) {
			played.Set(uint32_t(0), toJs(env, hand.played.value()));
		}
		Napi::Array entry = Napi::Array::New(env, 4);
		entry.Set(uint32_t(0), toJs(env, hand.toPlay));
		entry.Set(uint32_t(1), toJs(env, hand.shown));
		entry.Set(uint32_t(2), toJs(env, hand.collected));
		entry.Set(uint32_t(3), played);
		hands.Set(i, entry);
	}
	ret.Set("hands", hands);

	// stacks = [discard, [[card, val], ...]]
	Napi::Array stacks = Napi::Array::New(env, 2);
	stacks.Set(uint32_t(0), toJs(env, std::get<0>(gameData.stacks)));
	stacks.Set(uint32_t(1), toJs(env, std::get<1>(gameData.stacks)));
	ret.Set("stacks", stacks);

	ret.Set("scores", toJs(env, gameData.scores));

	ret.Set("round", Napi::Number::New(env, static_cast<double>(gameData.round)));

	Napi::Object settings = Napi::Object::New(env);
	settings.Set("spectatorPolicy", Napi::String::New(env, gameData.settings.spectatorPolicy));
	settings.Set("losingThreshold", Napi::Number::New(env, static_cast<double>(gameData.settings.losingThreshold)));
	settings.Set("expose3", Napi::Boolean::New(env, gameData.settings.expose3));
	settings.Set("zhuYangManJuan", Napi::Boolean::New(env, gameData.settings.zhuYangManJuan));
	settings.Set("allowCustomSeed", Napi::Boolean::New(env, gameData.settings.allowCustomSeed));
	settings.Set("customSeed", Napi::Number::New(env, static_cast<double>(gameData.settings.customSeed)));
	ret.Set("settings", settings);

	ret.Set("currentFrame", Napi::BigInt::New(env, gameData.currentFrame));

	return ret;
}

GameData gameDataFromJs(Napi::Object o) {
	GameData gameData;

	gameData.gameState = gameStateFromString(o.Get("gameState").As<Napi::String>().Utf8Value());
	gameData.numDecks = static_cast<uint8_t>(o.Get("numDecks").As<Napi::Number>().Uint32Value());
	gameData.minPlayers = static_cast<uint16_t>(o.Get("minPlayers").As<Napi::Number>().Uint32Value());
	gameData.maxPlayers = static_cast<uint16_t>(o.Get("maxPlayers").As<Napi::Number>().Uint32Value());

	const Napi::Array decks = o.Get("decks").As<Napi::Array>();
	for (uint32_t i = 0; i < decks.Length(); i++) {
		gameData.decks.push_back(cardsFromJs(decks.Get(i).As<Napi::Array>()));
	}

	gameData.turnOrder = playersFromJs(o.Get("turnOrder").As<Napi::Array>());
	gameData.turnFirstIdx = static_cast<size_t>(o.Get("turnFirstIdx").As<Napi::Number>().Int64Value());

	const Napi::Array needToAct = o.Get("needToAct").As<Napi::Array>();
	for (uint32_t i = 0; i < needToAct.Length(); i++) {
		gameData.needToAct.push_back(needToAct.Get(i).As<Napi::Number>().Int32Value() != 0);
	}

	const Napi::Array hands = o.Get("hands").As<Napi::Array>();
	for (uint32_t i = 0; i < hands.Length(); i++) {
		const Napi::Array entry = hands.Get(i).As<Napi::Array>();
		Hand hand;
		hand.toPlay = cardsFromJs(entry.Get(uint32_t(0)).As<Napi::Array>());
		hand.shown = cardsFromJs(entry.Get(uint32_t(1)).As<Napi::Array>());
		hand.collected = cardsFromJs(entry.Get(uint32_t(2)).As<Napi::Array>());
		Napi::Array played = entry.Get(uint32_t(3)).As<Napi::Array>();
		if (played.Length() > 0) {
			hand.played = cardFromJs(played.Get(uint32_t(0)));
		}
		gameData.hands.push_back(std::move(hand));
	}

	const Napi::Array stacks = o.Get("stacks").As<Napi::Array>();
	std::get<0>(gameData.stacks) = cardsFromJs(stacks.Get(uint32_t(0)).As<Napi::Array>());
	const Napi::Array shownStack = stacks.Get(uint32_t(1)).As<Napi::Array>();
	for (uint32_t i = 0; i < shownStack.Length(); i++) {
		Napi::Array shown = shownStack.Get(i).As<Napi::Array>();
		std::get<1>(gameData.stacks).emplace_back(
			cardFromJs(shown.Get(uint32_t(0))),
			static_cast<uint8_t>(shown.Get(uint32_t(1)).As<Napi::Number>().Uint32Value())
		);
	}

	const Napi::Array scores = o.Get("scores").As<Napi::Array>();
	for (uint32_t i = 0; i < scores.Length(); i++) {
		const Napi::Array score = scores.Get(i).As<Napi::Array>();
		gameData.scores.emplace_back(
			score.Get(uint32_t(0)).As<Napi::Number>().Int64Value(),
			score.Get(uint32_t(1)).As<Napi::Number>().Int64Value()
		);
	}

	gameData.round = static_cast<uint64_t>(o.Get("round").As<Napi::Number>().Int64Value());

	const Napi::Object settings = o.Get("settings").As<Napi::Object>();
	gameData.settings.spectatorPolicy = settings.Get("spectatorPolicy").As<Napi::String>().Utf8Value();
	gameData.settings.losingThreshold = settings.Get("losingThreshold").As<Napi::Number>().Int64Value();
	gameData.settings.expose3 = settings.Get("expose3").As<Napi::Boolean>().Value();
	gameData.settings.zhuYangManJuan = settings.Get("zhuYangManJuan").As<Napi::Boolean>().Value();
	gameData.settings.allowCustomSeed = settings.Get("allowCustomSeed").As<Napi::Boolean>().Value();
	gameData.settings.customSeed = settings.Get("customSeed").As<Napi::Number>().Int64Value();

	bool lossless = false;
	gameData.currentFrame = o.Get("currentFrame").As<Napi::BigInt>().Int64Value(&lossless);

	return gameData;
}

} // namespace gong_zhu
} // namespace cards
