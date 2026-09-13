#include "gong_zhu/game_state.h"

#include "serializer.h"


namespace cards {
namespace gong_zhu {

Napi::Value toJs(Napi::Env env, const Card& card) {
	if (card.getIsHidden()) {
		return env.Null();
	}
	return Napi::Number::New(env, card.getCardId());
}

Napi::String toJs(Napi::Env env, const Player& player) {
	return Napi::String::New(env, player.getName());
}

Napi::Object toJs(Napi::Env env, const Message& message) {
	Napi::Object ret = Napi::Object::New(env);

	ret.Set("msg", Napi::String::New(env, message.content));
	ret.Set("toAll", Napi::Boolean::New(env, message.toAll));

	return ret;
}

Napi::Object toJs(Napi::Env env, const ParsedCommand& parsedCommand) {
	Napi::Object ret = Napi::Object::New(env);

	Napi::Array command = Napi::Array::New(env, parsedCommand.command.size());
	for (uint32_t i = 0; i < parsedCommand.command.size(); i++) {
		command.Set(i, Napi::String::New(env, parsedCommand.command[i]));
	}

	Napi::Object tags = Napi::Object::New(env);
	for (const auto& [tag, args] : parsedCommand.tags) {
		Napi::Array argsArr = Napi::Array::New(env, args.size());
		for (uint32_t i = 0; i < args.size(); i++) {
			argsArr.Set(i, Napi::String::New(env, args[i]));
		}
		tags.Set(Napi::String::New(env, tag), argsArr);
	}

	ret.Set("command", command);
	ret.Set("tags", tags);

	return ret;
}

Napi::Object toJs(Napi::Env env, const GameData& gameData) {
	Napi::Object ret = Napi::Object::New(env);

	ret.Set("gameState", Napi::String::New(env, gameData.gameState == GameState::UNDEFINED ? std::string() : to_string(gameData.gameState)));
	ret.Set("numDecks", Napi::Number::New(env, gameData.numDecks));
	ret.Set("minPlayers", Napi::Number::New(env, gameData.minPlayers));
	ret.Set("maxPlayers", Napi::Number::New(env, gameData.maxPlayers));

	Napi::Array decks = Napi::Array::New(env, gameData.decks.size());
	for (uint32_t i = 0; i < gameData.decks.size(); i++) {
		decks.Set(i, toJs(env, gameData.decks[i]));
	}
	ret.Set("decks", decks);

	ret.Set("turnOrder", toJs(env, gameData.turnOrder));
	ret.Set("turnFirstIdx", Napi::Number::New(env, static_cast<double>(gameData.turnFirstIdx)));

	Napi::Array needToAct = Napi::Array::New(env, gameData.needToAct.size());
	for (uint32_t i = 0; i < gameData.needToAct.size(); i++) {
		needToAct.Set(i, Napi::Number::New(env, gameData.needToAct[i] ? 1 : 0));
	}
	ret.Set("needToAct", needToAct);

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
	Napi::Array shownStack = Napi::Array::New(env, std::get<1>(gameData.stacks).size());
	for (uint32_t i = 0; i < std::get<1>(gameData.stacks).size(); i++) {
		const std::tuple<Card, uint8_t>& shown = std::get<1>(gameData.stacks)[i];
		Napi::Array pair = Napi::Array::New(env, 2);
		pair.Set(uint32_t(0), Napi::Number::New(env, std::get<0>(shown).getCardId()));
		pair.Set(uint32_t(1), Napi::Number::New(env, std::get<1>(shown)));
		shownStack.Set(i, pair);
	}
	Napi::Array stacks = Napi::Array::New(env, 2);
	stacks.Set(uint32_t(0), toJs(env, std::get<0>(gameData.stacks)));
	stacks.Set(uint32_t(1), shownStack);
	ret.Set("stacks", stacks);

	Napi::Array scores = Napi::Array::New(env, gameData.scores.size());
	for (uint32_t i = 0; i < gameData.scores.size(); i++) {
		Napi::Array score = Napi::Array::New(env, 2);
		score.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(std::get<0>(gameData.scores[i]))));
		score.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(std::get<1>(gameData.scores[i]))));
		scores.Set(i, score);
	}
	ret.Set("scores", scores);

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

Card cardFromJs(Napi::Value value) {
	return Card(static_cast<uint8_t>(value.As<Napi::Number>().Uint32Value()));
}

std::vector<Card> cardsFromJs(Napi::Array arr) {
	std::vector<Card> ret;
	ret.reserve(arr.Length());
	for (uint32_t i = 0; i < arr.Length(); i++) {
		ret.push_back(cardFromJs(arr.Get(i)));
	}
	return ret;
}

Player playerFromJs(Napi::Value value) {
	return Player(value.As<Napi::String>().Utf8Value());
}

std::vector<Player> playersFromJs(Napi::Array arr) {
	std::vector<Player> ret;
	ret.reserve(arr.Length());
	for (uint32_t i = 0; i < arr.Length(); i++) {
		ret.push_back(playerFromJs(arr.Get(i)));
	}
	return ret;
}

Message messageFromJs(Napi::Value value) {
	Napi::Object message = value.As<Napi::Object>();
	return Message(
		message.Get("msg").As<Napi::String>().Utf8Value(),
		message.Get("toAll").As<Napi::Boolean>().Value()
	);
}

std::vector<Message> messagesFromJs(Napi::Array arr) {
	std::vector<Message> ret;
	ret.reserve(arr.Length());
	for (uint32_t i = 0; i < arr.Length(); i++) {
		ret.push_back(messageFromJs(arr.Get(i)));
	}
	return ret;
}

ParsedCommand parsedCommandFromJs(Napi::Object o) {
	std::vector<std::string> command;
	std::unordered_map<std::string, std::vector<std::string>> tags;

	const Napi::Array commandJs = o.Get("command").As<Napi::Array>();
	command.reserve(commandJs.Length());
	for (uint32_t i = 0; i < commandJs.Length(); i++) {
		command.push_back(commandJs.Get(i).As<Napi::String>().Utf8Value());
	}

	const Napi::Object tagsJs = o.Get("tags").As<Napi::Object>();
	const Napi::Array tagKeys = tagsJs.GetPropertyNames();
	for (uint32_t i = 0; i < tagKeys.Length(); i++) {
		const Napi::Value key = tagKeys.Get(i);
		const std::string keyStr = key.As<Napi::String>().Utf8Value();

		const Napi::Array tagArgs = tagsJs.Get(key).As<Napi::Array>();
		tags[keyStr].clear();
		tags[keyStr].reserve(tagArgs.Length());
		for (uint32_t j = 0; j < tagArgs.Length(); j++) {
			tags[keyStr].push_back(tagArgs.Get(j).As<Napi::String>().Utf8Value());
		}
	}

	return ParsedCommand{command, tags};
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
