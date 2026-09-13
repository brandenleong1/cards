#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <napi.h>

#include "cards/message.h"
#include "cards/player.h"
#include "cards/rng.h"
#include "gong_zhu/game_data.h"

#include "cards/binding.h"
#include "cards/serializer.h"
#include "serializer.h"

using namespace cards;
using namespace cards::gong_zhu;

namespace {

Napi::Value clearGameData(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();
	if (!validateArgs(info,
		"clearGameData(gameData)",
		{ArgType::Object})
	) {
		return env.Null();
	}

	GameData gameData = gameDataFromJs(info[0].As<Napi::Object>());

	gameData.clearGameData();
	return toJs(env, gameData);
}

Napi::Value initGame(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();
	if (!validateArgs(info,
		"initGame(gameData, turnOrder, seed)",
		{ArgType::Object, ArgType::Array, ArgType::Number})
	) {
		return env.Null();
	}

	GameData gameData = gameDataFromJs(info[0].As<Napi::Object>());
	const std::vector<Player> turnOrder = playersFromJs(info[1].As<Napi::Array>());
	const uint32_t seed = info[2].As<Napi::Number>().Uint32Value();

	SeededShuffler shuffler(seed);
	gameData.initGame(turnOrder, shuffler);
	return toJs(env, gameData);
}

Napi::Value applyCommand(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();
	if (!validateArgs(info,
		"applyCommand(gameData, seat, command, seed, newTurnOrder)",
		{ArgType::Object, ArgType::Number, ArgType::Object, ArgType::Number, ArgType::NullableArray})
	) {
		return env.Null();
	}

	Napi::Object gameDataJs = info[0].As<Napi::Object>();
	GameData gameData = gameDataFromJs(gameDataJs);
	const size_t turnOrderIdx = static_cast<size_t>(info[1].As<Napi::Number>().Int64Value());
	const ParsedCommand parsedCommand = parsedCommandFromJs(info[2].As<Napi::Object>());
	const uint32_t seed = info[3].As<Napi::Number>().Uint32Value();
	const std::optional<std::vector<Player>> newTurnOrderOpt =
		(info[4].IsNull() || info[4].IsUndefined())
			? std::optional<std::vector<Player>>{}
			: std::optional<std::vector<Player>>{playersFromJs(info[4].As<Napi::Array>())};
	const std::vector<Player>* const newTurnOrder = newTurnOrderOpt.has_value() ? &(newTurnOrderOpt.value()) : nullptr;

	SeededShuffler shuffler(seed);
	const std::tuple<int8_t, std::vector<Message>> ret = gameData.applyCommand(turnOrderIdx, parsedCommand, shuffler, newTurnOrder);

	const Napi::Object newGameDataJs = toJs(env, gameData);
	const Napi::Array newGameDataJsKeys = newGameDataJs.GetPropertyNames();
	for (uint32_t i = 0; i < newGameDataJsKeys.Length(); i++) {
		const Napi::Value key = newGameDataJsKeys.Get(i);
		gameDataJs.Set(key, newGameDataJs.Get(key));
	}

	Napi::Object retJs = Napi::Object::New(env);
	retJs.Set("status", Napi::Number::New(env, std::get<0>(ret)));
	retJs.Set("events", toJs(env, std::get<1>(ret)));

	return retJs;
}

Napi::Value obfuscateGameData(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();
	if (!validateArgs(info,
		"obfuscateGameData(gameData, seat)",
		{ArgType::Object, ArgType::Number})
	) {
		return env.Null();
	}

	GameData gameData = gameDataFromJs(info[0].As<Napi::Object>());
	const size_t turnOrderIdx = static_cast<size_t>(info[1].As<Napi::Number>().Int64Value());

	return toJs(env, gameData.obfuscateGameData(turnOrderIdx));
}

Napi::Value getDefaultSettings(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();
	if (!validateArgs(info,
		"getDefaultSettings()",
		{})
	) {
		return env.Null();
	}

	return toJs(env, GameData{});
}

} // namespace

Napi::Object Init(Napi::Env env, Napi::Object exports) {
	exports.Set("clearGameData", Napi::Function::New(env, clearGameData));
	exports.Set("initGame", Napi::Function::New(env, initGame));
	exports.Set("applyCommand", Napi::Function::New(env, applyCommand));
	exports.Set("obfuscateGameData", Napi::Function::New(env, obfuscateGameData));
	exports.Set("getDefaultSettings", Napi::Function::New(env, getDefaultSettings));
	registerSharedBindings(env, exports);
	return exports;
}

NODE_API_MODULE(gong_zhu, Init)
