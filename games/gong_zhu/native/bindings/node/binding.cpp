#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <napi.h>

#include "cards/cards.h"
#include "cards/command.h"
#include "cards/player.h"
#include "cards/rng.h"
#include "gong_zhu/game_data.h"
#include "gong_zhu/game_state.h"

#include "serializer.h"

using namespace cards;
using namespace cards::gong_zhu;

namespace {

enum class ArgType { Object, Array, Number, String, NullableArray };

bool checkArgType(const Napi::Value& value, ArgType type) {
	switch (type) {
		case ArgType::Object:			return value.IsObject() && !value.IsArray();
		case ArgType::Array:			return value.IsArray();
		case ArgType::Number:			return value.IsNumber();
		case ArgType::String:			return value.IsString();
		case ArgType::NullableArray:	return value.IsArray() || value.IsNull() || value.IsUndefined();
	}
	return false;
}

bool validateArgs(const Napi::CallbackInfo& info, const char* signature, std::initializer_list<ArgType> types) {
	Napi::Env env = info.Env();

	if (info.Length() < types.size()) {
		Napi::TypeError::New(env,
			std::string(signature) + ": expected " + std::to_string(types.size()) +
			" argument(s), got " + std::to_string(info.Length())
		).ThrowAsJavaScriptException();
		return false;
	}

	size_t i = 0;
	for (const ArgType type : types) {
		if (!checkArgType(info[i], type)) {
			Napi::TypeError::New(env,
				std::string(signature) + ": argument " + std::to_string(i) + " has the wrong type"
			).ThrowAsJavaScriptException();
			return false;
		}
		i++;
	}

	return true;
}

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

Napi::Value parseCommandJs(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();
	if (!validateArgs(info,
		"parseCommand(input, tagArgCounts?)",
		{ArgType::String})
	) {
		return env.Null();
	}

	const std::string input = info[0].As<Napi::String>().Utf8Value();

	std::unordered_map<std::string, size_t> tagArgCounts;
	if (info.Length() > 1 && info[1].IsObject() && !info[1].IsArray()) {
		const Napi::Object tagArgCountsJs = info[1].As<Napi::Object>();
		const Napi::Array tagArgCountsJsKeys = tagArgCountsJs.GetPropertyNames();
		for (uint32_t i = 0; i < tagArgCountsJsKeys.Length(); i++) {
			const Napi::Value key = tagArgCountsJsKeys.Get(i);
			const std::string tag = key.As<Napi::String>().Utf8Value();
			tagArgCounts[tag] = static_cast<size_t>(tagArgCountsJs.Get(key).As<Napi::Number>().Int64Value());
		}
	}

	ParsedCommand parsedCommand = parseCommand(input, tagArgCounts);
	return toJs(env, parsedCommand);
}

} // namespace

Napi::Object Init(Napi::Env env, Napi::Object exports) {
	exports.Set("clearGameData", Napi::Function::New(env, clearGameData));
	exports.Set("initGame", Napi::Function::New(env, initGame));
	exports.Set("applyCommand", Napi::Function::New(env, applyCommand));
	exports.Set("obfuscateGameData", Napi::Function::New(env, obfuscateGameData));
	exports.Set("getDefaultSettings", Napi::Function::New(env, getDefaultSettings));
	exports.Set("parseCommand", Napi::Function::New(env, parseCommandJs));
	return exports;
}

NODE_API_MODULE(gong_zhu, Init)
