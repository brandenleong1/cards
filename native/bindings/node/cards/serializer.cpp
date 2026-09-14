#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cards/serializer.h"


namespace cards {

Napi::Array toJs(Napi::Env env, const std::vector<bool>& v) {
	Napi::Array ret = Napi::Array::New(env, v.size());
	for (uint32_t i = 0; i < v.size(); i++) {
		ret.Set(i, Napi::Number::New(env, v[i] ? 1 : 0));
	}
	return ret;
}

Napi::Array toJs(Napi::Env env, const std::tuple<int64_t, int64_t>& pair) {
	Napi::Array ret = Napi::Array::New(env, 2);
	ret.Set(uint32_t(0), Napi::Number::New(env, static_cast<double>(std::get<0>(pair))));
	ret.Set(uint32_t(1), Napi::Number::New(env, static_cast<double>(std::get<1>(pair))));
	return ret;
}

Napi::Array toJs(Napi::Env env, const std::tuple<Card, uint8_t>& pair) {
	Napi::Array ret = Napi::Array::New(env, 2);
	ret.Set(uint32_t(0), Napi::Number::New(env, std::get<0>(pair).getCardId()));
	ret.Set(uint32_t(1), Napi::Number::New(env, std::get<1>(pair)));
	return ret;
}

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

	return ParsedCommand{std::move(command), std::move(tags)};
}

} // namespace cards
