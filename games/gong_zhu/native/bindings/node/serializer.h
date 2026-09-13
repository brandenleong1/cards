#include <cstdint>
#include <vector>

#include <napi.h>

#include "cards/cards.h"
#include "cards/command.h"
#include "cards/message.h"
#include "cards/player.h"
#include "gong_zhu/game_data.h"

#pragma once


namespace cards {
namespace gong_zhu {

Napi::Value toJs(Napi::Env env, const Card& card);
Napi::String toJs(Napi::Env env, const Player& player);
Napi::Object toJs(Napi::Env env, const Message& message);
Napi::Object toJs(Napi::Env env, const ParsedCommand& parsedCommand);
Napi::Object toJs(Napi::Env env, const GameData& gameData);

template <typename T>
Napi::Array toJs(Napi::Env env, const std::vector<T>& v) {
	Napi::Array ret = Napi::Array::New(env, v.size());
	for (uint32_t i = 0; i < v.size(); i++) {
		ret.Set(i, toJs(env, v[i]));
	}
	return ret;
}

Card cardFromJs(Napi::Value value);
std::vector<Card> cardsFromJs(Napi::Array arr);
Player playerFromJs(Napi::Value value);
std::vector<Player> playersFromJs(Napi::Array arr);
Message messageFromJs(Napi::Value value);
std::vector<Message> messagesFromJs(Napi::Array arr);
ParsedCommand parsedCommandFromJs(Napi::Object o);
GameData gameDataFromJs(Napi::Object o);

} // namespace gong_zhu
} // namespace cards
