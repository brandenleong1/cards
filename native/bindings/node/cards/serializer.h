#include <cstdint>
#include <vector>

#include <napi.h>

#include "cards/cards.h"
#include "cards/command.h"
#include "cards/message.h"
#include "cards/player.h"

#pragma once


namespace cards {

Napi::Value toJs(Napi::Env env, const Card& card);
Napi::String toJs(Napi::Env env, const Player& player);
Napi::Object toJs(Napi::Env env, const Message& message);
Napi::Object toJs(Napi::Env env, const ParsedCommand& parsedCommand);

Card cardFromJs(Napi::Value value);
std::vector<Card> cardsFromJs(Napi::Array arr);
Player playerFromJs(Napi::Value value);
std::vector<Player> playersFromJs(Napi::Array arr);
Message messageFromJs(Napi::Value value);
std::vector<Message> messagesFromJs(Napi::Array arr);
ParsedCommand parsedCommandFromJs(Napi::Object o);

} // namespace cards
