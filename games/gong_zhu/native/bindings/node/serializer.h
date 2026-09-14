#include <napi.h>

#include "gong_zhu/game_data.h"

#include "cards/serializer.h"

#pragma once


namespace cards {
namespace gong_zhu {

using cards::toJs;

Napi::Object toJs(Napi::Env env, const GameData& gameData);
GameData gameDataFromJs(Napi::Object o);

} // namespace gong_zhu
} // namespace cards
