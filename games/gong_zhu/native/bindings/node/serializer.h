#include <napi.h>

#include "gong_zhu/game_data.h"

#include "cards/serializer.h"

#pragma once


namespace cards {
namespace gong_zhu {

using cards::toJs;

Napi::Object toJs(Napi::Env env, const GameData& gameData);
GameData gameDataFromJs(Napi::Object o);

template <typename T>
Napi::Array toJs(Napi::Env env, const std::vector<T>& v) {
	Napi::Array ret = Napi::Array::New(env, v.size());
	for (uint32_t i = 0; i < v.size(); i++) {
		ret.Set(i, toJs(env, v[i]));
	}
	return ret;
}

} // namespace gong_zhu
} // namespace cards
