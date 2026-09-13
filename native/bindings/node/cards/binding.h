#include <initializer_list>

#include <napi.h>

#pragma once


namespace cards {

enum class ArgType { Object, Array, Number, String, NullableArray };

bool checkArgType(const Napi::Value& value, ArgType type);
bool validateArgs(const Napi::CallbackInfo& info, const char* signature, std::initializer_list<ArgType> types);

void registerSharedBindings(Napi::Env env, Napi::Object& exports);

} // namespace cards
