#include <cstddef>
#include <string>
#include <unordered_map>

#include "cards/command.h"

#include "cards/binding.h"
#include "cards/serializer.h"


namespace cards {

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

namespace {

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

void registerSharedBindings(Napi::Env env, Napi::Object& exports) {
	exports.Set("parseCommand", Napi::Function::New(env, parseCommandJs));
}

} // namespace cards
