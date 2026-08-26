#include <optional>

#include "cards/command.h"


namespace cards {

[[ nodiscard ]] std::vector<std::string> lexTokens(const std::string_view s) {
	std::vector<std::string> tokens;

	const size_t n = s.size();
	for (size_t i = 0; i < n; ) {
		while (i < n && std::isspace(s[i])) {
			i++;
		}

		if (i >= n) {
			break;
		}

		if (s[i] == '"') {
			const size_t close = s.find('"', i + 1);
			if (close != std::string_view::npos && close > i + 1) {
				tokens.emplace_back(s.substr(i, close - i + 1));
				i = close + 1;
				continue;
			}
		}

		const size_t start = i;
		while (i < n && !std::isspace(s[i])) {
			i++;
		}

		tokens.emplace_back(s.substr(start, i - start));
	}

	return tokens;
}

[[ nodiscard ]] ParsedCommand parseCommand(
	const std::string_view input,
	const std::unordered_map<std::string, size_t>& tagArgCounts
) {
	ParsedCommand parsedCommand;

	const auto stripQuotes = [](const std::string& s) -> std::string {
		return (s.size() >= 2 && s.front() == '"' && s.back() == '"') ? s.substr(1, s.size() - 2) : s;
	};

	const auto getExpectedTagArgCount = [&tagArgCounts](const std::string& s) -> size_t {
		const auto it = tagArgCounts.find(s);
		return it == tagArgCounts.end() ? 0 : it->second;
	};

	std::optional<std::string> tag;
	std::vector<std::string> tagArgs;

	for (const std::string& s : lexTokens(input)) {
		if (!tag.has_value()) {
			if (!s.empty() && s.front() == '-') {
				tag = s;
			} else {
				parsedCommand.command.push_back(stripQuotes(s));
			}
		} else {
			tagArgs.push_back(stripQuotes(s));
		}

		if (tag.has_value() && tagArgs.size() == getExpectedTagArgCount(tag.value())) {
			parsedCommand.tags[tag.value()] = std::move(tagArgs);
			tag.reset();
			tagArgs.clear();
		}
	}

	if (tag.has_value()) {
		const size_t expectedTagArgCount = getExpectedTagArgCount(tag.value());
		while (tagArgs.size() < expectedTagArgCount) {
			tagArgs.push_back("");
		}
		parsedCommand.tags[tag.value()] = std::move(tagArgs);
	}

	return parsedCommand;
}

} // namespace cards
