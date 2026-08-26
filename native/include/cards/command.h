#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>


#pragma once


namespace cards {

struct ParsedCommand {
	std::vector<std::string>                                    command;
	std::unordered_map<std::string, std::vector<std::string>>   tags;
};

[[ nodiscard ]] std::vector<std::string> lexTokens(const std::string_view s);
[[ nodiscard ]] ParsedCommand parseCommand(
	const std::string_view input,
	const std::unordered_map<std::string, size_t>& tagArgCounts = {}
);

} // namespace cards
