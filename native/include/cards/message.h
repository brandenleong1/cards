#include <string>

#pragma once


namespace cards {

struct Message {
	std::string     content;
	bool            toAll;

	Message(std::string content = "", bool toAll = true);
};

} // namespace cards
