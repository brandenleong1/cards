#include <string>

#pragma once


struct Message {
	std::string     content;
	bool            toAll;

	Message(std::string content = "", bool toAll = true);
};
