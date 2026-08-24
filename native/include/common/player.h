#include <string>

#pragma once


class Player {
private:
	std::string name;

public:
	Player(const std::string& name);

	inline std::string getName() const {
		return this->name;
	}
};
