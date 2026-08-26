#include <string>

#pragma once


namespace cards {

class Player {
private:
	std::string name;

public:
	Player(const std::string& name);

	inline std::string getName() const {
		return this->name;
	}
};

} // namespace cards
