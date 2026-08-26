#include <cassert>

#include "cards/player.h"


namespace cards {

Player::Player(const std::string& name) : name(name) {
	assert(this->name.size() > 0 && "[Player::Player] name must be non-empty");
}

} // namespace cards
