#include "cards/message.h"


namespace cards {

Message::Message(std::string content, bool toAll) :
	content(std::move(content)), toAll(toAll) {}

} // namespace cards
