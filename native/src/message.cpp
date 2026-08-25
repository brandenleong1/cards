#include "common/message.h"


Message::Message(std::string content, bool toAll) :
	content(std::move(content)), toAll(toAll) {}
