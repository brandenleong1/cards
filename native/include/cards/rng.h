#include <deque>
#include <vector>

#include "cards/cards.h"

#pragma once


namespace cards {

class Shuffler {
public:
	virtual ~Shuffler() = default;
	virtual void shuffle(std::vector<Card>& deck) = 0;
};

class SeededShuffler : public Shuffler {
private:
	uint32_t state;

public:
	explicit SeededShuffler(const uint32_t seed);

	double next01();
	void shuffle(std::vector<Card>& deck) override;
};

class ScriptedShuffler : public Shuffler {
private:
	std::deque<std::vector<Card>> deque;

public:
	void push(const std::vector<Card>& permutation);
	void shuffle(std::vector<Card>& deck) override;
};

} // namespace cards
