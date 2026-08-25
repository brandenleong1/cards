#include <utility>

#include "common/rng.h"


SeededShuffler::SeededShuffler(const uint32_t seed) : state(seed) {}

double SeededShuffler::next01() {
	this->state += 0x6D2B79F5u;
	uint32_t t = state;
	t = (t ^ (t >> 15)) * (t | 1u);
	t ^= t + (t ^ (t >> 7)) * (t | 61u);
	t = t ^ (t >> 14);
	return t / 4294967296.0;
}

void SeededShuffler::shuffle(std::vector<Card>& deck) {
	for (size_t i = deck.size(); i-- > 1; ) {
		size_t j = static_cast<size_t>(this->next01() * static_cast<double>(i + 1));
		std::swap(deck[i], deck[j]);
	}
}

void ScriptedShuffler::push(const std::vector<Card>& permutation) {
	this->deque.push_back(permutation);
}

void ScriptedShuffler::shuffle(std::vector<Card>& deck) {
	deck = std::move(this->deque.front());
	this->deque.pop_front();
}
