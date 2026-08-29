#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cards/player.h"
#include "cards/rng.h"
#include "gong_zhu/game_data.h"

#include "serializer.h"

namespace py = pybind11;
using namespace cards;
using namespace cards::gong_zhu;

namespace {

template <typename T>
T dictGet(const py::dict& settings, const char* key, const T& fallback) {
	return settings.contains(key) ? settings[key].cast<T>() : fallback;
}

class Env {
public:
	explicit Env(const py::dict& settings) {
		this->numPlayers = dictGet<uint64_t>(settings, "num_players", this->numPlayers);
		this->numDecks = dictGet<uint64_t>(settings, "num_decks", this->numDecks);
		this->gameSettings.spectatorPolicy = dictGet<std::string>(settings, "spectator_policy", this->gameSettings.spectatorPolicy);
		this->gameSettings.losingThreshold = dictGet<int64_t>(settings, "losing_threshold", this->gameSettings.losingThreshold);
		this->gameSettings.expose3 = dictGet<bool>(settings, "expose3", this->gameSettings.expose3);
		this->gameSettings.zhuYangManJuan = dictGet<bool>(settings, "zhu_yang_man_juan", this->gameSettings.zhuYangManJuan);
		this->gameSettings.allowCustomSeed = dictGet<bool>(settings, "allow_custom_seed", this->gameSettings.allowCustomSeed);
		this->gameSettings.customSeed = dictGet<int64_t>(settings, "custom_seed", this->gameSettings.customSeed);
		for (uint64_t i = 0; i < this->numPlayers; i++) {
			this->turnOrder.emplace_back(std::to_string(i));
		}
	}

	void reset(uint32_t seed) {
		this->shuffler = std::make_unique<SeededShuffler>(seed);
		this->gameData = GameData();
		this->gameData.numDecks = static_cast<uint8_t>(this->numDecks);
		this->gameData.settings = this->gameSettings;
		this->gameData.initGame(this->turnOrder, *this->shuffler);
	}

	py::tuple applyCommand(uint64_t seat, const std::string& cmd) {
		const auto [status, messages] = this->gameData.applyCommand(static_cast<size_t>(seat), cmd, *this->shuffler, nullptr);
		return py::make_tuple(static_cast<int64_t>(status), toPy(messages));
	}

	py::object rawState(uint64_t seat) const {
		return toPy(this->gameData.obfuscateGameData(static_cast<size_t>(seat)));
	}

	py::list legalMoves(uint64_t seat) const {
		py::list cards;
		for (const Card& card : this->gameData.getLegalMoves(static_cast<size_t>(seat))) {
			cards.append(static_cast<int64_t>(card.getCardId()));
		}
		return cards;
	}

	std::string gameState() const { return to_string(this->gameData.gameState); }
	int64_t currentFrame() const { return this->gameData.currentFrame; }
	uint64_t numPlayersCount() const { return this->numPlayers; }

	int64_t currentSeat() const {
		for (size_t i = 0; i < this->gameData.needToAct.size(); i++) {
			if (this->gameData.needToAct[i]) {
				return static_cast<int64_t>(i);
			}
		}
		return -1;
	}

private:
	uint64_t numPlayers = 4;
	uint64_t numDecks = 1;
	GameSettings gameSettings;
	std::vector<Player> turnOrder;
	GameData gameData;
	std::unique_ptr<Shuffler> shuffler;
};

} // namespace

PYBIND11_MODULE(gong_zhu, m) {
	m.doc() = "Gong Zhu native rules core (pybind11)";

	py::class_<Env>(m, "Env")
		.def(py::init<const py::dict&>(), py::arg("settings") = py::dict())
		.def("reset", &Env::reset, py::arg("seed"))
		.def("apply_command", &Env::applyCommand, py::arg("seat"), py::arg("cmd"))
		.def("raw_state", &Env::rawState, py::arg("seat"))
		.def("legal_moves", &Env::legalMoves, py::arg("seat"))
		.def("game_state", &Env::gameState)
		.def("current_frame", &Env::currentFrame)
		.def("current_seat", &Env::currentSeat)
		.def("num_players", &Env::numPlayersCount);
}
