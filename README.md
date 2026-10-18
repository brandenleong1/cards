# Cards

Play multiplayer card games online with your friends, or train an AI model to play against.

## Overview

This repository contains a Node.js server which clients can connect to via WebSocket. Currently, we support the following games:

 - Gong Zhu (拱猪)

Each game comes with a script in which you can train a model that learns to play the game.

## Getting Started

To clone the repository, run `git clone https://github.com/brandenleong1/cards.git --recurse-submodules`.

### Prerequisites

| Dependency    | Version                      | Purpose                      |
| ------------- | ---------------------------- | ---------------------------- |
| Node.js       | ≥22                          | Server Hosting               |
| Python        | ≥3.10                        | RL Agent                     |
| CMake         | ≥3.16                        | Native Core (Server + Agent) |
| clang / gcc   | C++17; Clang ≥ 14 / GCC ≥ 11 | Build Native Core            |

GPU support is availble for training/running models.

### Build

To automatically build and configure the C++ native core, Python environment, and Node.js bindings, run `./build.sh`.

To specify a C++ compiler of choice, use `--clang` or `--gcc`, or set `CARDS_COMPILER` environment variable.

#### Manual Build

For only building the server, run `npm install && npm run build:native`.

For only building the agent, run `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install --no-build-isolation -e .`.

## Starting the Server

Run either `npm start` or equivalently `node server.js`, which opens the server on port 8080.

To specify a port of choice, set the `PORT` environment variable.

## Playing a Game

Select the game of choice. Each game contains its own lobbies sorted by newest first. Select a username, create or join an existing lobby, then wait for the host to start the game.

Players interact with the game via the in-game console. Typing `HELP` lists all available commands. For rules of each game, click the help icon in the bottom right of the screen.

## Running the Agent

Run the `player.py` file in `games/*/` to activate one of three modes:

 - Training (`--train --model-dir <model-dir>`) - Start model training from scratch or from an existing checkpoint
 - Evaluation (`--eval --model-dir <model-dir>`) - Compare models against each other using an Elo ranking system
 - Inference (`--infer --model <model> --lobby <lobby> --host <host>`) - Join a human lobby to play against

Pretrained models are available at https://huggingface.co/EffOfEcks/cards, which mirror the directory layout in this repository.

Note: `./build.sh` creates but does not activate `.venv`.

## Architecture

The client (`games/*/script.js`) communicates with the server (`server.js`) via WebSocket.

Each game's `Game` subclass manages its lobbies and players. Both the JS `Game` subclass and the Python bots share the game implementation in the C++ native core.

## RL Agent

We choose a PPO self-play approach to train the model.

Each batch until the first checkpoint trains on seats identically filled with the current model and training episodes are aggregated from all agents. After the first checkpoint is saved, prior checkpoints are used to determine how the other seats are filled; only one agent per table is the non-frozen model.

## License

MIT License -- see `LICENSE`

Copyright (c) 2026 Branden Leong
