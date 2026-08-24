#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

if command -v clang++ &> /dev/null; then
	PRESET=clang
	WANT_CXX=clang++
else
	echo "Warning: clang++ not found." >&2
	if ! command -v g++ &> /dev/null; then
		echo "Error: neither clang++ nor g++ is installed; install a C++ compiler and retry." >&2
		exit 1
	fi
	read -r -p "Use gcc (g++) instead? [y/N] " reply
	case "$reply" in
		[Yy]*) ;;
		*) echo "Aborting."; exit 1 ;;
	esac
	PRESET=gcc
	WANT_CXX=g++
fi

if [ -f build/CMakeCache.txt ]; then
	CACHED_CXX=$(basename "$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' build/CMakeCache.txt | head -1)")
	if [ -n "$CACHED_CXX" ] && [ "$CACHED_CXX" != "$WANT_CXX" ]; then
		echo "Compiler changed ($CACHED_CXX -> $WANT_CXX); removing stale build/ ..."
		rm -rf build
	fi
fi

echo "Configuring ($PRESET)..."
cmake --preset "$PRESET"

echo "Building..."
cmake --build build -j
