#!/bin/bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

REQUESTED="${CARDS_COMPILER:-}"
for arg in "$@"; do
	case "$arg" in
		--clang) REQUESTED=clang ;;
		--gcc) REQUESTED=gcc ;;
		*) echo "Error: unknown argument '$arg' (expected --clang or --gcc)." >&2; exit 1 ;;
	esac
done

case "$REQUESTED" in
	clang)
		if ! command -v clang++ &> /dev/null; then
			echo "Error: clang requested but clang++ is not installed." >&2
			exit 1
		fi
		PRESET=clang
		WANT_CXX=clang++
		;;
	gcc)
		if ! command -v g++ &> /dev/null; then
			echo "Error: gcc requested but g++ is not installed." >&2
			exit 1
		fi
		PRESET=gcc
		WANT_CXX=g++
		;;
	"")
		if command -v clang++ &> /dev/null; then
			PRESET=clang
			WANT_CXX=clang++
		elif command -v g++ &> /dev/null; then
			PRESET=gcc
			WANT_CXX=g++
		else
			echo "Error: neither clang++ nor g++ is installed; install a C++ compiler and retry." >&2
			exit 1
		fi
		;;
	*)
		echo "Error: CARDS_COMPILER must be 'clang' or 'gcc' (got '$REQUESTED')." >&2
		exit 1
		;;
esac

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
