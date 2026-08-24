cmake_minimum_required(VERSION 3.16)
project(cards_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CARDS_WERROR "Treat warnings as errors" ON)

set(cards_warn_gnu
	-Wall -Wextra -Wpedantic
	-Wshadow=local -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
	-Woverloaded-virtual -Wnull-dereference -Wdouble-promotion
	-Wformat=2 -Wimplicit-fallthrough -Wmisleading-indentation
	-Wconversion -Wsign-conversion
	-Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast
)
set(cards_warn_clang
	-Wall -Wextra -Wpedantic
	-Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
	-Woverloaded-virtual -Wnull-dereference -Wdouble-promotion
	-Wformat=2 -Wimplicit-fallthrough
	-Wconversion -Wsign-conversion
)
set(cards_warn_msvc /W4 /permissive-)
if(CARDS_WERROR)
	list(APPEND cards_warn_gnu -Werror)
	list(APPEND cards_warn_clang -Werror)
	list(APPEND cards_warn_msvc /WX)
endif()

add_library(cards_warnings INTERFACE)
target_compile_options(cards_warnings INTERFACE
	$<$<CXX_COMPILER_ID:GNU>:${cards_warn_gnu}>
	$<$<CXX_COMPILER_ID:Clang,AppleClang>:${cards_warn_clang}>
	$<$<CXX_COMPILER_ID:MSVC>:${cards_warn_msvc}>
)

function(cards_add_library name)
	add_library(${name} STATIC ${ARGN})
	target_link_libraries(${name} PUBLIC cards)
	target_link_libraries(${name} PRIVATE cards_warnings)
	target_compile_features(${name} PUBLIC cxx_std_17)
endfunction()

add_subdirectory(native)
add_subdirectory(games/gong_zhu/native)
