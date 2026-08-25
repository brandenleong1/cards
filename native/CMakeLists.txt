add_library(cards STATIC
	src/cards.cpp
	src/message.cpp
	src/player.cpp
	src/rng.cpp
	src/utils.cpp
)
target_include_directories(cards PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(cards PUBLIC cxx_std_17)
target_link_libraries(cards PRIVATE cards_warnings)

find_package(TBB QUIET)
if(TBB_FOUND)
	target_link_libraries(cards PUBLIC TBB::tbb)
endif()
