cards_add_library(gong_zhu
	src/game_data.cpp
	src/game_state.cpp
)
target_include_directories(gong_zhu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CARDS_BUILD_PYTHON)
	add_subdirectory(bindings/python)
endif()
