execute_process(
	COMMAND node -p "require('node-addon-api').include"
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE NODE_ADDON_API_DIR
)
string(REGEX REPLACE "[\r\n\"]" "" NODE_ADDON_API_DIR "${NODE_ADDON_API_DIR}")
if(NOT NODE_ADDON_API_DIR)
	message(FATAL_ERROR "node-addon-api not found — run `npm install` in the project root first")
endif()
message(STATUS "node-addon-api include dir: ${NODE_ADDON_API_DIR}")

add_library(cards_node_bindings STATIC cards/serializer.cpp cards/binding.cpp)
target_include_directories(cards_node_bindings PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_JS_INC} ${NODE_ADDON_API_DIR})
target_link_libraries(cards_node_bindings PUBLIC cards)
target_compile_definitions(cards_node_bindings PUBLIC NAPI_VERSION=8)
