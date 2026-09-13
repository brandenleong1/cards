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

add_library(gong_zhu_node SHARED ${CMAKE_JS_SRC} binding.cpp serializer.cpp)
set_target_properties(gong_zhu_node PROPERTIES
	PREFIX ""
	SUFFIX ".node"
	OUTPUT_NAME "gong_zhu"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_include_directories(gong_zhu_node PRIVATE ${CMAKE_JS_INC} ${NODE_ADDON_API_DIR})
target_link_libraries(gong_zhu_node PRIVATE gong_zhu ${CMAKE_JS_LIB})
target_compile_definitions(gong_zhu_node PRIVATE NAPI_VERSION=8)
