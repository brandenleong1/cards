add_library(gong_zhu_node SHARED ${CMAKE_JS_SRC} binding.cpp serializer.cpp)
set_target_properties(gong_zhu_node PROPERTIES
	PREFIX ""
	SUFFIX ".node"
	OUTPUT_NAME "gong_zhu"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)
target_link_libraries(gong_zhu_node PRIVATE gong_zhu cards_node_bindings ${CMAKE_JS_LIB})
