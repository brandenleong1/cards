add_library(cards_py_bindings STATIC cards/serializer.cpp)
target_include_directories(cards_py_bindings PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cards_py_bindings PUBLIC cards pybind11::module)
