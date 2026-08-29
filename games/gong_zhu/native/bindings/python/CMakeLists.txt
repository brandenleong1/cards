pybind11_add_module(gong_zhu_native bindings.cpp serializer.cpp)
set_target_properties(gong_zhu_native PROPERTIES OUTPUT_NAME gong_zhu)
target_link_libraries(gong_zhu_native PRIVATE gong_zhu)

install(TARGETS gong_zhu_native LIBRARY DESTINATION cards)
