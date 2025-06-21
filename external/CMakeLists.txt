add_subdirectory(json)
add_subdirectory(args)
add_subdirectory(glfw)
add_library(glm INTERFACE IMPORTED GLOBAL)
target_include_directories(glm INTERFACE glm)

set(GLAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glad)
add_library(glad STATIC
        ${GLAD_DIR}/src/glad.c
)
target_include_directories(glad PUBLIC
        ${GLAD_DIR}/include
)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
add_library(imgui STATIC
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
)
# ImGui needs GLFW headers
target_link_libraries(imgui PUBLIC glfw)
