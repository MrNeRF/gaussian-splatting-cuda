# Geometry module
set(GEOMETRY_SOURCES
    bounding_box.cpp
    euclidean_transform.cpp
)

add_library(gs_geometry STATIC ${GEOMETRY_SOURCES})

target_include_directories(gs_geometry
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(gs_geometry
    PUBLIC
        gs_core  # Depends on core
        glm::glm
)

set_target_properties(gs_geometry PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    DEBUG_POSTFIX d
)
