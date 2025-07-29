add_library(gs_loader OBJECT
        # Core interfaces and types
        loader_interface.hpp
        loader_types.hpp
        loader_registry.hpp
        loader_service.hpp

        # Service implementation
        loader_service.cpp

        # Concrete loader implementations
        loaders/ply_loader.hpp
        loaders/ply_loader.cpp
        loaders/colmap_loader.hpp
        loaders/colmap_loader.cpp
        loaders/blender_loader.hpp
        loaders/blender_loader.cpp
)

# Set include directories
target_include_directories(gs_loader
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..  # For "loader/..." includes
        ${CMAKE_CURRENT_SOURCE_DIR}/../..  # For "core/..." includes
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include  # For headers in include/core/
        ${CMAKE_CURRENT_SOURCE_DIR}/../../gsplat  # For Common.h
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies - now we need torch for the actual loading
target_link_libraries(gs_loader
        PUBLIC
        ${TORCH_LIBRARIES}
        spdlog::spdlog
)

# Set C++ standard
target_compile_features(gs_loader PUBLIC cxx_std_23)

# Compiler options
target_compile_options(gs_loader PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
