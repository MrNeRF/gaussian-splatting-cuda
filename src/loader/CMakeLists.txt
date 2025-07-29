add_library(gs_loader OBJECT
        # Implementation file only (header is in include/loader/)
        loader.cpp

        # Internal implementation files
        # Core interfaces and types
        loader_interface.hpp
        loader_registry.hpp
        loader_service.hpp
        loader_service.cpp

        # Format implementations
        formats/ply.hpp
        formats/ply.cpp
        formats/colmap.hpp
        formats/colmap.cpp
        formats/transforms.hpp
        formats/transforms.cpp

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
        # ONLY expose the include directory for public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        # Internal paths for implementation only
        ${CMAKE_CURRENT_SOURCE_DIR}  # For internal includes like "loader_service.hpp"
        ${CMAKE_CURRENT_SOURCE_DIR}/..  # For includes like "loader/..."
        ${CMAKE_CURRENT_SOURCE_DIR}/../..  # For "core/..." includes
        ${CMAKE_CURRENT_SOURCE_DIR}/../../gsplat  # For Common.h
)

# Link dependencies
target_link_libraries(gs_loader
        PUBLIC
        ${TORCH_LIBRARIES}  # Only because LoadResult uses torch::Tensor
        PRIVATE
        TBB::tbb
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

# Set C++ standard
target_compile_features(gs_loader PUBLIC cxx_std_23)

# Compiler options
target_compile_options(gs_loader PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# AVX2 support for ply.cpp
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(gs_loader PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )
        target_compile_definitions(gs_loader PRIVATE HAS_AVX2_SUPPORT)
    endif()
endif()

# Export only the public header
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../include/loader/loader.hpp
        DESTINATION include/loader)