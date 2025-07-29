add_library(gs_visualizer OBJECT
        # Public API implementation
        visualizer.cpp
        visualizer_impl.cpp

        # Window management
        window/window_manager.cpp

        # Rendering subsystem
        rendering/rendering_pipeline.cpp
        rendering/render_bounding_box.cpp

        # GUI subsystem
        gui/gui_manager.cpp
        gui/scene_panel.cpp

        # Scene management
        scene/scene.cpp
        scene/scene_manager.cpp

        # Input handling
        input/input_handler.cpp
        input/camera_controller.cpp

        # Training integration
        training/training_manager.cpp
)


# CUDA-OpenGL interop support
if(CUDA_GL_INTEROP_ENABLED)
    target_sources(gs_visualizer PRIVATE
        rendering/cuda_gl_interop.cpp
    )
endif()

# Set include directories
target_include_directories(gs_visualizer
    PUBLIC
        # ONLY expose the include directory for public headers
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        # Internal paths for implementation only
        ${CMAKE_CURRENT_SOURCE_DIR}                      # For internal includes
        ${CMAKE_SOURCE_DIR}/include                       # For core headers
        ${CMAKE_SOURCE_DIR}/src                          # For "loader/loader.hpp"
        ${CMAKE_SOURCE_DIR}/gsplat                       # For gsplat headers
        ${CUDAToolkit_INCLUDE_DIRS}                      # CUDA headers
        ${OPENGL_INCLUDE_DIRS}                           # OpenGL headers
        ${Python3_INCLUDE_DIRS}                          # Python headers
)

# Link dependencies - ALL PRIVATE to hide from consumers
target_link_libraries(gs_visualizer
    PRIVATE
        ${TORCH_LIBRARIES}
        glad::glad
        glfw
        imgui::imgui
        glm::glm
        TBB::tbb
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        Python3::Python
        CUDA::cudart
        ${OPENGL_LIBRARIES}
        gs_loader           # Use the loader module
        gaussian_host       # For access to core functionality
        gaussian_kernels    # For CUDA kernels if needed
)

# Set C++ standard
target_compile_features(gs_visualizer PUBLIC cxx_std_23)

# Compiler options
target_compile_options(gs_visualizer PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang>>:-O0 -g>
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3>
)

# Handle resources (shaders and assets)
set(VISUALIZER_BUILD_RESOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/resources")
set(VISUALIZER_SOURCE_RESOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")

# Create resource directories in build folder
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/shaders")
file(MAKE_DIRECTORY "${VISUALIZER_BUILD_RESOURCE_DIR}/assets")

# Copy shaders to build directory
file(GLOB SHADER_FILES "${VISUALIZER_SOURCE_RESOURCE_DIR}/shaders/*")
foreach(SHADER_FILE ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
    configure_file(${SHADER_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/shaders/${SHADER_NAME}" COPYONLY)
endforeach()

# Copy assets to build directory
file(GLOB ASSET_FILES "${VISUALIZER_SOURCE_RESOURCE_DIR}/assets/*")
foreach(ASSET_FILE ${ASSET_FILES})
    get_filename_component(ASSET_NAME ${ASSET_FILE} NAME)
    configure_file(${ASSET_FILE} "${VISUALIZER_BUILD_RESOURCE_DIR}/assets/${ASSET_NAME}" COPYONLY)
endforeach()

# Compile definitions
target_compile_definitions(gs_visualizer PRIVATE
    # CUDA interop
    $<$<BOOL:${CUDA_GL_INTEROP_ENABLED}>:CUDA_GL_INTEROP_ENABLED>
    
    # Project root for fallback paths
    PROJECT_ROOT_PATH="${PROJECT_SOURCE_DIR}"
    
    # Resource paths
    VISUALIZER_SHADER_PATH="${VISUALIZER_BUILD_RESOURCE_DIR}/shaders"
    VISUALIZER_ASSET_PATH="${VISUALIZER_BUILD_RESOURCE_DIR}/assets"
    VISUALIZER_SOURCE_SHADER_PATH="${VISUALIZER_SOURCE_RESOURCE_DIR}/shaders"
    VISUALIZER_SOURCE_ASSET_PATH="${VISUALIZER_SOURCE_RESOURCE_DIR}/assets"
    
    # Build configuration
    $<$<CONFIG:Debug>:DEBUG_BUILD>
    $<$<CONFIG:Release>:RELEASE_BUILD>
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(gs_visualizer PRIVATE
        NOMINMAX
        _USE_MATH_DEFINES
    )
elseif(UNIX AND NOT APPLE)
    target_link_libraries(gs_visualizer PRIVATE
        GL
        GLU
        X11
    )
endif()

# Export only the public header
install(FILES ${CMAKE_SOURCE_DIR}/include/visualizer/visualizer.hpp
    DESTINATION include/visualizer
)

# Optionally install resources
install(DIRECTORY ${VISUALIZER_SOURCE_RESOURCE_DIR}/
    DESTINATION share/gaussian_splatting/visualizer/resources
    FILES_MATCHING
    PATTERN "*.vert"
    PATTERN "*.frag"
    PATTERN "*.ttf"
)

# Add custom target for resource files (makes them show up in IDEs)
file(GLOB_RECURSE VISUALIZER_RESOURCE_FILES
    "${VISUALIZER_SOURCE_RESOURCE_DIR}/shaders/*.vert"
    "${VISUALIZER_SOURCE_RESOURCE_DIR}/shaders/*.frag"
    "${VISUALIZER_SOURCE_RESOURCE_DIR}/assets/*.ttf"
)
if(VISUALIZER_RESOURCE_FILES)
    add_custom_target(visualizer_resources SOURCES ${VISUALIZER_RESOURCE_FILES})
endif()
