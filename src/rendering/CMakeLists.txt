# Add the internal CUDA subdirectory first
add_subdirectory(cuda)

set(RENDERING_SOURCES
        rendering_engine.cpp
        rendering_engine_impl.cpp
        rendering_pipeline.cpp
        gs_rasterizer.cpp
        point_cloud_renderer.cpp
        translation_gizmo.cpp
        grid_renderer.cpp
        bbox_renderer.cpp
        axes_renderer.cpp
        text_renderer.cpp
        viewport_gizmo.cpp
        framebuffer_factory.cpp
        shader_manager.cpp
        screen_renderer.cpp
)

# Conditionally add CUDA GL interop
if(CUDA_GL_INTEROP_ENABLED)
    list(APPEND RENDERING_SOURCES cuda_gl_interop.cpp)
endif()

add_library(gs_rendering STATIC ${RENDERING_SOURCES})

target_link_libraries(gs_rendering
        PRIVATE  # ALL dependencies should be PRIVATE
        ${TORCH_LIBRARIES}
        glm::glm
        gaussian_host
        gaussian_kernels
        rendering_cuda_backend
        glad::glad
        ${OPENGL_LIBRARIES}
        CUDA::cudart
        spdlog::spdlog
        Freetype::Freetype
)

target_include_directories(gs_rendering
        PUBLIC
        # ONLY expose the include directory for public headers
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        # Internal paths for implementation only
        ${CMAKE_CURRENT_SOURCE_DIR}                           # For internal includes within rendering
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/rasterization/include # For CUDA rasterization headers
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/utils                # For CUDA utils
        ${CMAKE_SOURCE_DIR}/include                           # For public headers
        ${CMAKE_SOURCE_DIR}/src                               # For other internal modules
        ${CUDAToolkit_INCLUDE_DIRS}                           # CUDA headers
        ${OPENGL_INCLUDE_DIRS}                                # OpenGL headers
        ${Python3_INCLUDE_DIRS}                               # Python headers
)

# Handle shaders
set(RENDERING_BUILD_RESOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/resources")
set(RENDERING_SOURCE_RESOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")

file(MAKE_DIRECTORY "${RENDERING_BUILD_RESOURCE_DIR}/shaders")

file(GLOB SHADER_FILES "${RENDERING_SOURCE_RESOURCE_DIR}/shaders/*")
foreach(SHADER_FILE ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
    configure_file(${SHADER_FILE} "${RENDERING_BUILD_RESOURCE_DIR}/shaders/${SHADER_NAME}" COPYONLY)
endforeach()

target_compile_definitions(gs_rendering PRIVATE
        SHADER_PATH="${RENDERING_BUILD_RESOURCE_DIR}/shaders"
        RENDERING_SOURCE_SHADER_PATH="${RENDERING_SOURCE_RESOURCE_DIR}/shaders"
        PROJECT_ROOT_PATH="${PROJECT_SOURCE_DIR}"
        $<$<BOOL:${CUDA_GL_INTEROP_ENABLED}>:CUDA_GL_INTEROP_ENABLED>
)

# Export only the public header
install(FILES ${CMAKE_SOURCE_DIR}/include/rendering/rendering.hpp
        DESTINATION include/rendering
)
