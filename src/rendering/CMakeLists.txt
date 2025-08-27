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
        PUBLIC
        gs_core                # Link core module
        gs_geometry            # Link geometry module for BoundingBox and EuclideanTransform
        gs_kernels             # Link kernels module
        PRIVATE
        rendering_cuda_backend
        ${TORCH_LIBRARIES}
        glm::glm
        glad::glad
        glfw
        ${OPENGL_LIBRARIES}
        CUDA::cudart
        spdlog::spdlog
        Freetype::Freetype
)

# Since we're linking to gs_core PUBLIC, we inherit its include directories
# which include: ${CMAKE_BINARY_DIR}/include (for config.h)
#                ${CMAKE_SOURCE_DIR}/gsplat (for Common.h)
target_include_directories(gs_rendering
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/rasterization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/utils
        ${CMAKE_SOURCE_DIR}/src
        ${OPENGL_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
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
        $<$<BOOL:${CUDA_GL_INTEROP_ENABLED}>:CUDA_GL_INTEROP_ENABLED>
)

# Export only the public header
install(FILES ${CMAKE_SOURCE_DIR}/include/rendering/rendering.hpp
        DESTINATION include/rendering
)
