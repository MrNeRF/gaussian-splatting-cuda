set(RENDERING_SOURCES
        rendering_engine.cpp
        rendering_engine_impl.cpp
        rendering_pipeline.cpp
        point_cloud_renderer.cpp
        grid_renderer.cpp
        bbox_renderer.cpp
        axes_renderer.cpp
)

# Conditionally add CUDA GL interop
if(CUDA_GL_INTEROP_ENABLED)
    list(APPEND RENDERING_SOURCES cuda_gl_interop.cpp)
endif()

add_library(gs_rendering STATIC ${RENDERING_SOURCES})

target_link_libraries(gs_rendering
    PUBLIC
        ${TORCH_LIBRARIES}
        glm::glm
    PRIVATE
        gaussian_host
        gaussian_kernels
        gsplat_backend
        glad::glad
        ${OPENGL_LIBRARIES}
        CUDA::cudart
        spdlog::spdlog
)

target_include_directories(gs_rendering
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/gsplat
        ${CUDAToolkit_INCLUDE_DIRS}
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
        PROJECT_ROOT_PATH="${PROJECT_SOURCE_DIR}"
        $<$<BOOL:${CUDA_GL_INTEROP_ENABLED}>:CUDA_GL_INTEROP_ENABLED>
)
