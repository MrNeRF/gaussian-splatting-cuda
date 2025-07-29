# windows/linux platform independent
set(RAW_VCPKG_PATH $ENV{VCPKG_ROOT})
file(TO_CMAKE_PATH "${RAW_VCPKG_PATH}" VCPKG_ROOT)
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

cmake_minimum_required(VERSION 3.24...3.30)
project(gaussian_splatting_cuda LANGUAGES CUDA CXX C)

# Setup
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

option(ENABLE_CUDA_GL_INTEROP "Enable CUDA-OpenGL interoperability" ON)

# Set Torch_DIR based on platform and build type
if (WIN32)
    if(CMAKE_CONFIGURATION_TYPES)
        # https://github.com/pytorch/pytorch/issues/155667
        set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
    else()
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
        elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(Torch_DIR "${PROJ_ROOT_DIR}/external/debug/libtorch/share/cmake/Torch")
        else()
            message(FATAL_ERROR "libtorch binaries only available for Debug and Release on Windows. Current build type: '${CMAKE_BUILD_TYPE}'")
        endif()
    endif()
else()
    set(Torch_DIR "${PROJ_ROOT_DIR}/external/libtorch/share/cmake/Torch")
endif()


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Parallel build
include(ProcessorCount)
ProcessorCount(total_cores)
if(total_cores GREATER 1)
    math(EXPR used_cores "${total_cores} - 2")
    set(ENV{MAKEFLAGS} "-j${used_cores}")
    message(STATUS "Building with ${used_cores} cores")
endif()

# CUDA setup
enable_language(CUDA)
find_package(CUDAToolkit 12.8 REQUIRED)

# Create nvToolsExt target BEFORE finding Torch (PyTorch expects this)
if(NOT TARGET CUDA::nvToolsExt)
    find_library(NVTOOLSEXT_LIBRARY
            NAMES nvToolsExt
            PATHS
            ${CUDAToolkit_LIBRARY_ROOT}
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDAToolkit_TARGET_DIR}/lib64
            /usr/local/cuda-12.8/lib64
            PATH_SUFFIXES lib lib64
            NO_DEFAULT_PATH
    )

    if(NVTOOLSEXT_LIBRARY)
        add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
        set_target_properties(CUDA::nvToolsExt PROPERTIES
                IMPORTED_LOCATION ${NVTOOLSEXT_LIBRARY}
        )
        message(STATUS "Created CUDA::nvToolsExt target: ${NVTOOLSEXT_LIBRARY}")
    else()
        # Create empty interface library (nvToolsExt is deprecated in CUDA 12+)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        message(STATUS "nvToolsExt not found (normal for CUDA 12+), created empty interface target")
    endif()
endif()

message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} found at ${CUDAToolkit_TARGET_DIR}")

# Other packages
find_package(Torch REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glad REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(args REQUIRED)
find_package(spdlog REQUIRED)

# Python packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(WIN32)
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared") # when using INTEROP in windows static and dynamic symbols are mixed
endif()

# Check for CUDA-OpenGL interop capability (only if enabled)
if(ENABLE_CUDA_GL_INTEROP)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES})
    check_cxx_source_compiles("
        #ifdef _WIN32
        #include <windows.h>
        #endif
        #include <cuda_runtime.h>
        #include <cuda_gl_interop.h>
        int main() {
            cudaGraphicsResource_t resource;
            return 0;
        }
    " CUDA_GL_INTEROP_FOUND)

    if(CUDA_GL_INTEROP_FOUND)
        message(STATUS "CUDA-OpenGL interop support: ENABLED")
        set(CUDA_GL_INTEROP_ENABLED 1)
    else()
        message(WARNING "CUDA-OpenGL interop support: DISABLED (headers not found)")
        set(CUDA_GL_INTEROP_ENABLED 0)
    endif()
else()
    message(STATUS "CUDA-OpenGL interop support: DISABLED (by user option)")
    set(CUDA_GL_INTEROP_FOUND FALSE)
    set(CUDA_GL_INTEROP_ENABLED 0)
endif()

# Create a configuration header
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/config.h"
        @ONLY
)

add_subdirectory(gsplat)

# =============================================================================
# HOST LIBRARY - Compiled with g++ (fast!)
# =============================================================================

set(HOST_SOURCES
        src/application.cpp
        src/argument_parser.cpp
        src/mcmc.cpp
        src/camera.cpp
        src/image_io.cpp
        src/parameters.cpp
        src/splat_data.cpp
        src/trainer.cpp
        src/rasterizer.cpp
        src/metrics.cpp
        src/rasterizer_autograd.cpp
        src/visualizer/camera_controller.cpp
        src/visualizer/gui_manager.cpp
        src/visualizer/input_handler.cpp
        src/visualizer/rendering_pipeline.cpp
        src/visualizer/scene.cpp
        src/visualizer/scene_panel.cpp
        src/visualizer/scene_manager.cpp
        src/visualizer/viewer.cpp
        src/visualizer/training_manager.cpp
        src/visualizer/window_manager.cpp
        src/external/tinyply.cpp
        src/bilateral_grid.cpp
        src/selective_adam.cpp
        src/training_setup.cpp
        src/bounding_box.cpp
        src/visualizer/render_bounding_box.cpp)

add_library(gaussian_host STATIC ${HOST_SOURCES} $<TARGET_OBJECTS:gs_loader>)
add_subdirectory(src/loader)

target_include_directories(gaussian_host
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include  # For generated config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/gsplat  # Add gsplat headers
        ${CUDAToolkit_INCLUDE_DIRS}         # Add CUDA headers for interop
        PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
)

target_link_libraries(gaussian_host
        PUBLIC
        ${TORCH_LIBRARIES}
        TBB::tbb
        nlohmann_json::nlohmann_json
        glm::glm
        glfw
        glad::glad
        imgui::imgui
        Threads::Threads
        taywee::args
        gs_loader
        Python3::Python
        gsplat_backend  # Link to gsplat
        ${OPENGL_LIBRARIES}
        CUDA::cudart    # Add CUDA runtime for interop
        spdlog::spdlog
)

if(UNIX)
    target_link_libraries(gaussian_host PUBLIC dl)
endif()

# Fast C++ compilation with proper debug symbols
target_compile_options(gaussian_host PRIVATE
        $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer -DDEBUG>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
)

# =============================================================================
# AVX2 SUPPORT FOR HOST LIBRARY
# =============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_AVX2)
        # Add AVX2 compilation flags
        target_compile_options(gaussian_host PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
        )

        # Define the macros that the code checks for
        target_compile_definitions(gaussian_host PRIVATE
                HAS_AVX2_SUPPORT
        )

        message(STATUS "✓ AVX2 support enabled for gaussian_host")

        # Verify the flags are set
        get_target_property(HOST_COMPILE_OPTIONS gaussian_host COMPILE_OPTIONS)
        get_target_property(HOST_COMPILE_DEFINITIONS gaussian_host COMPILE_DEFINITIONS)
        message(STATUS "  Compile options: ${HOST_COMPILE_OPTIONS}")
        message(STATUS "  Compile definitions: ${HOST_COMPILE_DEFINITIONS}")
    else()
        message(WARNING "✗ Compiler does not support AVX2")
    endif()
endif()

# Ensure debug symbols in debug builds
set_target_properties(gaussian_host PROPERTIES
        DEBUG_POSTFIX d
)

add_definitions(-DPROJECT_ROOT_PATH="${PROJ_ROOT_DIR}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
endif()

# =============================================================================
# KERNEL LIBRARY
# =============================================================================

set(KERNEL_SOURCES
        kernels/ssim.cu
        kernels/bilateral_grid_forward.cu
        kernels/bilateral_grid_backward.cu
        kernels/bilateral_grid_tv.cu
)

# Add CUDA-OpenGL interop kernels if supported
if(CUDA_GL_INTEROP_FOUND)
    list(APPEND KERNEL_SOURCES kernels/cuda_gl_interop.cu)
endif()

add_library(gaussian_kernels STATIC ${KERNEL_SOURCES})

set_target_properties(gaussian_kernels PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gaussian_kernels
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include  # For generated config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        PRIVATE
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(gaussian_kernels
        PUBLIC
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        ${TORCH_LIBRARIES}
        glm::glm
        ${OPENGL_LIBRARIES}
        spdlog::spdlog
)

target_compile_options(gaussian_kernels PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 -use_fast_math --ptxas-options=-v>
)

# =============================================================================
# VISUALIZER LIBRARY
# =============================================================================

if(CUDA_GL_INTEROP_FOUND)
    set(VISUALIZER_SOURCES
            src/visualizer/cuda_gl_interop.cpp
    )

    add_library(gaussian_visualizer STATIC ${VISUALIZER_SOURCES})

    target_include_directories(gaussian_visualizer
            PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
            ${OPENGL_INCLUDE_DIRS}
    )

    target_link_libraries(gaussian_visualizer
            PUBLIC
            ${TORCH_LIBRARIES}
            ${OPENGL_LIBRARIES}
            CUDA::cudart
            glad::glad
            glfw
            glm::glm
            imgui::imgui
            spdlog::spdlog
    )
endif()

# =============================================================================
# MAIN EXECUTABLE
# =============================================================================

add_executable(${PROJECT_NAME} src/main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/gsplat
        ${Python3_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
)

set(MAIN_LINK_LIBRARIES
        gaussian_host
        gaussian_kernels
        gsplat_backend
        Python3::Python
        ${OPENGL_LIBRARIES}
        CUDA::cudart
        spdlog::spdlog
)

if(CUDA_GL_INTEROP_FOUND AND TARGET gaussian_visualizer)
    list(APPEND MAIN_LINK_LIBRARIES gaussian_visualizer)
endif()

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${MAIN_LINK_LIBRARIES}
        taywee::args
)

# Platform-specific settings
if(WIN32)
    file(GLOB TORCH_DLLS "${Torch_DIR}/../../../lib/*.dll")
    foreach(TORCH_DLL ${TORCH_DLLS})
        add_custom_command(
                TARGET ${PROJECT_NAME}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_DLL}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
    endforeach()
elseif(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE GL GLU)

    find_path(TORCH_LIB_DIR libtorch_cpu.so
            PATHS "${Torch_DIR}/../../../lib"
            NO_DEFAULT_PATH)

    if(TORCH_LIB_DIR)
        set_target_properties(${PROJECT_NAME} PROPERTIES
                INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}:${TORCH_LIB_DIR}"
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_RPATH_USE_LINK_PATH TRUE
        )
        message(STATUS "Torch library directory: ${TORCH_LIB_DIR}")
    else()
        message(WARNING "Could not find Torch library directory")
    endif()
endif()

# Build type configuration
function(configure_build_type target)
    # Check if target is an INTERFACE library
    get_target_property(target_type ${target} TYPE)

    if(target_type STREQUAL "INTERFACE_LIBRARY")
        # For INTERFACE libraries, use INTERFACE properties
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${target} INTERFACE DEBUG_BUILD)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${target} INTERFACE RELEASE_BUILD)
        endif()
    else()
        # For regular libraries, use PRIVATE properties
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${target} PRIVATE DEBUG_BUILD)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_definitions(${target} PRIVATE RELEASE_BUILD)
        endif()
    endif()
endfunction()

configure_build_type(gaussian_host)
configure_build_type(gaussian_kernels)
configure_build_type(gsplat_backend)
if(CUDA_GL_INTEROP_FOUND AND TARGET gaussian_visualizer)
    configure_build_type(gaussian_visualizer)
endif()
configure_build_type(${PROJECT_NAME})

# =============================================================================
# TESTING (Optional)
# =============================================================================

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)

    set(TEST_SOURCES
            tests/test_mcmc.cpp
            tests/test_basic.cpp
            tests/test_rasterization.cpp
            tests/test_gsplat_ops.cpp
            tests/test_intersect_debug.cpp
            tests/test_autograd.cpp
            tests/test_numerical_gradients.cpp
            tests/test_garden_data.cpp
            tests/torch_impl.cpp
    )

    add_executable(gaussian_tests ${TEST_SOURCES})

    set_target_properties(gaussian_tests PROPERTIES
            CUDA_ARCHITECTURES native
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_include_directories(gaussian_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/gsplat
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
            ${Python3_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
            ${OPENGL_INCLUDE_DIRS}
    )

    set(TEST_LINK_LIBRARIES
            gaussian_host
            gaussian_kernels
            gsplat_backend
            GTest::gtest
            GTest::gtest_main
            Python3::Python
            ${OPENGL_LIBRARIES}
            CUDA::cudart
            spdlog::spdlog
    )

    if(CUDA_GL_INTEROP_FOUND AND TARGET gaussian_visualizer)
        list(APPEND TEST_LINK_LIBRARIES gaussian_visualizer)
    endif()

    target_link_libraries(gaussian_tests PRIVATE ${TEST_LINK_LIBRARIES})

    target_compile_options(gaussian_tests PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo -Xcudafe --device-debug>
    )

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_data")
        add_custom_command(TARGET gaussian_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/test_data"
                "$<TARGET_FILE_DIR:gaussian_tests>/test_data"
        )
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/test_garden_data.pt")
        add_custom_command(TARGET gaussian_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:gaussian_tests>/tests/data"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/test_garden_data.pt"
                "$<TARGET_FILE_DIR:gaussian_tests>/tests/data/"
        )
    endif()

    include(GoogleTest)
    gtest_discover_tests(gaussian_tests)

    add_custom_target(run_tests
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            DEPENDS gaussian_tests
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    message(STATUS "Tests enabled. Build with 'make gaussian_tests' and run with 'make run_tests' or 'ctest'")
endif()

# =============================================================================
# BUILD INFO & OPTIMIZATIONS
# =============================================================================

message(STATUS "===========================================")
message(STATUS "Build Configuration:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  Torch Version: ${Torch_VERSION}")
message(STATUS "  Python Version: ${Python3_VERSION}")
message(STATUS "  OpenGL Found: ${OPENGL_FOUND}")
message(STATUS "  CUDA-GL Interop Option: ${ENABLE_CUDA_GL_INTEROP}")
message(STATUS "  CUDA-GL Interop Available: ${CUDA_GL_INTEROP_FOUND}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "")
message(STATUS "Include paths:")
message(STATUS "  - Main: ${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "  - Generated: ${CMAKE_CURRENT_BINARY_DIR}/include")
message(STATUS "  - Kernels: ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels")
message(STATUS "  - Python: ${Python3_INCLUDE_DIRS}")
message(STATUS "  - CUDA: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  - OpenGL: ${OPENGL_INCLUDE_DIRS}")
message(STATUS "===========================================")

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()
