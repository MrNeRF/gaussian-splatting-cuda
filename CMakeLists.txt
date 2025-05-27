cmake_minimum_required(VERSION 3.24)
project(gaussian_splatting_cuda LANGUAGES CUDA CXX)

# Setup
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
set(Torch_DIR "${PROJ_ROOT_DIR}/external/libtorch/share/cmake/Torch")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
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

# Packages
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP)
find_package(Threads REQUIRED)

# Python packages - CRITICAL for linking
find_package(PythonLibs REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(CUDAToolkit_VERSION VERSION_LESS "11.7")
    message(FATAL_ERROR "Requires CUDA 11.7+")
endif()

add_subdirectory(external)

# =============================================================================
# HOST LIBRARY - Compiled with g++ (fast!)
# =============================================================================

set(HOST_SOURCES
        src/main.cpp
        src/gaussian.cpp
        src/gaussian_init.cpp
        src/camera.cpp
        src/mean_neighbor_dist.cpp
        src/camera_utils.cpp
        src/rasterizer.cpp
        src/read_utils.cpp
        src/parameters.cpp
        src/argument_parser.cpp
)

add_library(gaussian_host STATIC ${HOST_SOURCES})

# Set up include directories for host library
target_include_directories(gaussian_host
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # This allows #include "core/header.hpp"
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyply/source  # For tinyply.h
        ${PYTHON_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(gaussian_host
        PUBLIC
        ${TORCH_LIBRARIES}
        nlohmann_json::nlohmann_json
        tinyply
        eigen3
        glm
        Threads::Threads
        args
        ${PYTHON_LIBRARIES}
        Python3::Python
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gaussian_host PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found – compiling with parallel point-cloud loader")
else()
    message(STATUS "OpenMP NOT found – falling back to single-thread loader")
endif()

# Fast C++ compilation
target_compile_options(gaussian_host PRIVATE
        $<$<CONFIG:Debug>:-O0 -g -Wall>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
)

# =============================================================================
# KERNEL LIBRARY - Compiled with nvcc (only what needs it!)
# =============================================================================

set(KERNEL_SOURCES
        kernels/ssim.cu
        kernels/rasterize_points.cu
        kernels/cuda_rasterizer/forward.cu
        kernels/cuda_rasterizer/backward.cu
        kernels/cuda_rasterizer/rasterizer_impl.cu
)

add_library(gaussian_kernels STATIC ${KERNEL_SOURCES})

set_target_properties(gaussian_kernels PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

# Set up include directories correctly for CUDA kernels
target_include_directories(gaussian_kernels
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include          # For kernels/header.cuh and core/header.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels  # So kernels can find headers directly as "config.h"
        PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(gaussian_kernels
        PUBLIC
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        ${TORCH_LIBRARIES}
        glm
)

# CUDA optimizations
target_compile_options(gaussian_kernels PRIVATE
        $<$<CONFIG:Debug>:-O0 -g -G -lineinfo>
        $<$<CONFIG:Release>:-O3 -use_fast_math --ptxas-options=-v>
)

# =============================================================================
# MAIN EXECUTABLE
# =============================================================================

add_executable(${PROJECT_NAME} src/main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES native
)

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyply/source  # For tinyply.h in main.cpp
        ${PYTHON_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        gaussian_host
        gaussian_kernels
        # Add Python libraries to main executable too
        ${PYTHON_LIBRARIES}
        Python3::Python
)

# Build type configuration
function(configure_build_type target)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${target} PRIVATE DEBUG_BUILD)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_definitions(${target} PRIVATE RELEASE_BUILD)
    endif()
endfunction()

configure_build_type(gaussian_host)
configure_build_type(gaussian_kernels)
configure_build_type(${PROJECT_NAME})

# =============================================================================
# TESTING (Optional)
# =============================================================================

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_host.cpp")
        add_executable(test_host tests/test_host.cpp)
        target_include_directories(test_host PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyply/source
        )
        target_link_libraries(test_host PRIVATE gaussian_host)
        add_test(NAME HostTests COMMAND test_host)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernels.cu")
        add_executable(test_kernels tests/test_kernels.cu)
        set_target_properties(test_kernels PROPERTIES CUDA_ARCHITECTURES native)
        target_include_directories(test_kernels PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyply/source
        )
        target_link_libraries(test_kernels PRIVATE gaussian_kernels gaussian_host)
        add_test(NAME KernelTests COMMAND test_kernels)
    endif()
endif()

# =============================================================================
# BUILD INFO & OPTIMIZATIONS
# =============================================================================

message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "Torch Version: ${Torch_VERSION}")
message(STATUS "Python Version: ${Python3_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Include paths:")
message(STATUS "  - Main: ${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "  - Kernels: ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels")
message(STATUS "  - TinyPLY: ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyply/source")
message(STATUS "  - Python: ${PYTHON_INCLUDE_DIRS}")

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()
