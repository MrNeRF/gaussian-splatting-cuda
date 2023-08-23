cmake_minimum_required(VERSION 3.24)
project(gaussian_splatting_cuda LANGUAGES CUDA CXX)

# Determine the project root directory
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# Define the absolute path to libtorch
set(LIBTORCH_DIR "${PROJ_ROOT_DIR}/external/libtorch")

# Set CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_89,code=compute_89")

# --------------------------------------
# Core count calculation
# --------------------------------------

include(ProcessorCount)
ProcessorCount(total_cores)
if (NOT DEFINED total_cores OR total_cores EQUAL 0)
    set(total_cores 1)
    set(used_cores 1)
elseif (total_cores GREATER 1)
    math(EXPR used_cores "${total_cores} - 2") # use total_cores-2 if total_cores > 1
endif ()
set(ENV{MAKEFLAGS} "-j${used_cores}")
message(STATUS "Building with ${used_cores} out of ${total_cores} available cores")

# --------------------------------------
# Files
# --------------------------------------

set(HEADERS
        includes/read_utils.cuh
        includes/gaussian.cuh
        includes/camera.cuh
        includes/loss_monitor.cuh
        includes/image.cuh
        includes/stb_image.h
        includes/point_cloud.cuh
        includes/scene.cuh
        includes/scene_info.cuh
        includes/camera_utils.cuh
        includes/general_utils.cuh
        includes/sh_utils.cuh
        includes/parameters.cuh
        includes/loss_utils.cuh
        includes/render_utils.cuh
        includes/rasterizer.cuh
        includes/rasterize_points.cuh
        includes/camera_info.cuh
        includes/debug_utils.cuh
        cuda_rasterizer/backward.h
        cuda_rasterizer/forward.h
        cuda_rasterizer/auxiliary.h
        cuda_rasterizer/rasterizer_impl.h
        cuda_rasterizer/rasterizer.h
        cuda_rasterizer/serialization.h
)

set(SOURCES
        src/main.cu
        src/read_utils.cu
        src/gaussian.cu
        src/camera.cu
        src/loss_monitor.cu
        src/image.cu
        src/scene.cu
        src/camera_utils.cu
        src/general_utils.cu
        src/rasterizer.cu
        src/rasterize_points.cu
        cuda_rasterizer/backward.cu
        cuda_rasterizer/forward.cu
        cuda_rasterizer/rasterizer_impl.cu
)

# Get rid of pesky warnings from stb_image.h produced in src/utils.cu
set_source_files_properties(src/utils.cu PROPERTIES COMPILE_FLAGS "-Xcudafe --diag_suppress=550 ")

# --------------------------------------
# Project target
# --------------------------------------

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/includes ${PROJECT_SOURCE_DIR}/cuda_rasterizer ${PROJECT_SOURCE_DIR}/external/tinyply/source)

# Torch setup
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

# Build options based on CMAKE_BUILD_TYPE
function(configure_build_type)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${PROJECT_NAME} PRIVATE -O0 -g -G -Xcompiler -Werror)
        set_target_properties(${PROJECT_NAME} PROPERTIES
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                CUDA_SEPARABLE_COMPILATION ON
        )
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${PROJECT_NAME} PRIVATE -O3 -use_fast_math -Xcompiler -Ofast )
    endif ()
endfunction()
configure_build_type()

# Required packages
find_package(CUDAToolkit REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)

# Check CUDA version
if (CUDAToolkit_VERSION VERSION_LESS "11.7")
    message(FATAL_ERROR "This project requires CUDA 11.7 or higher")
endif ()

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
# --------------------------------------
# External Libraries
# --------------------------------------

add_subdirectory(external)

# --------------------------------------
# Linking
# --------------------------------------

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        simple-knn
        TBB::tbb # parallel reading and writing optimization
        tinyply
        eigen3
        nlohmann_json::nlohmann_json
        glm
        Threads::Threads
        args
)

# --------------------------------------
# Add googletest
# --------------------------------------
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.10.0
    )

    FetchContent_GetProperties(googletest)
    if (NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()

    # Remove -Werror from the compile options of Google Test
    set_target_properties(gtest gtest_main PROPERTIES
            COMPILE_OPTIONS "-Wno-error"
    )

    # Enable CTest
    enable_testing()

    set(TEST_SOURCES
            cuda_rasterizer/backward.cu
            cuda_rasterizer/forward.cu
            cuda_rasterizer/rasterizer_impl.cu
            src/gaussian.cu
            src/camera.cu
            src/image.cu
            src/scene.cu
            src/camera_utils.cu
            src/general_utils.cu
            src/rasterizer.cu
            src/rasterize_points.cu
            src/read_utils.cu
            tests/render_test.cu
            tests/testing.cu
    )
    # Add test executable
    add_executable(testing ${TEST_SOURCES})

    set_target_properties(testing PROPERTIES
            CUDA_ARCHITECTURES native
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
    )
    target_include_directories(testing PRIVATE ${PROJECT_SOURCE_DIR}/includes ${PROJECT_SOURCE_DIR}/cuda_rasterizer ${PROJECT_SOURCE_DIR}/external/tinyply/source)

    target_link_libraries(testing
            PRIVATE
            ${TORCH_LIBRARIES}
            gtest
            gtest_main
            simple-knn
            TBB::tbb
            tinyply
            eigen3
            nlohmann_json::nlohmann_json
            glm
    )
endif()
