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
# GSPLAT LIBRARY - New backend
# =============================================================================

set(GSPLAT_SOURCES
        gsplat/Adam.cpp
        gsplat/AdamCUDA.cu
        gsplat/Intersect.cpp
        gsplat/IntersectTile.cu
        gsplat/Null.cpp
        gsplat/NullCUDA.cu
        gsplat/Projection.cpp
        gsplat/ProjectionEWA3DGSFused.cu
        gsplat/ProjectionEWA3DGSPacked.cu
        gsplat/ProjectionEWASimple.cu
        gsplat/ProjectionUT3DGSFused.cu
        gsplat/QuatScaleToCovar.cpp
        gsplat/QuatScaleToCovarCUDA.cu
        gsplat/Rasterization.cpp
        gsplat/RasterizeToIndices3DGS.cu
        gsplat/RasterizeToPixels3DGSBwd.cu
        gsplat/RasterizeToPixels3DGSFwd.cu
        gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu
        gsplat/RasterizeToPixelsFromWorld3DGSFwd.cu
        gsplat/Relocation.cpp
        gsplat/RelocationCUDA.cu
        gsplat/SphericalHarmonics.cpp
        gsplat/SphericalHarmonicsCUDA.cu
        # 2DGS files if needed
        gsplat/Projection2DGSFused.cu
        gsplat/Projection2DGSPacked.cu
        gsplat/RasterizeToIndices2DGS.cu
        gsplat/RasterizeToPixels2DGSBwd.cu
        gsplat/RasterizeToPixels2DGSFwd.cu
)

add_library(gsplat_backend STATIC ${GSPLAT_SOURCES})

set_target_properties(gsplat_backend PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        # Enable device linking for this library
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gsplat_backend
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/gsplat
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(gsplat_backend
        PUBLIC
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        ${TORCH_LIBRARIES}
)

# CUDA optimizations for gsplat - only apply to CUDA files
target_compile_options(gsplat_backend PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 -use_fast_math --ptxas-options=-v>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g -fno-omit-frame-pointer>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
)

# Ensure debug symbols are kept
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gsplat_backend PRIVATE _DEBUG)
endif()

# =============================================================================
# HOST LIBRARY - Compiled with g++ (fast!)
# =============================================================================

set(HOST_SOURCES
        src/main.cpp
        src/inria_adc.cpp
        src/camera.cpp
        src/mean_neighbor_dist.cpp
        src/camera_utils.cpp
        src/rasterizer.cpp
        src/read_utils.cpp
        src/parameters.cpp
        src/splat_data.cpp
        src/trainer.cpp
        src/argument_parser.cpp
        src/gsplat_rasterizer.cpp
        src/external/tinyply.cpp
)

add_library(gaussian_host STATIC ${HOST_SOURCES})

# Set up include directories for host library
target_include_directories(gaussian_host
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/gsplat  # Add gsplat headers
        PRIVATE
        ${PYTHON_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(gaussian_host
        PUBLIC
        ${TORCH_LIBRARIES}
        nlohmann_json::nlohmann_json
        glm
        Threads::Threads
        args
        ${PYTHON_LIBRARIES}
        Python3::Python
        gsplat_backend  # Link to gsplat
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gaussian_host PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found – compiling with parallel point-cloud loader")
else()
    message(STATUS "OpenMP NOT found – falling back to single-thread loader")
endif()

# Fast C++ compilation with proper debug symbols
target_compile_options(gaussian_host PRIVATE
        $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer -DDEBUG>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
)

# Ensure debug symbols in debug builds
set_target_properties(gaussian_host PROPERTIES
        DEBUG_POSTFIX d
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
endif()

# =============================================================================
# KERNEL LIBRARY - Keep minimal for backwards compatibility
# =============================================================================

set(KERNEL_SOURCES
        kernels/ssim.cu
)

add_library(gaussian_kernels STATIC ${KERNEL_SOURCES})

set_target_properties(gaussian_kernels PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        # Enable device linking for this library
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gaussian_kernels
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels
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

target_compile_options(gaussian_kernels PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 -use_fast_math --ptxas-options=-v>
)

# =============================================================================
# MAIN EXECUTABLE - WITH CUDA DEVICE LINKING
# =============================================================================

# Create the main executable with device linking enabled
add_executable(${PROJECT_NAME} src/main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        # This is the key property for device linking
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/gsplat
        ${PYTHON_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

# Link directly to all libraries
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        gaussian_host
        gaussian_kernels
        gsplat_backend
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
configure_build_type(gsplat_backend)
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
        )
        target_link_libraries(test_host PRIVATE gaussian_host)
        add_test(NAME HostTests COMMAND test_host)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernels.cu")
        add_executable(test_kernels tests/test_kernels.cu)
        set_target_properties(test_kernels PROPERTIES
                CUDA_ARCHITECTURES native
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        target_include_directories(test_kernels PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
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
message(STATUS "  - Python: ${PYTHON_INCLUDE_DIRS}")

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()