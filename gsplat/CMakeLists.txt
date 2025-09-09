cmake_minimum_required(VERSION 3.30)

# Get torch from parent
if(NOT DEFINED TORCH_INCLUDE_DIRS)
    find_package(Torch REQUIRED)
endif()

# Ensure CUDA targets are available from parent
if(NOT TARGET CUDA::cudart)
    message(FATAL_ERROR "CUDA::cudart not found - ensure parent CMakeLists.txt sets up CUDA properly")
endif()

if(NOT TARGET CUDA::curand)
    message(FATAL_ERROR "CUDA::curand not found - ensure parent CMakeLists.txt sets up CUDA properly")
endif()

if(NOT TARGET CUDA::cublas)
    message(FATAL_ERROR "CUDA::cublas not found - ensure parent CMakeLists.txt sets up CUDA properly")
endif()

# All gsplat sources together
set(GSPLAT_SOURCES
        # C++ files
        Intersect.cpp
        Projection.cpp
        QuatToRotmat.cpp
        Rasterization.cpp
        Relocation.cpp
        SphericalHarmonics.cpp

        # CUDA files
        IntersectTile.cu
        ProjectionUT3DGSFused.cu
        QuatToRotmatCUDA.cu
        RasterizeToPixelsFromWorld3DGSBwd.cu
        RasterizeToPixelsFromWorld3DGSFwd.cu
        RelocationCUDA.cu
        SphericalHarmonicsCUDA.cu
)

# One unified library
add_library(gsplat_backend STATIC ${GSPLAT_SOURCES})

set_target_properties(gsplat_backend PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${gaussian_splatting_cuda_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gsplat_backend
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
        PRIVATE
)

target_link_libraries(gsplat_backend
        PUBLIC
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        glm::glm
)

target_compile_options(gsplat_backend PRIVATE
  # CUDA
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g -G -lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math --expt-relaxed-constexpr --diag-suppress=20012,186,221>

  # C++ MSVC
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/Od /Z7>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /DNDEBUG>

  # C++ no-MSVC (GCC/Clang/AppleClang/IntelLLVM)
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -DNDEBUG>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(gsplat_backend PRIVATE DEBUG DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(gsplat_backend PRIVATE RELEASE_BUILD)
endif()

# Print debug info
message(STATUS "gsplat_backend configuration:")
message(STATUS "  - CUDA targets available: ${_archs}")
message(STATUS "  - Sources: ${GSPLAT_SOURCES}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - C++ Standard: 23")
message(STATUS "  - CUDA Standard: 20")
message(STATUS "  - CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  - Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "  - Torch libraries: ${TORCH_LIBRARIES}")
