cmake_minimum_required (VERSION 3.22)

project(gaussian_splatting_cuda LANGUAGES CUDA CXX)

set(HEADERS
	includes/read_utils.cuh
	includes/gaussian.cuh
	includes/camera.cuh
	includes/image.cuh
    includes/utils.cuh
    includes/stb_image.h
)

set(SOURCES
	src/main.cu	
	src/read_utils.cu
	src/gaussian.cu
	src/camera.cu
	src/image.cu
    src/utils.cu
)

add_subdirectory(external)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/includes)

#target compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${PROJECT_NAME} PRIVATE -g -G -Xcompiler -Werror)
  set_target_properties(${PROJECT_NAME} PROPERTIES
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_SEPARABLE_COMPILATION ON
  )
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${PROJECT_NAME} PRIVATE -O3)
endif()


find_package(CUDAToolkit REQUIRED)
find_package(TBB REQUIRED)

#Check CUDA version
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "This project requires CUDA 12.0 or higher")
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        TBB::tbb # tbb is used for later optimization
        tinyply
        eigen3
)

source_group(TREE "${PROJECT_SOURCE_DIR}/includes" PREFIX "Header Files" FILES ${HEADERS})
source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${SOURCES})