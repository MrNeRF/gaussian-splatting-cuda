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

#target_compile_options(${PROJECT_NAME} PRIVATE -g -G -Xcompiler -Werror)
target_compile_options(${PROJECT_NAME} PRIVATE -O3) # Uncomment for release mode faster compilation

#set_target_properties(${PROJECT_NAME} PROPERTIES
#	CUDA_RESOLVE_DEVICE_SYMBOLS ON
#	CUDA_SEPARABLE_COMPILATION ON
#)

find_package(CUDAToolkit REQUIRED)

# Check CUDA version
if(CUDAToolkit_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "This project requires CUDA 12.0 or higher")
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        tinyply
        eigen3
)

source_group(TREE "${PROJECT_SOURCE_DIR}/includes" PREFIX "Header Files" FILES ${HEADERS})
source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${SOURCES})