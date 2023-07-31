cmake_minimum_required (VERSION 3.22)

# Determine the project root directory
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# Define the absolute path to libtorch
set(LIBTORCH_DIR "${PROJ_ROOT_DIR}/external/libtorch")

# Set CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

project(gaussian_splatting_cuda LANGUAGES CUDA CXX)

include(ProcessorCount)
ProcessorCount(total_cores)
if(NOT DEFINED total_cores OR total_cores EQUAL 0)
    set(total_cores 1)
    set(used_cores 1)
elseif(total_cores GREATER 1)
    math(EXPR used_cores "${total_cores} - 2") # use total_cores-2 if total_cores > 1
endif()
set(ENV{MAKEFLAGS} "-j${used_cores}")
message(STATUS "Building with ${used_cores} out of ${total_cores} available cores")

set(HEADERS
	includes/read_utils.cuh
	includes/gaussian.cuh
	includes/camera.cuh
	includes/image.cuh
    includes/utils.cuh
    includes/stb_image.h
    includes/point_cloud.cuh
    includes/scene.cuh
    includes/scene_info.cuh
    includes/camera_utils.cuh
    includes/general_utils.cuh
)

set(SOURCES
	src/main.cu	
	src/read_utils.cu
	src/gaussian.cu
	src/camera.cu
	src/image.cu
    src/utils.cu
    src/scene.cu
    src/scene_info.cu
    src/camera_utils.cu
    src/general_utils.cu
)

# Get rid of pesky warnings from stb_image.h produced in src/utils.cu
set_source_files_properties(src/utils.cu PROPERTIES COMPILE_FLAGS "-Xcudafe --diag_suppress=550")

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


# Torch setup
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

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
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        TBB::tbb # parallel reading and writing optimization
        tinyply
        eigen3
        nlohmann_json::nlohmann_json
)

source_group(TREE "${PROJECT_SOURCE_DIR}/includes" PREFIX "Header Files" FILES ${HEADERS})
source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${SOURCES})