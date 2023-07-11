# CMakeList.txt : Top-level CMake project file, do global configuration and
# include sub-projects here.
#
cmake_minimum_required (VERSION 3.22)


cmake_policy(SET CMP0104 NEW)
SET(CMAKE_CUDA_ARCHITECTURES 89)
SET(CMAKE_CUDA_COMPILER  /usr/local/cuda-12.0/bin/nvcc)
project (gaussian_splatting_cuda LANGUAGES CUDA CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


#set(CMAKE_FIND_DEBUG_MODE TRUE)
find_package(CUDA 12.0 REQUIRED)
#set(CMAKE_FIND_DEBUG_MODE FALSE)
include_directories(${CUDA_INCLUDE_DIRS})

# Include sub-projects.

# src project

set(HEADERS
	includes/read_utils.cuh
	includes/gaussian.cuh
	includes/camera.cuh
	includes/image.cuh
)

set(SOURCES
	# common
	src/main.cu	
	src/read_utils.cu
	src/gaussian.cu
	src/camera.cu
	src/image.cu
)


add_subdirectory(external/tinyply)
include_directories(${PROJECT_SOURCE_DIR}/external/tinyply/source)
include_directories(${PROJECT_SOURCE_DIR}/includes)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_compile_options(${PROJECT_NAME} PRIVATE -g -G -Xcompiler -Werror)

# CUDA includes / linkage
set_target_properties(${PROJECT_NAME} PROPERTIES
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
	${CUDA_LIBRARIES}
	curand
	cublas
	PRIVATE
	tinyply
)
