cmake_minimum_required(VERSION 3.24...3.30)


# Get torch from parent
if(NOT DEFINED TORCH_INCLUDE_DIRS)
    find_package(Torch REQUIRED)
endif()

set (DENSEPOINTS_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/src/track_handler.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/matcher.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/adjacency_graph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/dense_point_cloud.cpp
)

set (DENSEPOINTS_HEADERS
	${CMAKE_CURRENT_SOURCE_DIR}/include/track_handler.h
	${CMAKE_CURRENT_SOURCE_DIR}/include/matcher.h
	${CMAKE_CURRENT_SOURCE_DIR}/include/adjacency_graph.h
	${CMAKE_CURRENT_SOURCE_DIR}/include/dense_point_cloud.h
)

add_library(dense_points STATIC ${DENSEPOINTS_SOURCES} ${DENSEPOINTS_HEADERS})

set_target_properties(dense_points PROPERTIES
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(dense_points
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${OPENGL_INCLUDE_DIRS}
		${TORCH_INCLUDE_DIRS}
)

target_link_libraries(dense_points
	PUBLIC
		${TORCH_LIBRARIES}
		gaussian_host  # Circular dependency with gaussian_host. Not ideal.
		CUDA::cudart
)

# Compile options for both CUDA and C++
target_compile_options(dense_points PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 --use_fast_math --expt-relaxed-constexpr -diag-suppress=20012,186>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
)
