add_library(eigen3 INTERFACE IMPORTED GLOBAL)
target_include_directories(eigen3 INTERFACE eigen)
add_subdirectory(tinyply)
add_subdirectory(json)
add_subdirectory(args)
add_library(glm INTERFACE IMPORTED GLOBAL)
target_include_directories(glm INTERFACE glm)
# --------------------------------------
# simple-knn library
# --------------------------------------

set(SIMPLE_KNN_SOURCES
    simple-knn/simple_knn.cu
    simple-knn/spatial.cu
)

add_library(simple-knn ${SIMPLE_KNN_SOURCES})

set_target_properties(simple-knn PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_STANDARD 17 
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17 
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(simple-knn 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/simple-knn # we need this public to easily include the spatical.h header in our main program
    PRIVATE 
    ${TORCH_INCLUDE_DIRS}
)
