# 设置最低CMake版本
cmake_minimum_required(VERSION 3.11)
#set project name,languages
PROJECT(trt_cuda VERSION 1.0 LANGUAGES CXX CUDA)
# set cxx 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 自定义CUDA路径
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.5" CACHE PATH "path to cuda toolkit")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
# 查找CUDA工具包
find_package(CUDA REQUIRED)           #cmake 3.17之前
# find_package(CUDAToolkit REQUIRED)  #cmake 3.17之后
# include
include_directories(${CUDA_INCLUDE_DIRS}/include)

# add cuda src
set(CUDA_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
# add cuda file
file(GLOB CUDA_SOURCES ${CUDA_SRC_DIR}/*.cu)

# set exec file
add_executable(${PROJECT_NAME} ${CUDA_SOURCES})

# link cuda lib
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

#set cuda languages
set_target_properties(${PROJECT_NAME} PROPERTIES 
                  CUDA_ARCHITECTURES "86"
                  LINK_LANGUAGE CUDA)



MESSAGE("🐼CMAKE_HOST_SYSTEM_PROCESSOR:${CMAKE_HOST_SYSTEM_PROCESSOR}")
MESSAGE("🐼CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
MESSAGE("🐼CUDA_LIBRARYS:${CUDA_LIBRARIES}")
MESSAGE("🐼CUDA_SOURCES:${CUDA_SOURCES}")










