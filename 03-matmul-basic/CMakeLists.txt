# 设置最低CMake版本
cmake_minimum_required(VERSION 3.11)
#set project name,languages
PROJECT(trt_cuda VERSION 1.0 LANGUAGES CXX CUDA)
# set cxx 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#CXX
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")
# 自定义CUDA路径
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.5" CACHE PATH "path to cuda toolkit")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
# 查找CUDA工具包
find_package(CUDA REQUIRED)           #cmake 3.17之前
# find_package(CUDAToolkit REQUIRED)  #cmake 3.17之后


# include
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${SRC_DIR})

# add  src path
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
# add cuda file
file(GLOB CUDA_SOURCES ${SRC_DIR}/*.cu)
# add cpp file
file(GLOB CPP_SOURCES ${SRC_DIR}/*.cpp)

# set exec file
add_executable(${PROJECT_NAME} ${CUDA_SOURCES} ${CPP_SOURCES})


# set include
# target_include_directories(${PROJECT_NAME} PRIVATE
#                               ${CUDA_INCLUDE_DIRS})

#set cuda languages
set_target_properties(${PROJECT_NAME} PROPERTIES 
                  # CUDA_ARCHITECTURES "86"
                  # LINK_LANGUAGE CUDA
                  CUDA_SEPARABLE_COMPILATION ON)

# link cuda lib
# target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})


MESSAGE("🐼CMAKE_HOST_SYSTEM_PROCESSOR:${CMAKE_HOST_SYSTEM_PROCESSOR}")
MESSAGE("🐼CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER}")
MESSAGE("🐼CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
MESSAGE("🐼CUDA_LIBRARYS:${CUDA_LIBRARIES}")
MESSAGE("🐼CUDA_SOURCES:${CUDA_SOURCES}")
MESSAGE("🐼CUDA_SOURCES:${CPP_SOURCES}")










