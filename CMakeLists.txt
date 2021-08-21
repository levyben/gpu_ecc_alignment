cmake_minimum_required(VERSION 3.17)  # CMake version check
project(ecc_gpu)               
set(CMAKE_CXX_STANDARD 14)         
find_package(OpenCV 4.5 REQUIRED )
find_package(CUDAToolkit REQUIRED)

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

include_directories(ecc)

set(SOURCE_FILES ecc/main.cpp ecc/ecc_cuda.cpp ecc/ecc_cuda.h )

add_executable(ecc_gpu ${SOURCE_FILES})
TARGET_LINK_LIBRARIES (ecc_gpu ${OpenCV_LIBS} ${CUDA_LIBRARY})         
