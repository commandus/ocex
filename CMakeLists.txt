# https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases
cmake_minimum_required(VERSION 3.9)
project(ocex VERSION 1.0)

INCLUDE_DIRECTORIES()

set(SRC_OCEX
	hello.cpp
)

add_executable(ocex ${SRC_OCEX})
target_link_libraries(ocex PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ocl-sdk-light/lib/x86_64/opencl.lib")
target_include_directories(ocex PRIVATE "third_party/ocl-sdk-light/include")
