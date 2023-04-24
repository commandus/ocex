cmake_minimum_required(VERSION 3.9)
project(ocex VERSION 1.0)

set(SRC_OCEX
	hello.cpp
)

add_executable(ocex ${SRC_OCEX})
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
# https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases
	target_include_directories(ocex PRIVATE "third_party/ocl-sdk-light/include")
	target_link_libraries(ocex PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ocl-sdk-light/lib/x86_64/opencl.lib")
else()	
# sudo apt install opencl-headers ocl-icd-opencl-dev -y
	target_link_libraries(ocex PRIVATE OpenCL)
endif()

