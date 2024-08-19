# ============================================================================
#  Project:             OpenCV-CPP
#  Version:             1.0.0
#  Author:              Ashish Upadhyay
#  Copyright:           2023 Ashish Upadhyay
#  License:             MIT License (see LICENSE file for details)
#  Description:         A C++ wrapper for OpenCV library (TODO) to perform object detection 
#  Repository:          https://github.com/itsashishupadhyay/object_detection_opencv_cpp
#  Contributors:        Ashish Upadhyay
#  Disclaimer:          This software is distributed on an "AS IS" BASIS, WITHOUT
#                       WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#                       implied.
# ============================================================================

cmake_minimum_required(VERSION 3.30.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME opencv_cpp)

project(${PROJECT_NAME} VERSION 0.1.0 LANGUAGES C CXX)

# Set the build type to DEBUG or RELEASE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

# Set the compiler flags for DEBUG and RELEASE builds
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set the executable suffix based on the build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(EXECUTABLE_SUFFIX "_debug")
else()
  set(EXECUTABLE_SUFFIX "_release")
endif()

add_executable(${PROJECT_NAME}${EXECUTABLE_SUFFIX} main.cpp)
add_subdirectory(libs)
target_link_libraries(${PROJECT_NAME}${EXECUTABLE_SUFFIX} PUBLIC my_frame_processing)