cmake_minimum_required(VERSION 3.30.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(opencv_cpp VERSION 0.1.0 LANGUAGES C CXX)

add_executable(opencv_cpp main.cpp)

add_subdirectory(libs)

target_link_libraries(opencv_cpp PUBLIC my_frame_processing)
