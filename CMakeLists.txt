cmake_minimum_required(VERSION 3.30.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(opencv_cpp VERSION 0.1.0 LANGUAGES C CXX)


add_executable(opencv_cpp main.cpp)


add_subdirectory(libs)

target_link_libraries(opencv_cpp PUBLIC my_frame_processing)

# Run clang-tidy on source and include files in this directory only
find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
if(CLANG_TIDY_COMMAND)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    file(SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    file(INC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
    add_custom_target(run_clang_tidy
        COMMAND ${CLANG_TIDY_COMMAND} -p=${CMAKE_BINARY_DIR} ${SRC_FILES} ${INC_FILES}
        COMMENT "Running clang-tidy on source and include files"
    )
endif()