
# Find OpenCV package
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Create a library
add_library(my_frame_processing STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_processing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/video_processing.cpp
)

# Include directories
target_include_directories(my_frame_processing
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(my_frame_processing PUBLIC ${OpenCV_LIBS})

###############################Formatting##########################################
find_program(CLANG_FORMAT_BIN clang-format)
if (NOT CLANG_FORMAT_BIN)
    message(FATAL_ERROR "clang-format not found. Please install clang-format.")
endif()
# Collect all .h, .c, and .cpp files in the src and include directories
file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)

# Set other clang-format options
set(CLANG_FORMAT_OPTIONS
    "-style=llvm"
    "-fallback-style=none"
    "-sort-includes"
    "--verbose"
)

message(STATUS "Running clang-format...")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

# Define the clang-format target
add_custom_target(format
    COMMAND ${CLANG_FORMAT_BIN}
    ${CLANG_FORMAT_OPTIONS}
    -i
    ${SRC_FILES} ${INC_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running clang-format on source files..."
)

# Add a custom command to automatically format before building
add_custom_target(pre_build ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Running clang-format before build..."
     COMMAND ${CLANG_FORMAT_BIN}
    -i
    ${CLANG_FORMAT_OPTIONS}
    ${SRC_FILES} ${INC_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
###################################################################################
