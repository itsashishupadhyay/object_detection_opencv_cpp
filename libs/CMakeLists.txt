
# Find OpenCV package
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Create a library
add_library(my_image_processing STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_processing.cpp
)

# Include directories
target_include_directories(my_image_processing
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(my_image_processing PUBLIC ${OpenCV_LIBS})