#include "image_processing.h"
#include <iostream>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./program_name path_to_image\n";
        return 1;
    }

    char** path2image = &argv[1];
    DETECTION_IMAGE_PROCESSING::image_processing my_image;
    my_image.display_image(path2image);

    return 0;
}