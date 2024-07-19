#include "image_processing.h"
#include "video_processing.h"
#include <iostream>


#include "image_processing.h"
#include <iostream>
#include <string>
#include <vector>


void help_menu() {
    std::cout << "Usage: ./program_name [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help    Display this help message\n";
    std::cout << "  -i, --image   Process an image\n";
    std::cout << "  -v, --video   Process a video\n";
    std::cout << "  -p, --path    Specify the path to the image or video\n";
}

int main(int argc, char** argv) {
    char** path2file;
    bool helpFlag = false;
    bool imageFlag = false;
    bool videoFlag = false;
    bool pathFlag = false;
    bool webcamFlag = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            helpFlag = true;
        } else if (arg == "-i" || arg == "--image") {
            imageFlag = true;
        } else if (arg == "-v" || arg == "--video") {
            videoFlag = true;
        } else if (arg == "-w" || arg == "--webcam") {
            webcamFlag = true;
        }else if (arg == "-p" || arg == "--path") {
            pathFlag = true;
            if (i + 1 < argc) {
                path2file = &argv[i + 1];
                i++; // Skip the next argument since it is the path value
            } else {
                std::cout << "Error: Path argument requires a value.\n";
                return 1;
            }
        } else {
            std::cout << "Error: Unknown argument '" << arg << "'.\n";
            return 1;
        }
    }

    if (helpFlag) {
        help_menu();
        return 0;
    }

    if (!imageFlag && !videoFlag && !webcamFlag) {
        std::cout << "Error: Either -i/--image or -v/--video or -w/--webcam flag must be specified.\n";
        return 1;
    }

    if (imageFlag) {
        if (!pathFlag) {
            std::cout << "Error: -i/--image flag requires -p/--path argument.\n";
            return 1;
        }
        DETECTION_IMAGE_PROCESSING::image_processing my_image;
        my_image.display_image(path2file);
        my_image.display_image_greyscale(path2file);
    }

    if (videoFlag){
        if (!pathFlag) {
            std::cout << "Error: -i/--image flag requires -p/--path argument.\n";
            return 1;
        }
        DETECTION_VIDEO_PROCESSING::video_processing my_video;
        my_video.display_video(path2file);
    }

    if (webcamFlag) {
        DETECTION_VIDEO_PROCESSING::video_processing my_video;
        my_video.display_webcam();
        return 0;
    }

    return 0;
}