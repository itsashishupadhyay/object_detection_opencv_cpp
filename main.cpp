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
  std::cout << "  -h, --help        Display this help message\n";
  std::cout << "  -i, --image       Process an image\n";
  std::cout << "  -w, --webcam      Process the webcam\n";
  std::cout << "  -v, --video       Process a video\n";
  std::cout << "  -o, --object      Process an object detection on image\n";
  std::cout << "  -d, --detect      Process an object detection on video\n";
  std::cout << "  -p, --path        Specify the path to the image or video\n";
  std::cout << "  -l, --label       Specify the path to the label file\n";
  std::cout << "  -m, --model       Specify the path to the ONNX model file\n";
  std::cout << "\n";
  std::cout << "Examples:\n";
  std::cout << "  ./program_name -i -p /path/to/image.jpg\n";
  std::cout << "  ./program_name -v -p /path/to/video.mp4\n";
  std::cout << "  ./program_name -o -p /path/to/image.jpg -l "
               "/path/to/label.txt -m /path/to/model.onnx\n";
  std::cout << "  ./program_name -d -p /path/to/video.mp4 -l "
               "/path/to/label.txt -m /path/to/model.onnx\n";
}

int main(int argc, char **argv) {
  std::string path2file;
  std::string path2label = "";
  std::string path2onnxmodel = "";
  bool helpFlag = false;
  bool imageFlag = false;
  bool videoFlag = false;
  bool pathFlag = false;
  bool webcamFlag = false;
  bool object_detection = false;
  bool object_detection_video = false;

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
    } else if (arg == "-o" || arg == "--object") {
      object_detection = true;
    } else if (arg == "-d" || arg == "--detect") {
      object_detection_video = true;
    } else if (arg == "-p" || arg == "--path") {
      pathFlag = true;
      if (i + 1 < argc) {
        path2file = argv[i + 1];
        i++; // Skip the next argument since it is the path value
      } else {
        std::cout << "Error: Path argument requires a value.\n";
        return 1;
      }
    } else if (arg == "-l" || arg == "--label") {
      if (i + 1 < argc) {
        path2label = argv[i + 1];
        i++; // Skip the next argument since it is the label value
      } else {
        std::cout << "Error: Label argument requires a value.\n";
        return 1;
      }
    } else if (arg == "-m" || arg == "--model") {
      if (i + 1 < argc) {
        path2onnxmodel = argv[i + 1];
        i++; // Skip the next argument since it is the model value
      } else {
        std::cout << "Error: Model argument requires a value.\n";
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

  if (!imageFlag && !videoFlag && !webcamFlag && !object_detection &&
      !object_detection_video) {
    std::cout << "Error: Either -i/--image or -v/--video or -w/--webcam or "
                 "-o/--object or -d/--detect flag must be specified.\n";
    return 1;
  }

  if (imageFlag) {
    if (!pathFlag) {
      std::cout << "Error: -i/--image flag requires -p/--path argument.\n";
      return 1;
    }
    DETECTION_IMAGE_PROCESSING::image_processing my_image;
    my_image.IMAGE_TEST_BLOCK(path2file);
  }

  if (object_detection) {
    if (!pathFlag) {
      std::cout << "Error: -o/--object flag requires -p/--path argument.\n";
      return 1;
    }
    DETECTION_IMAGE_PROCESSING::image_processing my_image;
    my_image.detect_objects_in_image(path2file, path2label, path2onnxmodel);
  }

  if (videoFlag) {
    if (!pathFlag) {
      std::cout << "Error: -i/--image flag requires -p/--path argument.\n";
      return 1;
    }
    DETECTION_VIDEO_PROCESSING::video_processing my_video;
    my_video.display_video(path2file);
  }

  if (object_detection_video) {
    if (!pathFlag) {
      std::cout << "Error: -d/--detect flag requires -p/--path argument.\n";
      return 1;
    }
    DETECTION_VIDEO_PROCESSING::video_processing my_video;
    my_video.run_object_detetion(path2file, path2label, path2onnxmodel);
  }

  if (webcamFlag) {
    DETECTION_VIDEO_PROCESSING::video_processing my_video;
    my_video.display_webcam();
    return 0;
  }

  return 0;
}