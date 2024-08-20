# object_detection_opencv_cpp
## Requirements
To successfully compile and run this project, the following requirements must be met:
1. clang-format
clang-format is a tool to format C++ code according to a set of style rules. To install clang-format, follow these steps:
```
# On Ubuntu-based systems
sudo apt-get install clang-format

# On macOS (with Homebrew)
brew install clang-format
```
2. CMake
CMake is a cross-platform build system generator. To install CMake, follow these steps:
```
# On Ubuntu-based systems
sudo apt-get install cmake

# On macOS (with Homebrew)
brew install cmake
```
3. OpenCV
OpenCV is a computer vision library. To install OpenCV, follow these steps:
```
# On Ubuntu-based systems
sudo apt-get install libopencv-dev

# On macOS (with Homebrew)
brew install opencv
```

### Directory Structure

The project directory structure is as follows:
```include 
object_detection_opencv_cpp
├── .gitignore
├── .gitmodule
├── CMakeLists.txt
├── README.md
├── external_components
│   └── yolov5
│         └── .......     
├── libs
│   ├── CMakeLists.txt
│   ├── inc
│   │   ├── image_processing.h
│   │   └── video_processing.h
│   └── src
│       ├── image_processing.cpp
│       └── video_processing.cpp
├── main.cpp
└── weight
    ├── coco.names
    └── yolov5s.onnx
```

## How to Get the YOLO ONNX Weight
To obtain the YOLO ONNX weight, follow these steps:
```
# Navigate to the external_components/yolov5 directory
cd external_components/yolov5

# Install required libraries
python -m pip install -r requirements.txt

# Run the export script to generate the YOLO ONNX weight
python export.py \
--weights yolov5s.pt \
--img 640 \
--simplify \
--optimize \
--include onnx

# Validate the generated YOLO ONNX weight
python detect.py --weights yolov5s.onnx --dnn

# Move the generated YOLO ONNX weight to the /weight directory
mv yolov5s.onnx ../weight/
```
## Compile and Run
To compile and run the project, follow these steps:
#### Debug Build
```
# Create a debug build
cmake -DCMAKE_BUILD_TYPE=Debug ../CMakeLists.txt && make all

# Run the debug binary
./opencv_cpp_debug -i -d -l '~/object_detection_opencv_cpp/weight/coco.names' -m '~/object_detection_opencv_cpp/weight/yolov5s.onnx' -p '~/object_detection_opencv_cpp/external_components/yolov5/data/images/bus.jpg'
```
#### Release Build
```
# Create a release build
cmake -DCMAKE_BUILD_TYPE=Release ../CMakeLists.txt && make all

# Run the release binary
./opencv_cpp_release -i -d -l '~/object_detection_opencv_cpp/weight/coco.names' -m '~/object_detection_opencv_cpp/weight/yolov5s.onnx' -p '~/object_detection_opencv_cpp/external_components/yolov5/data/images/bus.jpg'
```
### Help Menu
To view the help menu for the binary, run the following command:
```
./opencv_cpp_debug -h
```
>Note: Make sure to replace ~/object_detection_opencv_cpp/ with the actual path to the project directory.

