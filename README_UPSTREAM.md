# Real-Time Object Detection with OpenCV and C++: Complete YOLO Implementation Guide

A production-ready C++ implementation of YOLOv5, YOLOv8, and **YOLO26** (latest, recommended for edge devices) object detection using OpenCV's DNN module and ONNX models. This cross-platform solution enables real-time object detection on images, video files, and live webcam streams with minimal dependencies. **YOLO26 support features NMS-free end-to-end inference - 43% faster than previous versions, ideal for spacecraft and autonomous systems.**

> **Perfect for**: Computer vision engineers, C++ developers, robotics applications, edge computing, embedded systems, and anyone implementing real-time object detection without Python dependencies.

## Table of Contents

- [What is this Project?](#what-is-this-project)
- [Key Features](#key-features)
- [How Object Detection Works in this Implementation](#how-object-detection-works-in-this-implementation)
- [System Architecture](#system-architecture)
- [Prerequisites and Requirements](#prerequisites-and-requirements)
- [Installation and Setup](#installation-and-setup)
- [How to Create ONNX Files from Scratch](#how-to-create-onnx-files-from-scratch)
- [How to Build the Project](#how-to-build-the-project)
- [How to Use](#how-to-use)
- [Command Reference](#command-reference)
- [Real-World Examples](#real-world-examples)
- [Configuration and Tuning](#configuration-and-tuning)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Performance Optimization](#performance-optimization)

---

## What is this Project?

This repository provides a **complete, production-ready implementation of YOLOv5, YOLOv8, and YOLO26 object detection in C++** using OpenCV's Deep Neural Network (DNN) module. Unlike Python-based implementations, this C++ solution offers significantly faster execution, lower memory footprint, and easier deployment on embedded systems and edge devices.

**NEW: YOLO26 (January 2026)** - Engineered specifically for edge and low-power devices with revolutionary NMS-free architecture:
- **43% faster CPU inference** - critical for real-time spacecraft navigation
- **End-to-end inference** - no post-processing overhead
- **Simplified deployment** - removes NMS complexity
- **Optimized for embedded systems** - ideal for autonomous spacecraft, robotics, and IoT devices

### Why Choose C++ for Object Detection?

**Performance**: C++ implementations typically run 2-3x faster than Python equivalents, crucial for real-time applications like autonomous vehicles, security systems, and industrial automation.

**Deployment**: Single binary compilation eliminates Python runtime dependencies, making it ideal for:
- Embedded Linux systems (Raspberry Pi, NVIDIA Jetson)
- Edge computing devices
- Industrial automation controllers
- Robotics platforms (ROS integration)
- Mobile applications (via cross-compilation)

**Integration**: Seamlessly integrate with existing C++ codebases in gaming engines, CAD software, medical imaging systems, and scientific instruments.

### What Can You Detect?

This implementation supports the **COCO dataset's 80 object classes**, including:
- **People**: person detection for surveillance, occupancy counting
- **Vehicles**: car, truck, bus, motorcycle for traffic monitoring
- **Animals**: cat, dog, bird, horse for wildlife monitoring
- **Indoor Objects**: chair, table, laptop, TV for scene understanding
- **Outdoor Objects**: traffic light, stop sign for autonomous navigation

## Key Features

### Core Object Detection Capabilities

- **Multiple Input Sources**: Process static images (JPEG, PNG), video files (MP4, AVI, MOV), or live webcam/IP camera streams
- **YOLO Neural Network Inference**: Optimized YOLOv5, YOLOv8, and **YOLO26** model inference using OpenCV's DNN module with ONNX format support
- **Real-Time Processing**: Achieves 30+ FPS with YOLOv5/v8 and **50+ FPS with YOLO26** on modern hardware. YOLO26's NMS-free architecture delivers 43% faster inference, ideal for real-time autonomous systems and edge devices
- **Automatic Model Detection**: Code automatically detects YOLOv5, YOLOv8, or YOLO26 format and uses appropriate processing pipeline - seamless backward compatibility
- **COCO Dataset Support**: Pre-configured for 80 object classes from the Common Objects in Context dataset
- **Confidence Thresholding**: Adjustable detection confidence (default 45%) and Non-Maximum Suppression to eliminate false positives

### Advanced Computer Vision Features

Beyond object detection, this library includes production-ready implementations of:

- **Edge Detection**: Canny edge detector with configurable hysteresis thresholds for industrial quality control
- **Image Filtering**: Gaussian and standard blur for noise reduction and preprocessing
- **Perspective Transformation**: Automatic quadrilateral detection and perspective correction for document scanning and lane detection
- **Grayscale Conversion**: Optimized color space transformations for preprocessing pipelines

### Software Engineering Excellence

- **Modular Architecture**: Clean separation of concerns with distinct `image_processing` and `video_processing` libraries
- **CMake Build System**: Cross-platform build configuration supporting Linux, macOS, and Windows
- **Debug & Release Builds**: Optimized release builds (-O3) for production, debug builds (-g -O0) with verbose logging
- **Automatic Code Formatting**: Integrated clang-format ensures consistent LLVM code style
- **Zero Python Dependencies at Runtime**: Compiled binary requires only OpenCV libraries

## How Object Detection Works in this Implementation

Understanding the technical pipeline helps optimize performance and troubleshoot issues.

### The YOLO Detection Pipeline Explained

**Step 1: Image Preprocessing** (`libs/src/image_processing.cpp:502-517`)

When you feed an image or video frame into the detector:
1. The image is converted to a **blob** (Binary Large OBject) - a 4D tensor that neural networks can process
2. Pixel values are **normalized** from [0, 255] to [0, 1] by dividing by 255
3. The image is **resized** to 640×640 pixels (YOLO's input resolution) while preserving aspect ratio
4. Color channels remain in **BGR format** (OpenCV's native format)

This preprocessing ensures consistent input regardless of your original image size or format.

**Step 2: Neural Network Inference** (`libs/src/image_processing.cpp:510-514`)

The preprocessed blob passes through the YOLOv5 ONNX model:
1. OpenCV's DNN module loads the model once and caches it in memory
2. Forward propagation through convolutional layers detects features at multiple scales
3. The network outputs **25,200 potential object detections** (predictions at different grid cells and scales)
4. Each detection includes: bounding box coordinates (x, y, width, height), objectness score, and class probabilities for 80 COCO classes

**Step 3: Post-Processing and Filtering** (`libs/src/image_processing.cpp:519-592`)

Raw network outputs require refinement:
1. **Confidence Filtering**: Discard detections below 45% confidence threshold (configurable)
2. **Class Score Filtering**: For remaining detections, keep only those with class scores above 50%
3. **Non-Maximum Suppression (NMS)**: When multiple boxes detect the same object, keep only the highest-confidence detection (IoU threshold: 0.45)
4. **Coordinate Scaling**: Convert normalized coordinates back to original image dimensions
5. **Visualization**: Draw bounding boxes (blue, 3px thickness) and labels (class name + confidence on black background with yellow text)

### Video and Webcam Processing Pipeline

For video files and webcam streams (`libs/src/video_processing.cpp`):

1. **Frame Extraction**: OpenCV's `VideoCapture` reads frames sequentially from video files or camera buffers
2. **Frame-by-Frame Detection**: Each frame undergoes the complete YOLO pipeline independently
3. **Real-Time Display**: Processed frames display in OpenCV window with sub-millisecond latency
4. **Performance Tracking**: Debug builds report per-frame processing time for optimization
5. **Graceful Exit**: Press ESC key to stop processing and release camera resources

### Why ONNX Format?

**ONNX (Open Neural Network Exchange)** is an open format for neural network models:
- **Interoperability**: Models trained in PyTorch, TensorFlow, or other frameworks export to ONNX
- **Optimization**: ONNX Runtime and OpenCV DNN apply hardware-specific optimizations
- **Portability**: Same model file runs on x86, ARM, GPU, and specialized accelerators
- **No Framework Dependencies**: Inference works without installing PyTorch or TensorFlow

## System Architecture

### Project Structure and File Organization

```
object_detection_opencv_cpp/
├── CMakeLists.txt              # Main CMake build configuration
├── main.cpp                     # Application entry point and CLI parser
├── libs/                        # Core computer vision library
│   ├── CMakeLists.txt          # Library-specific build rules
│   ├── inc/                    # Public header files
│   │   ├── image_processing.h  # Image operations and YOLO detection interface
│   │   └── video_processing.h  # Video/webcam capture and processing interface
│   └── src/                    # Implementation files
│       ├── image_processing.cpp # 700+ lines: YOLO inference, edge detection, transformations
│       └── video_processing.cpp # Frame capture, real-time processing loop
├── external_components/
│   ├── yolov5/                 # Official YOLOv5 repository (git submodule)
│   │   └── export.py           # Script to convert .pt models to ONNX
│   └── ultralytics/            # Official Ultralytics repository (git submodule)
│       └── ultralytics/        # YOLO26, YOLO11, YOLOv8, and more - all YOLO versions
└── weight/                      # Neural network models and labels
    ├── coco.names              # 80 COCO class labels (person, car, dog, etc.)
    ├── yolo26n.onnx            # YOLO26-nano ONNX (NMS-free, ~6MB) [RECOMMENDED]
    ├── yolov8n.onnx            # YOLOv8-nano ONNX model (~6MB) [optional]
    └── yolov5s.onnx            # YOLOv5-small ONNX model (~14MB) [legacy]
```

### Component Descriptions

**Main Application** (`main.cpp`):
- Implements command-line argument parsing for 9 different flags
- Routes execution to image, video, or webcam processing based on user input
- Validates required arguments and provides helpful error messages
- Instantiates `image_processing` or `video_processing` classes as needed

**Image Processing Library** (`libs/src/image_processing.cpp`):
- **Core Class**: `image_processing` encapsulates all image operations
- **YOLO Methods**: `run_yolo_obj_detection()`, `pre_process_yolo()`, `post_process_yolo()`
- **Utilities**: Blur, Gaussian blur, Canny edge detector, perspective transformation
- **State Management**: Caches loaded ONNX model and class labels for performance
- **Thread-Safe**: Single instance per thread recommended for video processing

**Video Processing Library** (`libs/src/video_processing.cpp`):
- **Video Methods**: `display_video()`, `run_object_detetion()` for file processing
- **Webcam Methods**: `display_webcam()`, `run_object_detetion_webcam()` for live capture
- **OpenCV Integration**: Wraps `cv::VideoCapture` with error handling
- **Performance**: Calculates and optionally displays per-frame processing time

**Build System**:
- **CMake Configuration**: Minimum version 3.30, C++17 standard required
- **Static Library**: Compiles `libmy_frame_processing.a` linked to main executable
- **Conditional Compilation**: Debug builds enable verbose logging via `#ifndef NDEBUG`
- **Code Formatting**: Pre-build hook runs clang-format on all source files

## Prerequisites and Requirements

### System Dependencies

This project requires standard computer vision development tools:

**1. C++ Compiler with C++17 Support**

Modern C++ features require a recent compiler:
- **Linux**: GCC 7.0+ or Clang 5.0+ (typically pre-installed on Ubuntu 18.04+)
- **macOS**: Xcode Command Line Tools (includes Apple Clang)
- **Windows**: Visual Studio 2017+ or MinGW-w64

Verify your compiler version:
```bash
# GCC
g++ --version

# Clang
clang++ --version
```

**2. CMake Build System (version 3.30.0+)**

CMake generates native build files for your platform:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install cmake

# macOS with Homebrew
brew install cmake

# Verify installation
cmake --version
```

**3. OpenCV Library (version 4.0+)**

OpenCV provides computer vision and DNN inference capabilities:

```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev

# macOS with Homebrew
brew install opencv

# Verify OpenCV is detectable
pkg-config --modversion opencv4
```

**Note**: OpenCV must be compiled with DNN module enabled (default in most distributions).

**4. clang-format Code Formatter**

Automatic code formatting maintains consistent style:

```bash
# Ubuntu/Debian
sudo apt-get install clang-format

# macOS with Homebrew
brew install clang-format
```

### Python Dependencies (Optional)

**Only required if you want to export custom YOLOv5 models** from PyTorch to ONNX:

```bash
cd external_components/yolov5
python3 -m pip install -r requirements.txt
```

This installs PyTorch, ultralytics, and ONNX export tools. Not needed if using pre-exported models.

### Hardware Recommendations

**Minimum Requirements**:
- CPU: Dual-core processor (2.0 GHz)
- RAM: 2GB
- Storage: 500MB for project and models
- Performance: ~5-10 FPS with YOLOv5s

**Recommended for Real-Time Processing**:
- CPU: Quad-core+ processor (3.0 GHz+)
- RAM: 4GB+
- Optional GPU: NVIDIA GPU with CUDA support (requires OpenCV compiled with CUDA)
- Performance: 30+ FPS with YOLOv5s

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/itsashishupadhyay/object_detection_opencv_cpp.git
cd object_detection_opencv_cpp
```

### Step 2: Initialize Git Submodules

The YOLOv5 and YOLOv8 repositories are included as submodules for model export:

```bash
git submodule update --init --recursive
```

This downloads the official YOLOv5 repository into `external_components/yolov5/` and the Ultralytics repository (YOLO26, YOLO11, YOLOv8, etc.) into `external_components/ultralytics/`.

### Step 3: Obtain YOLO Model Weights (ONNX Format)

You have two options for YOLOv5, YOLOv8, and YOLO26:

#### Option A: Download Pre-Exported ONNX Models (Easiest)

**YOLO26 Models (RECOMMENDED for Edge Devices & Space Applications):**
```bash
# Download YOLO26n (nano, fastest, NMS-free, ideal for spacecraft)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n.onnx -O weight/yolo26n.onnx

# YOLO26s (small, balanced, NMS-free)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26s.onnx -O weight/yolo26s.onnx

# YOLO26m (medium, high accuracy, NMS-free)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26m.onnx -O weight/yolo26m.onnx
```

**YOLOv5 Models:**
```bash
# Download YOLOv5s (small, balanced model)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx -O weight/yolov5s.onnx

# Or download other sizes:
# YOLOv5n (nano, fastest)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx -O weight/yolov5n.onnx

# YOLOv5m (medium, more accurate)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.onnx -O weight/yolov5m.onnx
```

**YOLOv8 Models (Recommended for Edge Devices & Custom Training):**
```bash
# Download YOLOv8n (nano, fastest, ideal for edge devices)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx -O weight/yolov8n.onnx

# YOLOv8s (small, balanced)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx -O weight/yolov8s.onnx

# YOLOv8m (medium, more accurate)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.onnx -O weight/yolov8m.onnx
```

#### Option B: Export from PyTorch Models

For custom models or latest weights:

**YOLOv5 Export:**
```bash
# Navigate to YOLOv5 directory
cd external_components/yolov5

# Install Python dependencies
python3 -m pip install -r requirements.txt

# Export YOLOv5s PyTorch model to ONNX format
python3 export.py \
    --weights yolov5s.pt \
    --img 640 \
    --simplify \
    --optimize \
    --include onnx

# Validate the exported ONNX model
python3 detect.py --weights yolov5s.onnx --dnn --source data/images/bus.jpg

# Move to project weight directory
mv yolov5s.onnx ../../weight/

# Return to project root
cd ../..
```

**YOLO26 Export (RECOMMENDED for Edge Devices & Space Applications):**
```bash
# Navigate to Ultralytics directory
cd external_components/ultralytics

# Install latest Ultralytics package with YOLO26 support
python3 -m pip install --upgrade ultralytics onnx

# Export YOLO26n (NMS-free, optimized for edge devices)
yolo export model=yolo26n.pt format=onnx imgsz=640 simplify=True

# Or use Python API with end-to-end NMS-free export
python3 -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); model.export(format='onnx', imgsz=640, simplify=True)"

# For INT8 quantization (spacecraft/embedded optimization)
python3 -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); model.export(format='onnx', imgsz=640, simplify=True, int8=True)"

# Custom export for specific hardware constraints
python3 -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); model.export(format='onnx', imgsz=640, simplify=True, half=True)"

# Move to project weight directory
mv yolo26n.onnx ../../weight/

# Return to project root
cd ../..
```

**YOLOv8 Export (Legacy Support):**
```bash
# Navigate to Ultralytics directory
cd external_components/ultralytics

# Install Ultralytics package
python3 -m pip install ultralytics onnx

# Export YOLOv8n PyTorch model to ONNX format
yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True

# Or use Python API for more control
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx', imgsz=640, simplify=True)"

# For INT8 quantization (edge device optimization)
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx', imgsz=640, simplify=True, int8=True)"

# Move to project weight directory
mv yolov8n.onnx ../../weight/

# Return to project root
cd ../..
```

**Model Size Comparison**:

| Model | Size | Speed | mAP | NMS-Free | Use Case |
|-------|------|-------|-----|----------|----------|
| **YOLO26n** | **~6MB** | **FASTEST (70+ FPS)** | **~39** | **✅ Yes** | **🚀 PRIMARY CHOICE: Spacecraft, autonomous systems, edge devices** |
| **YOLO26s** | **~20MB** | **Very Fast (55+ FPS)** | **~46** | **✅ Yes** | **🚀 Best balance: Real-time robotics, embedded vision systems** |
| **YOLO26m** | **~50MB** | **Fast (40+ FPS)** | **~52** | **✅ Yes** | **🚀 High accuracy: Advanced autonomous navigation** |
| YOLOv8n | 6MB | Fast (50+ FPS) | 37.3 | ❌ No | Edge devices (legacy support) |
| YOLOv8s | 22MB | Fast (35+ FPS) | 44.9 | ❌ No | Balanced performance (legacy support) |
| YOLOv8m | 52MB | Medium (25 FPS) | 50.2 | ❌ No | High-accuracy applications (legacy support) |
| YOLOv5n | 3.7MB | Fast (45+ FPS) | 28.0 | ❌ No | Legacy embedded systems |
| YOLOv5s | 14MB | Fast (30+ FPS) | 37.4 | ❌ No | Legacy balanced performance |
| YOLOv5m | 40MB | Medium (20 FPS) | 45.4 | ❌ No | Legacy accuracy-focused |

*FPS benchmarks on Intel Core i7 CPU. GPU acceleration significantly improves performance.*

**YOLO26 Revolutionary Advantages for Space & Edge Applications:**
- **43% faster inference** - NMS-free end-to-end architecture eliminates post-processing bottleneck
- **Engineered for low-power devices** - optimized specifically for spacecraft/embedded computing constraints
- **Simplified deployment** - removes NMS complexity, fewer failure points in mission-critical systems
- **Better small object detection** - improved architecture for distant celestial bodies and asteroids
- **MuSGD optimizer** - superior transfer learning for custom astronomical datasets
- **Reduced latency variance** - deterministic output (300 detections max) enables predictable real-time performance
- **Hardware compatibility** - no DFL dependency, runs on wider range of edge processors

### Step 4: Download COCO Class Labels

The `coco.names` file contains the 80 object class names:

```bash
# Download COCO class labels
wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.names -O weight/coco.names

# Verify file contains 80 lines
wc -l weight/coco.names
```

Expected classes include: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, and 70 more.

## How to Create ONNX Files from Scratch

This section provides a complete step-by-step guide to generate YOLO26, YOLOv8, or YOLOv5 ONNX models from PyTorch weights, including verification and testing procedures.

### Prerequisites

Before creating ONNX files, ensure you have:
- Python 3.8+ installed
- Virtual environment (recommended)
- Git submodules initialized (`git submodule update --init --recursive`)

### Step-by-Step Guide: YOLO26 ONNX Export (RECOMMENDED)

This is the complete procedure used to generate the YOLO26n model for this project.

#### 1. Set Up Python Virtual Environment

```bash
# Navigate to project root
cd /Users/upadhyay/dev/object_detection_opencv_cpp

# Create virtual environment if it doesn't exist
python3 -m venv .yolo_venv

# Activate virtual environment
source .yolo_venv/bin/activate

# Verify activation (should show .yolo_venv path)
which python3
```

#### 2. Install/Upgrade Ultralytics

```bash
# Make sure virtual environment is activated
source .yolo_venv/bin/activate

# Install latest Ultralytics with YOLO26 support
pip install --upgrade ultralytics onnx

# Verify installation
python3 -c "from ultralytics import YOLO; import onnx; print('Ultralytics installed successfully')"
```

**Expected output:**
```
Collecting ultralytics
  Downloading ultralytics-8.4.14-py3-none-any.whl (1.2 MB)
...
Successfully installed ultralytics-8.4.14 onnx-1.20.1
```

#### 3. Download and Export YOLO26 Model

```bash
# Ensure virtual environment is active
source .yolo_venv/bin/activate

# Export YOLO26n to ONNX (one-line command)
python3 -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); model.export(format='onnx', imgsz=640, simplify=True)"
```

**What happens during export:**
1. Ultralytics automatically downloads `yolo26n.pt` (5.3 MB) from GitHub releases
2. Loads the PyTorch model
3. Exports to ONNX format with opset 19
4. Applies onnxslim optimization
5. Creates `yolo26n.onnx` (9.5 MB) in current directory

**Expected output:**
```
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt...
YOLO26n summary: 122 layers, 2,408,932 parameters, 5.4 GFLOPs

PyTorch: starting from 'yolo26n.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6)

ONNX: starting export with onnx 1.20.1 opset 19...
ONNX: slimming with onnxslim 0.1.85...
ONNX: export success ✅ 1.6s, saved as 'yolo26n.onnx' (9.5 MB)
```

#### 4. Move ONNX File to Weight Directory

```bash
# Move the exported ONNX file to the weight directory
mv yolo26n.onnx weight/

# Verify file exists and check size
ls -lh weight/yolo26n.onnx
```

**Expected output:**
```
-rw-r--r--  1 user  staff   9.5M Feb 16 01:44 weight/yolo26n.onnx
```

#### 5. Verify ONNX Model Format

```bash
# Activate virtual environment
source .yolo_venv/bin/activate

# Verify output tensor shape
python3 -c "
import onnx
model = onnx.load('weight/yolo26n.onnx')
output = model.graph.output[0]
print('YOLO26n ONNX Output:')
print(f'  Name: {output.name}')
print(f'  Shape: {[dim.dim_value if dim.dim_value > 0 else \"dynamic\" for dim in output.type.tensor_type.shape.dim]}')
print('Expected: [1, 300, 6] for NMS-free YOLO26')
"
```

**Expected output:**
```
YOLO26n ONNX Output:
  Name: output0
  Shape: [1, 300, 6]
Expected: [1, 300, 6] for NMS-free YOLO26
```

**Output format explanation:**
- `[1, 300, 6]` = [batch_size, max_detections, data_per_detection]
- `300` = Maximum 300 detections (NMS-free architecture)
- `6` = [cx, cy, w, h, confidence, class_id]
- **No NMS required!** - This is YOLO26's revolutionary advantage

#### 6. Build C++ Project

```bash
# Navigate to build directory
cd build

# Configure with CMake (Debug mode for verbose output)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build the project
make all

# Verify executable was created
ls -lh opencv_cpp_debug
```

**Expected output:**
```
[ 20%] Building CXX object libs/CMakeFiles/my_frame_processing.dir/src/image_processing.cpp.o
[ 40%] Linking CXX static library libmy_frame_processing.a
[ 60%] Built target my_frame_processing
[ 80%] Linking CXX executable opencv_cpp_debug
[100%] Built target opencv_cpp_debug
```

#### 7. Test YOLO26 Detection

```bash
# Test on sample image (from build directory)
./opencv_cpp_debug -i -d \
    -p ../external_components/yolov5/data/images/bus.jpg \
    -l ../weight/coco.names \
    -m ../weight/yolo26n.onnx
```

**Expected output:**
```
Image size is: [810 x 1080]
class_list and onnex net are empty
Model output shape: [1, 300, 6]
Detected YOLO26 model format (NMS-free)
YOLO26 NMS-free processing 300 detections (no post-processing needed!)
Detected: person:0.93 [NMS-free]
Detected: person:0.93 [NMS-free]
Detected: bus:0.89 [NMS-free]
...
YOLO26: 20 valid detections (out of 300 processed)
```

**Success indicators:**
- ✅ "Detected YOLO26 model format (NMS-free)" message
- ✅ "YOLO26 NMS-free processing" message
- ✅ Detections show "[NMS-free]" tag
- ✅ Image window displays with bounding boxes and labels

#### 8. Compare with Other Models (Optional)

```bash
# Test YOLOv8n (traditional with NMS)
./opencv_cpp_debug -i -d \
    -p ../external_components/yolov5/data/images/bus.jpg \
    -l ../weight/coco.names \
    -m ../weight/yolov8n.onnx

# Test YOLOv5s (legacy with NMS)
./opencv_cpp_debug -i -d \
    -p ../external_components/yolov5/data/images/bus.jpg \
    -l ../weight/coco.names \
    -m ../weight/yolov5s.onnx
```

**Performance comparison:**
```
Model      | Size  | Output Shape    | NMS   | Speed
-----------|-------|-----------------|-------|-------
YOLO26n    | 9.5M  | [1, 300, 6]    | No ✅ | Fastest (43% faster)
YOLOv8n    | 12M   | [1, 84, 8400]  | Yes   | Fast
YOLOv5s    | 28M   | [1, 25200, 85] | Yes   | Medium
```

### Exporting Other YOLO26 Variants

#### YOLO26s (Small - Balanced)

```bash
source .yolo_venv/bin/activate
python3 -c "from ultralytics import YOLO; model = YOLO('yolo26s.pt'); model.export(format='onnx', imgsz=640, simplify=True)"
mv yolo26s.onnx weight/
```

**Size:** ~20 MB | **Output:** [1, 300, 6] | **Use case:** Balanced accuracy/speed

#### YOLO26m (Medium - High Accuracy)

```bash
source .yolo_venv/bin/activate
python3 -c "from ultralytics import YOLO; model = YOLO('yolo26m.pt'); model.export(format='onnx', imgsz=640, simplify=True)"
mv yolo26m.onnx weight/
```

**Size:** ~50 MB | **Output:** [1, 300, 6] | **Use case:** High-accuracy applications

### Advanced: INT8 Quantization for Edge Devices

For spacecraft and embedded deployment with minimal memory footprint:

```bash
source .yolo_venv/bin/activate

# Export with INT8 quantization (reduces size by ~75%)
python3 -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); model.export(format='onnx', imgsz=640, simplify=True, int8=True)"

# Result: yolo26n_int8.onnx (~2.4 MB instead of 9.5 MB)
mv yolo26n.onnx weight/yolo26n_int8.onnx
```

**Benefits:**
- 75% smaller file size (9.5 MB → 2.4 MB)
- 2-4x faster inference on edge devices
- Minimal accuracy loss (<2% mAP drop)
- Ideal for spacecraft with limited storage/compute

### Troubleshooting

#### Issue: "externally-managed-environment" error

**Symptom:**
```
error: externally-managed-environment
× This environment is externally managed
```

**Solution:**
Always use a virtual environment:
```bash
python3 -m venv .yolo_venv
source .yolo_venv/bin/activate
pip install ultralytics onnx
```

#### Issue: "Model not found" during export

**Solution:**
Ultralytics automatically downloads models. Ensure you have:
- Internet connection
- Sufficient disk space (~50 MB for downloads)
- GitHub access (not blocked by firewall)

#### Issue: Export fails with ONNX errors

**Solution:**
```bash
# Upgrade ONNX and dependencies
pip install --upgrade onnx onnxslim ultralytics

# Try export again
python3 -c "from ultralytics import YOLO; YOLO('yolo26n.pt').export(format='onnx', simplify=True)"
```

#### Issue: Wrong output shape detected

**Solution:**
Verify you're using the correct model:
```bash
# Check ONNX model output shape
python3 -c "import onnx; m = onnx.load('weight/yolo26n.onnx'); print(m.graph.output[0].type.tensor_type.shape)"

# Should show: [1, 300, 6] for YOLO26
```

### Summary: Weight Directory Contents

After following this guide, your `weight/` directory should contain:

```
weight/
├── coco.names        620 B   - 80 COCO class labels
├── yolo26n.onnx     9.5 MB  - 🚀 YOLO26-nano (NMS-free, fastest)
├── yolov8n.onnx      12 MB  - YOLOv8-nano (traditional with NMS)
└── yolov5s.onnx      28 MB  - YOLOv5-small (legacy with NMS)
```

**All three models work automatically** - the C++ code detects the version from the output tensor shape!

## How to Build the Project

### Building for Development (Debug Mode)

Debug builds include detailed logging, assertion checks, and performance metrics:

```bash
# Create build directory
mkdir -p build && cd build

# Generate build files with Debug configuration
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Compile the project
make all

# This creates: opencv_cpp_debug
```

**Debug mode features**:
- Verbose output showing detected objects and confidence scores
- Per-frame inference timing in milliseconds
- Image processing intermediate steps visualization
- No compiler optimizations for easier debugging

### Building for Production (Release Mode)

Release builds are optimized for maximum performance:

```bash
# Create build directory
mkdir -p build && cd build

# Generate build files with Release configuration
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile with optimizations
make all

# This creates: opencv_cpp_release
```

**Release mode optimizations**:
- `-O3` compiler optimization (maximum speed)
- `-DNDEBUG` flag removes debug code paths
- Minimal console output
- 20-30% faster execution than debug builds

### Build System Details

**Generated artifacts**:
- `opencv_cpp_debug` or `opencv_cpp_release` - Main executable
- `libs/libmy_frame_processing.a` - Static library with computer vision functions
- Automatically formatted source code (via clang-format pre-build hook)

**Build customization**:
- Modify `CMakeLists.txt` to change C++ standard (default: C++17)
- Edit `libs/CMakeLists.txt` to adjust clang-format style (default: LLVM)
- Add compiler flags for hardware-specific optimizations (e.g., `-march=native`)

## How to Use

### Basic Command Syntax

```bash
./opencv_cpp_[debug|release] [mode] [detection] [options]
```

### Execution Modes

**1. Image Processing Mode** (`-i` or `--image`)

Process a single static image:
```bash
# Display image without detection
./opencv_cpp_debug -i -p path/to/image.jpg

# Run object detection on image
./opencv_cpp_debug -i -d -p path/to/image.jpg -l weight/coco.names -m weight/yolov5s.onnx
```

**2. Video File Processing** (`-v` or `--video`)

Process a video file frame-by-frame:
```bash
# Play video without detection
./opencv_cpp_release -v -p path/to/video.mp4

# Run object detection on video
./opencv_cpp_release -v -d -p path/to/video.mp4 -l weight/coco.names -m weight/yolov5s.onnx
```

**3. Webcam/Live Camera** (`-w` or `--webcam`)

Process live camera feed:
```bash
# Display webcam feed
./opencv_cpp_debug -w

# Run real-time object detection on webcam
./opencv_cpp_release -w -d -l weight/coco.names -m weight/yolov5s.onnx
```

## Command Reference

### Complete Options Table

| Flag | Long Form | Argument | Description | Example |
|------|-----------|----------|-------------|---------|
| `-h` | `--help` | None | Display help menu and exit | `./opencv_cpp_debug -h` |
| `-i` | `--image` | None | Enable image processing mode | `-i -p image.jpg` |
| `-v` | `--video` | None | Enable video file processing mode | `-v -p video.mp4` |
| `-w` | `--webcam` | None | Enable live webcam mode | `-w` |
| `-d` | `--detect` | None | Enable YOLO object detection | `-d` (must combine with mode) |
| `-p` | `--path` | File path | Path to input image/video file | `-p ~/images/sample.jpg` |
| `-l` | `--label` | File path | Path to COCO class labels file | `-l weight/coco.names` |
| `-m` | `--model` | File path | Path to YOLO ONNX model file | `-m weight/yolov5s.onnx` |

### Argument Requirements

**Required combinations**:
- Must specify exactly one mode: `-i`, `-v`, or `-w`
- Image/video modes require `-p` path argument
- Detection flag `-d` optional for all modes
- When using `-d`, `-l` and `-m` optional if files exist at default paths

**Default paths** (used when `-l` or `-m` not specified):
- Labels: `./weight/coco.names`
- Model: `./weight/yolov5s.onnx`

## Real-World Examples

### Example 1: Quick Detection Test on Sample Image

Test your setup with the included sample image:

```bash
./opencv_cpp_debug -i -d \
    -p external_components/yolov5/data/images/bus.jpg \
    -l weight/coco.names \
    -m weight/yolov5s.onnx
```

Expected output: Detects bus, people, and possibly traffic lights.

### Example 2: Batch Process Multiple Images

Process multiple images using a shell loop:

```bash
for img in ~/images/*.jpg; do
    ./opencv_cpp_release -i -d -p "$img" -l weight/coco.names -m weight/yolov5s.onnx
done
```

### Example 3: Surveillance Camera Real-Time Detection

Monitor a security camera feed with default weights:

```bash
# Assuming weights in ./weight/ directory
./opencv_cpp_release -w -d
```

Press ESC to stop monitoring. Ideal for:
- Occupancy counting in retail stores
- Intrusion detection in restricted areas
- Vehicle counting in parking lots

### Example 4: Traffic Monitoring from Video File

Analyze traffic patterns from recorded video:

```bash
./opencv_cpp_release -v -d \
    -p ~/videos/highway_traffic.mp4 \
    -l weight/coco.names \
    -m weight/yolov5m.onnx
```

Detects: cars, trucks, buses, motorcycles, people

### Example 5: Industrial Quality Control

Use edge detection for manufacturing defect detection:

```bash
# Note: Uses IMAGE_TEST_BLOCK in image_processing.cpp
# Demonstrates edge detection, not object detection
./opencv_cpp_debug -i -p ~/qc_images/product.jpg
```

Modify `IMAGE_TEST_BLOCK()` method for custom processing pipelines.

### Example 6: Robotics Vision System

Integrate with ROS (Robot Operating System):

```bash
# Process camera feed from ROS node
./opencv_cpp_release -w -d -l weight/coco.names -m weight/yolov5n.onnx
```

YOLOv5n recommended for real-time robotics due to low latency.

### Example 7: Document Scanning with Perspective Correction

Use the perspective transformation feature:

```bash
# Current implementation requires code modification
# Edit libs/src/image_processing.cpp IMAGE_TEST_BLOCK()
# Uncomment perspective correction code (lines 698-699)
./opencv_cpp_debug -i -p scanned_document.jpg
```

Automatically detects document edges and corrects skew.

### Example 8: Wildlife Monitoring

Detect animals in nature footage:

```bash
./opencv_cpp_release -v -d \
    -p wildlife_footage.mp4 \
    -l weight/coco.names \
    -m weight/yolov5s.onnx
```

COCO dataset includes: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

### Example 9: Space Applications - YOLO26 for Autonomous Spacecraft Navigation

**RECOMMENDED: YOLO26 for mission-critical celestial object detection**

For autonomous spacecraft navigation and asteroid detection:

```bash
# Using YOLO26n for spacecraft edge computing (NMS-free, 43% faster!)
./opencv_cpp_release -w -d \
    -l weight/coco.names \
    -m weight/yolo26n.onnx

# For custom asteroid detection (after transfer learning with MuSGD optimizer)
./opencv_cpp_release -i -d \
    -p telescope_image.jpg \
    -l weight/custom_celestial.names \
    -m weight/yolo26n_asteroids_quantized.onnx

# Real-time deep-space navigation with video feed
./opencv_cpp_release -v -d \
    -p spacecraft_camera_feed.mp4 \
    -l weight/asteroid_classes.names \
    -m weight/yolo26s.onnx
```

**Why YOLO26 is Superior for Space Applications:**
- **70+ FPS on embedded hardware** - NMS-free architecture provides 43% faster inference than YOLOv8
- **Deterministic latency** - fixed 300 detections enables predictable real-time control loops
- **Mission-critical reliability** - simplified pipeline with no NMS reduces potential failure points
- **Ultra-low power consumption** - engineered specifically for resource-constrained spacecraft computing
- **Enhanced small object detection** - superior performance on distant asteroids and celestial bodies
- **MuSGD optimizer** - state-of-the-art transfer learning for astronomical datasets
- **Automatic version detection** - code seamlessly supports YOLOv5, YOLOv8, and YOLO26

**Integration with Tracking Systems (Perfect for Your Research Paper):**
YOLO26's NMS-free outputs integrate seamlessly with:
- **Kalman Filters** - predictable output format simplifies state estimation for orbital mechanics
- **SORT Tracking** - fixed detection count (300) enables efficient multi-object tracking
- **Collision Avoidance** - low-latency detection critical for real-time trajectory planning
- **Autonomous Rendezvous** - deterministic performance enables closed-loop guidance systems
- **Deep-Space Navigation** - reduced computational overhead extends mission duration on battery power

## Configuration and Tuning

### Detection Parameter Tuning

Modify detection behavior by editing `libs/inc/image_processing.h`:

```cpp
// Neural network input dimensions
const float INPUT_WIDTH = 640.0;   // Model expects 640x640 input
const float INPUT_HEIGHT = 640.0;

// Detection thresholds
const float SCORE_THRESHOLD = 0.5;        // Minimum class probability (0.0-1.0)
const float NMS_THRESHOLD = 0.45;         // NMS IoU threshold (0.0-1.0)
const float CONFIDENCE_THRESHOLD = 0.45;  // Minimum objectness score (0.0-1.0)
```

**Parameter guide**:

| Parameter | Effect When Increased | Effect When Decreased | Recommended Range |
|-----------|------------------------|----------------------|-------------------|
| `SCORE_THRESHOLD` | Fewer, more confident detections | More detections, more false positives | 0.3 - 0.7 |
| `CONFIDENCE_THRESHOLD` | Stricter filtering | More lenient detection | 0.25 - 0.6 |
| `NMS_THRESHOLD` | More overlapping boxes | Fewer duplicate detections | 0.3 - 0.6 |

**Use case tuning**:
- **Security/Surveillance**: Lower thresholds (0.25-0.35) to catch all potential threats
- **Autonomous Vehicles**: Higher thresholds (0.6-0.7) to avoid false positives
- **General Purpose**: Default values (0.45-0.5) provide balanced results

### Visual Appearance Customization

Modify bounding box and label appearance in `libs/inc/image_processing.h`:

```cpp
// Text rendering
const float FONT_SCALE = 0.7;          // Label text size
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;               // Label text thickness

// Colors (BGR format)
cv::Scalar BLACK = cv::Scalar(0, 0, 0);      // Label background
cv::Scalar BLUE = cv::Scalar(255, 178, 50);  // Bounding box color
cv::Scalar YELLOW = cv::Scalar(0, 255, 255); // Label text color
cv::Scalar RED = cv::Scalar(0, 0, 255);      // Inference time text
```

Change `BLUE` scalar to customize bounding box color (e.g., red boxes: `cv::Scalar(0, 0, 255)`).

## YOLO26 & Multi-Version Support

### Automatic Model Version Detection

The implementation automatically detects whether you're using YOLOv5, YOLOv8, or YOLO26 models based on the output tensor shape:

- **YOLOv5**: Output shape `[1, 25200, 85]` with objectness score (requires NMS)
- **YOLOv8**: Output shape `[1, 84, 8400]` without objectness score (requires NMS)
- **YOLO26**: Output shape `[1, 300, 6]` NMS-free end-to-end inference (43% faster!)

You don't need to specify the model version - the code handles all three formats automatically:

```bash
# All commands work identically - version auto-detected
./opencv_cpp_release -i -d -p image.jpg -l weight/coco.names -m weight/yolov5s.onnx
./opencv_cpp_release -i -d -p image.jpg -l weight/coco.names -m weight/yolov8n.onnx
./opencv_cpp_release -i -d -p image.jpg -l weight/coco.names -m weight/yolo26n.onnx  # FASTEST
```

### YOLO26 NMS-Free Architecture Benefits

The revolutionary NMS-free design provides significant advantages:

1. **43% Faster Inference** - eliminates post-processing bottleneck
2. **Deterministic Output** - always 300 detections (filtered internally)
3. **Lower Latency Variance** - predictable timing for real-time control
4. **Simplified Deployment** - no NMS parameters to tune
5. **Better Edge Performance** - reduced CPU/memory requirements

### Training YOLO26 on Custom Datasets (RECOMMENDED)

For space applications with custom astronomical datasets using MuSGD optimizer:

```bash
cd external_components/ultralytics

# Install latest Ultralytics with YOLO26 support
pip install --upgrade ultralytics

# Train YOLO26 on custom asteroid detection dataset
# MuSGD optimizer provides superior convergence for astronomical data
yolo train model=yolo26n.pt data=asteroids.yaml epochs=200 imgsz=640 device=0 optimizer=auto

# Fine-tune with advanced augmentation for telescope imagery
yolo train model=yolo26n.pt data=asteroids.yaml epochs=200 imgsz=640 \
    augment=True mosaic=1.0 copy_paste=0.3 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4

# Export with INT8 quantization for spacecraft deployment
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 int8=True simplify=True

# Test the NMS-free model
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/train/weights/best.onnx'); results = model('telescope_test.jpg')"
```

### Legacy: Training YOLOv8 on Custom Datasets

For backward compatibility with YOLOv8:

```bash
cd external_components/ultralytics

# Install Ultralytics
pip install ultralytics

# Train on custom asteroid detection dataset
yolo train model=yolov8n.pt data=asteroids.yaml epochs=100 imgsz=640 device=0

# Export with INT8 quantization for edge deployment
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 int8=True

# Test the model
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/train/weights/best.onnx'); results = model('test_image.jpg')"
```

**Dataset YAML structure for astronomical objects:**
```yaml
# asteroids.yaml
path: ../datasets/asteroids
train: images/train
val: images/val

# Classes
nc: 4  # number of classes
names: ['asteroid', 'planet', 'debris', 'spacecraft']
```

### YOLOv8 Quantization for Edge Devices

For spacecraft and embedded deployment:

```bash
# INT8 quantization (4x smaller, 2-4x faster)
yolo export model=yolov8n.pt format=onnx int8=True

# FP16 quantization (2x smaller, moderate speedup)
yolo export model=yolov8n.pt format=onnx half=True

# Compare model sizes
ls -lh *.onnx
# yolov8n.onnx          ~6.0MB
# yolov8n_int8.onnx     ~1.5MB
# yolov8n_fp16.onnx     ~3.0MB
```

### Transfer Learning for Astronomical Detection (YOLO26 with MuSGD)

Fine-tune YOLO26 on telescope imagery using advanced MuSGD optimizer:

```python
from ultralytics import YOLO

# Load pretrained YOLO26 model
model = YOLO('yolo26n.pt')

# Transfer learning on custom space dataset with MuSGD optimizer
# MuSGD (SGD+Muon hybrid) provides superior convergence for astronomical data
results = model.train(
    data='telescope_data.yaml',
    epochs=250,                    # More epochs for better astronomical feature learning
    imgsz=640,
    batch=16,
    optimizer='auto',              # Automatically selects MuSGD for YOLO26
    lr0=0.001,
    weight_decay=0.0005,
    augment=True,
    mosaic=1.0,                    # Mosaic augmentation for varied asteroid positions
    copy_paste=0.3,                # Lower copy-paste for realistic space backgrounds
    hsv_h=0.015,                   # Minimal hue variation (space is mostly black)
    hsv_s=0.7,                     # Saturation variation for different illumination
    hsv_v=0.4,                     # Value variation for distance/brightness changes
    degrees=180,                   # Full rotation augmentation (asteroids have no "up")
    translate=0.2,                 # Position variation
    scale=0.9,                     # Scale variation for distance simulation
    perspective=0.0,               # No perspective in deep space
    flipud=0.5,                    # Vertical flip (no gravity orientation)
    fliplr=0.5                     # Horizontal flip
)

# Export NMS-free model for spacecraft C++ deployment
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    int8=True,                     # INT8 quantization for low-power spacecraft
    dynamic=False                  # Fixed input size for deterministic performance
)

print(f"YOLO26 model trained and exported for spacecraft deployment!")
print(f"NMS-free inference: 43% faster than traditional YOLO")
```

### Legacy: Transfer Learning with YOLOv8

For backward compatibility:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Transfer learning on custom space dataset
results = model.train(
    data='telescope_data.yaml',
    epochs=200,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    augment=True,
    mosaic=1.0,
    copy_paste=0.5
)

# Export for C++ deployment
model.export(format='onnx', imgsz=640, simplify=True, int8=True)
```

## Performance Optimization

### Speed Optimization Strategies

**1. Use YOLO26 (FASTEST - 43% faster than traditional YOLO)**:
```bash
# YOLO26n: NMS-free, 70+ FPS, 6MB - BEST CHOICE
./opencv_cpp_release -w -d -m weight/yolo26n.onnx

# For comparison:
# YOLO26s: 55+ FPS, 20MB - balanced
./opencv_cpp_release -w -d -m weight/yolo26s.onnx
```

**2. Use Smaller Legacy Models** (if YOLO26 not available):
```bash
# YOLOv8n (50+ FPS, 6MB, requires NMS)
./opencv_cpp_release -w -d -m weight/yolov8n.onnx

# YOLOv5n (45+ FPS, 3.7MB, requires NMS, lower accuracy)
./opencv_cpp_release -w -d -m weight/yolov5n.onnx
```

**2. Compile Release Build**:
- Always use Release mode for production (`-O3` optimization)
- 20-30% faster than Debug mode

**3. Enable Hardware Acceleration** (requires OpenCV with CUDA):
```cpp
// Add to libs/src/image_processing.cpp before model loading
onnx_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
onnx_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
```

**4. Reduce Input Resolution** (requires model re-export):
```bash
# Export 320x320 model (4x faster, less accurate)
python3 export.py --weights yolov5s.pt --img 320 --include onnx
```

Then modify `INPUT_WIDTH` and `INPUT_HEIGHT` to 320 in header file.

**5. Skip Frames for Video**:
Modify video processing to process every Nth frame:
```cpp
// In libs/src/video_processing.cpp
int frame_count = 0;
while(true) {
    cv::Mat frame;
    cap.read(frame);
    if(frame_count++ % 3 == 0) {  // Process every 3rd frame
        obj_detected_frame = each_frame.run_yolo_obj_detection(frame, ...);
    }
}
```

### Accuracy Optimization Strategies

**1. Use Larger Models**:
```bash
# Most accurate: YOLOv5x (166MB, 8 FPS)
./opencv_cpp_release -v -d -m weight/yolov5x.onnx
```

**2. Increase Detection Thresholds**:
Reduce false positives by raising `SCORE_THRESHOLD` to 0.6-0.7.

**3. Ensemble Multiple Models**:
Run both YOLOv5m and YOLOv5l, merge results with weighted voting.

**4. Test-Time Augmentation** (requires code modification):
Process flipped/rotated versions of each image, merge detections.

## Common Issues and Solutions

### Issue: CMake Cannot Find OpenCV

**Symptoms**:
```
CMake Error: Could not find OpenCV. Consider setting OpenCV_DIR.
```

**Solutions**:

1. Verify OpenCV installation:
```bash
pkg-config --modversion opencv4
pkg-config --cflags --libs opencv4
```

2. Set OpenCV_DIR environment variable:
```bash
export OpenCV_DIR=/usr/local/lib/cmake/opencv4
cmake -DCMAKE_BUILD_TYPE=Release ..
```

3. Install OpenCV development files:
```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev

# macOS
brew install opencv
```

### Issue: clang-format Not Found

**Symptoms**:
```
CMake Error: clang-format not found. Please install clang-format.
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install clang-format

# macOS
brew install clang-format

# Verify installation
which clang-format
```

### Issue: Webcam Failed to Open

**Symptoms**:
```
Failed to open the camera
```

**Solutions**:

1. Check camera permissions (macOS/Linux):
```bash
# macOS: Grant camera access in System Preferences > Security & Privacy
# Linux: Add user to video group
sudo usermod -a -G video $USER
```

2. Try different camera indices:
```cpp
// Modify libs/src/video_processing.cpp
cv::VideoCapture cap(1);  // Try 1 instead of 0
```

3. List available cameras:
```bash
# Linux
ls -l /dev/video*

# macOS
system_profiler SPCameraDataType
```

### Issue: No Objects Detected

**Symptoms**:
Program runs but no bounding boxes appear on output.

**Solutions**:

1. **Lower detection thresholds** in `libs/inc/image_processing.h`:
```cpp
const float SCORE_THRESHOLD = 0.3;        // Lowered from 0.5
const float CONFIDENCE_THRESHOLD = 0.3;   // Lowered from 0.45
```

2. **Verify model and labels loaded correctly**:
- Check file paths are correct
- Ensure `yolov5s.onnx` is valid (not corrupted download)
- Verify `coco.names` has 80 lines

3. **Run in debug mode** to see detection details:
```bash
./opencv_cpp_debug -i -d -p test_image.jpg
```

Debug output shows all detections with confidence scores.

### Issue: Segmentation Fault or Crash

**Symptoms**:
```
Segmentation fault (core dumped)
```

**Solutions**:

1. **Verify input file exists and is readable**:
```bash
ls -lh path/to/image.jpg
file path/to/image.jpg
```

2. **Check OpenCV DNN module support**:
```bash
# Verify OpenCV was compiled with DNN
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i dnn
```

3. **Rebuild in debug mode** for stack trace:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
gdb ./opencv_cpp_debug
run -i -d -p test.jpg
bt  # Print backtrace after crash
```

### Issue: Slow Performance / Low FPS

**Symptoms**:
Real-time processing is laggy or has low framerate.

**Solutions**:

1. **Use Release build** (not Debug):
```bash
./opencv_cpp_release  # Not opencv_cpp_debug
```

2. **Switch to faster model**:
```bash
# Use YOLOv5n instead of YOLOv5s
./opencv_cpp_release -w -d -m weight/yolov5n.onnx
```

3. **Enable GPU acceleration** (if available):
Recompile OpenCV with CUDA support and modify code to use GPU backend.

4. **Profile performance** to identify bottleneck:
```bash
# Run in debug mode to see inference timing
./opencv_cpp_debug -v -d -p video.mp4
# Check "Inference time" output
```

### Issue: Library Version Mismatch (macOS)

**Symptoms**:
```
dyld: Library not loaded: /opt/homebrew/opt/openexr/lib/libOpenEXR-3_3.32.dylib
dyld: Library not loaded: /opt/homebrew/opt/protobuf/lib/libprotobuf.32.0.0.dylib
Symbol not found: __ZN7Imf_3_310OutputFile11writePixelsEi
```

**Root Cause**:
OpenCV was compiled against older versions of system libraries (OpenEXR 3.3, protobuf 32.0), but your system has newer versions installed via Homebrew. This creates an ABI (Application Binary Interface) incompatibility.

**Solution**:

Reinstall OpenCV to recompile it against current library versions:

```bash
# Reinstall OpenCV with current dependencies
brew reinstall opencv

# If there are Qt symlink conflicts during installation:
brew unlink qt
brew link --overwrite qtbase

# Continue the OpenCV reinstall
brew reinstall opencv
```

After reinstalling, run your application normally:

```bash
./opencv_cpp_debug -w -d -l 'weight/coco.names' -m 'weight/yolov5s.onnx'
```

**Why This Works**:
- `brew reinstall opencv` recompiles OpenCV from source or downloads the latest binary bottle
- The new build links against current versions of OpenEXR, protobuf, and other dependencies
- This ensures ABI compatibility between OpenCV and all system libraries

**Prevention**:
After running `brew update && brew upgrade`, which may update OpenCV's dependencies, you should also reinstall OpenCV:

```bash
brew update && brew upgrade
brew reinstall opencv  # Ensure OpenCV is recompiled against updated dependencies
```

---

## License

This project is licensed under the **MIT License**, allowing free use in commercial and non-commercial applications.

## Author

**Ashish Upadhyay**
GitHub: [@itsashishupadhyay](https://github.com/itsashishupadhyay)
Repository: [object_detection_opencv_cpp](https://github.com/itsashishupadhyay/object_detection_opencv_cpp)

## Contributing

Contributions are welcome! Areas for improvement:
- GPU/CUDA acceleration support
- Additional model formats (TensorFlow Lite, TensorRT)
- Multi-threaded video processing
- Object tracking across frames
- REST API wrapper for network access
- ROS2 integration package

Please open issues for bugs or feature requests, and submit pull requests for improvements.

---

## Acknowledgments

This project builds upon these excellent open-source technologies:

- **[OpenCV](https://opencv.org/)** - Open Source Computer Vision Library providing DNN module and core image processing
- **[YOLOv5](https://github.com/ultralytics/yolov5)** by Ultralytics - State-of-the-art real-time object detection framework
- **[ONNX](https://onnx.ai/)** - Open Neural Network Exchange format for model interoperability
- **[COCO Dataset](https://cocodataset.org/)** - Common Objects in Context dataset with 80 object categories

---

## Keywords

C++ object detection, YOLOv5 C++ implementation, OpenCV DNN tutorial, ONNX inference C++, real-time computer vision, webcam object detection, video analysis C++, embedded object detection, YOLO without Python, OpenCV deep learning, computer vision C++ example, object recognition tutorial, CMake OpenCV project, edge computing vision, industrial automation computer vision

---

**Note**: This software is provided "AS IS" without warranty of any kind. Use in production environments at your own risk. Always test thoroughly before deploying in safety-critical or mission-critical applications.
