# YOLO26 Export for OpenCV DNN

This directory contains a detailed Python script to export YOLO26 models in a format compatible with OpenCV's DNN module.

## Problem Statement

YOLO26 has two detection heads:

### 1. One-to-One Head (Default, end2end=True)
- **Output Shape**: `[batch, 300, 6]`
- **Format**: `[x1, y1, x2, y2, confidence, class_id]`
- **Advantages**: NMS-free, 43% faster inference
- **Issue**: ⚠️ **Not fully supported by OpenCV DNN** - Results in identical confidence scores for all detections

### 2. One-to-Many Head (end2end=False)
- **Output Shape**: `[batch, 84, 8400]`
- **Format**: Traditional YOLOv8/v11 format (4 bbox coords + 80 class scores)
- **Advantages**: ✓ **Fully compatible with OpenCV DNN module**
- **Requirement**: Needs NMS post-processing

## Solution

The `export_yolo26_opencv.py` script exports YOLO26 models with `end2end=False` to ensure full OpenCV DNN compatibility.

---

## Prerequisites

### 1. Install Dependencies

```bash
# Activate your virtual environment
source ../.yolo_venv/bin/activate

# Install required packages
pip install ultralytics onnx onnxslim
```

### 2. Download YOLO26 Model

You have two options:

#### Option A: Download Pretrained Model

```bash
# Download YOLO26 nano model (smallest, fastest)
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

# Or download other sizes:
# wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt  # Small
# wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt  # Medium
# wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt  # Large
# wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt  # XLarge
```

#### Option B: Auto-Download (Using Python)

```python
from ultralytics import YOLO

# This will automatically download the model if not present
model = YOLO('yolo26n.pt')
```

---

## Usage

### Basic Usage

```bash
# Export YOLO26 nano model for OpenCV
python3 export_yolo26_opencv.py --model yolo26n.pt

# Output: yolo26n_opencv.onnx
```

### Advanced Usage

```bash
# Specify custom output name
python3 export_yolo26_opencv.py --model yolo26n.pt --output custom_name.onnx

# Export with different image size
python3 export_yolo26_opencv.py --model yolo26s.pt --imgsz 640

# Export without model simplification
python3 export_yolo26_opencv.py --model yolo26l.pt --no-simplify

# Use different ONNX opset version
python3 export_yolo26_opencv.py --model yolo26m.pt --opset 17

# Quiet mode (minimal output)
python3 export_yolo26_opencv.py --model yolo26n.pt --quiet
```

### Help

```bash
python3 export_yolo26_opencv.py --help
```

---

## Model Sizes Reference

| Model    | Parameters | GFLOPs | Speed      | Accuracy | Use Case                    |
|----------|------------|--------|------------|----------|-----------------------------|
| yolo26n  | 2.6M       | 6.1    | Fastest    | Good     | Edge devices, real-time     |
| yolo26s  | 10.0M      | 22.8   | Fast       | Better   | Mobile devices              |
| yolo26m  | 21.9M      | 75.4   | Medium     | Great    | Desktop CPUs                |
| yolo26l  | 26.3M      | 93.8   | Slower     | Excellent| GPUs, servers               |
| yolo26x  | 59.0M      | 209.5  | Slowest    | Best     | Maximum accuracy needed     |

---

## Output Details

### What the Script Produces

```
Input:  yolo26n.pt
Output: yolo26n_opencv.onnx (or custom name)

Model Format:
  - Output shape: [1, 84, 8400]
  - 84 channels = 4 bbox coordinates + 80 class scores
  - 8400 anchors across multiple scales
  - Compatible with YOLOv8 post-processing code
```

### Verification

The script automatically verifies the exported model and displays:

```
✓ ONNX model is valid
✓ Output shape: [1, 84, 8400]
✓ Format confirmed: One-to-Many Head (OpenCV compatible)
```

---

## Using Exported Model in C++

### Load the Model

```cpp
#include <opencv2/dnn.hpp>

// Load the OpenCV-compatible YOLO26 model
cv::dnn::Net net = cv::dnn::readNetFromONNX("yolo26n_opencv.onnx");
```

### Post-Processing

**Important**: Use YOLOv8 post-processing, NOT YOLO26 one-to-one processing!

The exported model outputs traditional YOLO format:
- Shape: `[1, 84, 8400]`
- Layout: 4 bbox coords + 80 class scores per anchor
- Requires NMS post-processing

Your existing `post_process_yolov8()` function will work correctly!

---

## Example Workflow

### Complete Export and Test Workflow

```bash
# 1. Activate virtual environment
cd /Users/upadhyay/dev/object_detection_opencv_cpp/supporting_scripts
source ../.yolo_venv/bin/activate

# 2. Download model (if needed)
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

# 3. Export for OpenCV
python3 export_yolo26_opencv.py --model yolo26n.pt --output yolo26n_opencv.onnx

# 4. Copy to weight directory
cp yolo26n_opencv.onnx ../weight/

# 5. Test with C++
cd ../build
./opencv_cpp_debug -i -d -p '../test_images/polaroid_perspective.jpg' \
                    -m '../weight/yolo26n_opencv.onnx'
```

---

## Troubleshooting

### Issue: "Model file not found"

**Solution**: Ensure the .pt model file exists in the current directory or provide full path:

```bash
python3 export_yolo26_opencv.py --model /path/to/yolo26n.pt
```

### Issue: "Output shape [1, 300, 6]"

**Solution**: This means end2end=True was used. The script should prevent this, but if it happens:

```bash
# Force export with end2end=False by checking the model
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.export(format='onnx', imgsz=640, simplify=True, end2end=False)
"
```

### Issue: OpenCV DNN shows identical confidence scores

**Solution**: You're using the wrong model format. Re-export with this script!

---

## Technical Details

### Why This Matters

| Aspect               | One-to-One (end2end=True) | One-to-Many (end2end=False) |
|----------------------|---------------------------|------------------------------|
| Output Shape         | [1, 300, 6]               | [1, 84, 8400]                |
| NMS Required         | No (built-in)             | Yes (external)               |
| OpenCV DNN Support   | ⚠️ Partial                | ✅ Full                       |
| Speed                | 43% faster                | Standard                     |
| Post-processing      | Simple                    | Traditional YOLO             |

### Detection Format

**One-to-One (problematic with OpenCV DNN)**:
```
Output: [batch, max_detections=300, data=6]
Data:   [x1, y1, x2, y2, confidence, class_id]
```

**One-to-Many (OpenCV DNN compatible)**:
```
Output: [batch, channels=84, anchors=8400]
Data:   [cx, cy, w, h, class_0_score, ..., class_79_score] per anchor
```

---

## References

- **YOLO26 Documentation**: https://docs.ultralytics.com/models/yolo26
- **Ultralytics GitHub**: https://github.com/ultralytics/ultralytics
- **OpenCV DNN Module**: https://docs.opencv.org/master/d6/d0f/group__dnn.html

---

## Script Features

### Comprehensive Features

✅ Detailed progress output with 3-step process
✅ Automatic output shape verification
✅ OpenCV compatibility checking
✅ Error handling and validation
✅ Support for all YOLO26 model sizes
✅ Configurable image size, opset, simplification
✅ Command-line interface with help
✅ Quiet mode for automation

### Safety Features

- Validates input model exists
- Checks ONNX model integrity
- Verifies output format is correct
- Warns if wrong format detected
- Provides clear error messages

---

## Support

For issues related to:
- **Script**: Check this README
- **YOLO26 models**: https://github.com/ultralytics/ultralytics/issues
- **OpenCV DNN**: https://github.com/opencv/opencv/issues

---

**Last Updated**: 2026-02-16
**Script Version**: 2.0
