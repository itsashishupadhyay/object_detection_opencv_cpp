# YOLO26 Export Tools - Quick Start Guide

## 📁 Files Created

### 1. `export_yolo26_opencv.py` - Detailed Python Export Script
   - **Purpose**: Export YOLO26 models for OpenCV DNN compatibility
   - **Features**:
     - Comprehensive error checking
     - Detailed progress output
     - Model format verification
     - Support for all YOLO26 sizes
     - Command-line arguments
   - **Lines of Code**: ~350 (well-documented)

### 2. `quick_export_yolo26.sh` - One-Command Automation
   - **Purpose**: Automated end-to-end export workflow
   - **Features**:
     - Auto-downloads models
     - Exports to OpenCV format
     - Verifies output
     - Tests with C++ app
     - Colored terminal output

### 3. `README_YOLO26_EXPORT.md` - Complete Documentation
   - **Purpose**: Comprehensive reference guide
   - **Contents**:
     - Problem explanation
     - Step-by-step tutorials
     - Troubleshooting guide
     - Technical details
     - Model specifications

---

## 🚀 Quick Start (Choose One)

### Option 1: Automated Script (Easiest)

```bash
cd supporting_scripts

# Export nano model (default)
./quick_export_yolo26.sh

# Or export different sizes:
./quick_export_yolo26.sh s    # Small
./quick_export_yolo26.sh m    # Medium
./quick_export_yolo26.sh l    # Large
./quick_export_yolo26.sh x    # XLarge
```

**This will automatically:**
1. ✅ Check dependencies
2. ✅ Download YOLO26 model
3. ✅ Export to OpenCV format
4. ✅ Verify output
5. ✅ Install to weight directory
6. ✅ Test with your C++ app

---

### Option 2: Manual Python Script

```bash
cd supporting_scripts

# Download model first
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

# Export with detailed output
python3 export_yolo26_opencv.py --model yolo26n.pt

# Copy to weight directory
cp yolo26n_opencv.onnx ../weight/
```

---

## 📖 Usage Examples

### Python Script Examples

```bash
# Basic export
python3 export_yolo26_opencv.py --model yolo26n.pt

# Custom output name
python3 export_yolo26_opencv.py --model yolo26n.pt --output custom.onnx

# Different image size
python3 export_yolo26_opencv.py --model yolo26s.pt --imgsz 640

# Quiet mode
python3 export_yolo26_opencv.py --model yolo26l.pt --quiet

# Without simplification
python3 export_yolo26_opencv.py --model yolo26m.pt --no-simplify

# Get help
python3 export_yolo26_opencv.py --help
```

### Shell Script Examples

```bash
# Export different models
./quick_export_yolo26.sh n    # Nano (2.6M params, fastest)
./quick_export_yolo26.sh s    # Small (10M params)
./quick_export_yolo26.sh m    # Medium (22M params)
./quick_export_yolo26.sh l    # Large (26M params)
./quick_export_yolo26.sh x    # XLarge (59M params, most accurate)
```

---

## 🎯 What Gets Exported?

### Input Model (PyTorch)
```
File: yolo26n.pt
Format: PyTorch checkpoint
Heads: Both one-to-one and one-to-many
```

### Output Model (ONNX)
```
File: yolo26n_opencv.onnx
Format: ONNX (OpenVINO/TensorRT compatible)
Output Shape: [1, 84, 8400]
Layout: 4 bbox + 80 classes per anchor
Compatible: ✅ OpenCV DNN, ONNXRuntime, TensorRT
```

---

## ⚙️ Model Specifications

| Model    | Parameters | GFLOPs | File Size | Speed      | Accuracy | Best For                |
|----------|------------|--------|-----------|------------|----------|-------------------------|
| yolo26n  | 2.6M       | 6.1    | ~5 MB     | Fastest    | Good     | Edge devices, RPi       |
| yolo26s  | 10.0M      | 22.8   | ~20 MB    | Fast       | Better   | Mobile, embedded        |
| yolo26m  | 21.9M      | 75.4   | ~44 MB    | Medium     | Great    | Desktop CPU             |
| yolo26l  | 26.3M      | 93.8   | ~53 MB    | Slower     | Excellent| GPU inference           |
| yolo26x  | 59.0M      | 209.5  | ~118 MB   | Slowest    | Best     | Servers, max accuracy   |

---

## 🔧 Using Exported Model in C++

### Load Model

```cpp
#include <opencv2/dnn.hpp>

// Load exported model
cv::dnn::Net net = cv::dnn::readNetFromONNX("../weight/yolo26n_opencv.onnx");

// Set backend (optional but recommended)
net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
```

### Run Inference

```cpp
// Your existing pre-processing works!
std::vector<cv::Mat> outputs = pre_process_yolo(image, net);

// Use YOLOv8 post-processing (NOT YOLO26 one-to-one!)
cv::Mat result = post_process_yolov8(image, outputs, class_names);
```

### Test Command

```bash
cd build
./opencv_cpp_debug -i -d \
    -p '../test_images/polaroid_perspective.jpg' \
    -m '../weight/yolo26n_opencv.onnx'
```

---

## 🐛 Troubleshooting

### Problem: "Model file not found"

**Solution**: Ensure model is in current directory or provide full path
```bash
python3 export_yolo26_opencv.py --model /full/path/to/yolo26n.pt
```

### Problem: "All detections have identical confidence"

**Cause**: Using wrong model format (one-to-one head)

**Solution**: Re-export with this script!
```bash
./quick_export_yolo26.sh
```

### Problem: "ultralytics module not found"

**Solution**: Activate virtual environment and install
```bash
source ../.yolo_venv/bin/activate
pip install ultralytics onnx onnxslim
```

### Problem: "Output shape [1, 300, 6]"

**Cause**: Model exported with end2end=True

**Solution**: This script forces end2end=False. If still happening:
```python
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.export(format='onnx', imgsz=640, simplify=True, end2end=False)
```

---

## 📊 Expected Results

### Python Inference (Reference)
```
Using: ONNXRuntime
Detections: 8 persons
Confidence: 0.82, 0.72, 0.66, 0.56, 0.54, 0.52, 0.51, 0.50
```

### C++ Inference (Your Code)
```
Using: OpenCV DNN + exported model
Detections: Should match Python (±1)
Confidence: Should vary (NOT all identical)
```

**If C++ shows identical confidence → Wrong model format!**

---

## 🎓 Understanding the Issue

### Why This Script Exists

YOLO26 introduced **dual-head architecture**:

1. **One-to-One Head** (Default in YOLO26):
   - Output: `[1, 300, 6]`
   - Format: `[x1, y1, x2, y2, conf, class]`
   - Benefit: NMS-free, 43% faster
   - **Problem**: OpenCV DNN doesn't fully support this format
     - Results in **identical confidence scores**
     - **Wrong number of detections**

2. **One-to-Many Head** (Traditional YOLO):
   - Output: `[1, 84, 8400]`
   - Format: `[cx, cy, w, h, class_0, ..., class_79]`
   - Benefit: ✅ **Full OpenCV DNN support**
   - Trade-off: Requires NMS (but you already have this!)

### This Script Forces One-to-Many Export

```python
# The critical line in export_yolo26_opencv.py:
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    end2end=False  # ← This forces one-to-many head!
)
```

---

## 📚 Additional Resources

### Files in This Directory

1. **`export_yolo26_opencv.py`** - Main Python export tool
2. **`quick_export_yolo26.sh`** - Automated bash script
3. **`README_YOLO26_EXPORT.md`** - Detailed documentation
4. **`USAGE_GUIDE.md`** - This file (quick reference)

### External Documentation

- **YOLO26 Official**: https://docs.ultralytics.com/models/yolo26
- **Ultralytics GitHub**: https://github.com/ultralytics/ultralytics
- **OpenCV DNN**: https://docs.opencv.org/master/d6/d0f/group__dnn.html
- **ONNX Format**: https://onnx.ai/

---

## ✅ Verification Checklist

After export, verify:

- [ ] Output file exists: `yolo26n_opencv.onnx`
- [ ] Output shape is `[1, 84, 8400]` (not `[1, 300, 6]`)
- [ ] Model loads in OpenCV: `cv::dnn::readNetFromONNX()`
- [ ] C++ detections have **varying confidence scores**
- [ ] Detection count matches Python inference (±1)
- [ ] Bounding boxes look correct

---

## 🎉 Success Indicators

### You'll know it works when:

✅ Script outputs: "Format confirmed: One-to-Many Head (OpenCV compatible)"
✅ C++ shows varying confidence scores (e.g., 0.82, 0.71, 0.65...)
✅ Detection count is reasonable (e.g., 5-10 for test image)
✅ Bounding boxes are accurate
✅ No warnings about identical confidence

---

## 📞 Support

### If You're Stuck

1. **Read the error message carefully** - They're detailed!
2. **Check `README_YOLO26_EXPORT.md`** - Comprehensive troubleshooting
3. **Verify Python inference first** - Ensure model works
4. **Check model format** - Must be `[1, 84, 8400]`
5. **Use verbose mode** - `python3 export_yolo26_opencv.py --model yolo26n.pt`

### Common Mistakes

❌ Using original `yolo26n.onnx` (one-to-one format)
❌ Not activating virtual environment
❌ Using YOLO26 post-processing instead of YOLOv8
❌ Wrong confidence/NMS thresholds

---

**Last Updated**: 2026-02-16
**Version**: 2.0
**Author**: Claude Code

**Status**: ✅ Fully Tested & Working
