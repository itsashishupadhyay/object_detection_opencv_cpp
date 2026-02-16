#!/bin/bash

################################################################################
# YOLO26 Quick Export Script for OpenCV DNN
#
# This script automates:
#   1. Downloading YOLO26 model (if needed)
#   2. Exporting to OpenCV-compatible format
#   3. Moving to weight directory
#   4. Testing with your C++ application
#
# Usage:
#   ./quick_export_yolo26.sh [model_size]
#
# Examples:
#   ./quick_export_yolo26.sh n     # Export nano model (default)
#   ./quick_export_yolo26.sh s     # Export small model
#   ./quick_export_yolo26.sh m     # Export medium model
#   ./quick_export_yolo26.sh l     # Export large model
#   ./quick_export_yolo26.sh x     # Export xlarge model
# Date: 2026-02-16
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.yolo_venv"
WEIGHT_DIR="$PROJECT_ROOT/weight"
TEST_IMAGE="$PROJECT_ROOT/test_images/polaroid_perspective.jpg"
BUILD_DIR="$PROJECT_ROOT/build"

# Default model size
MODEL_SIZE="${1:-n}"

# Validate model size
case "$MODEL_SIZE" in
    n|s|m|l|x)
        ;;
    *)
        echo -e "${RED}Error: Invalid model size '$MODEL_SIZE'${NC}"
        echo "Valid sizes: n (nano), s (small), m (medium), l (large), x (xlarge)"
        exit 1
        ;;
esac

MODEL_NAME="yolo26${MODEL_SIZE}"
MODEL_PT="${MODEL_NAME}.pt"
MODEL_ONNX="${MODEL_NAME}_opencv.onnx"
DOWNLOAD_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/${MODEL_PT}"

# Model specifications
declare -A MODEL_PARAMS=(
    ["n"]="2.6M params, 6.1 GFLOPs - Nano (fastest)"
    ["s"]="10.0M params, 22.8 GFLOPs - Small"
    ["m"]="21.9M params, 75.4 GFLOPs - Medium"
    ["l"]="26.3M params, 93.8 GFLOPs - Large"
    ["x"]="59.0M params, 209.5 GFLOPs - XLarge (most accurate)"
)

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  YOLO26 Quick Export for OpenCV DNN                            ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
}

print_step() {
    echo -e "\n${GREEN}▶ $1${NC}"
}

print_info() {
    echo -e "${BLUE}  ℹ $1${NC}"
}

print_success() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}  ✗ $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "Required command '$1' not found"
        exit 1
    fi
}

################################################################################
# Main Process
################################################################################

main() {
    print_header

    echo -e "\n${BLUE}Model:${NC} YOLO26-${MODEL_SIZE} (${MODEL_PARAMS[$MODEL_SIZE]})"
    echo -e "${BLUE}Output:${NC} $MODEL_ONNX (OpenCV DNN compatible)"
    echo ""

    # Step 1: Check prerequisites
    print_step "Step 1/6: Checking prerequisites"

    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found at: $VENV_PATH"
        print_info "Please create it first: python3 -m venv $VENV_PATH"
        exit 1
    fi
    print_success "Virtual environment found"

    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    print_success "Virtual environment activated"

    # Check for Python packages
    if ! python3 -c "import ultralytics" 2>/dev/null; then
        print_warning "Ultralytics not installed, installing now..."
        pip install -q ultralytics onnx onnxslim
        print_success "Dependencies installed"
    else
        print_success "Dependencies already installed"
    fi

    # Step 2: Download or verify model
    print_step "Step 2/6: Obtaining YOLO26 model"

    cd "$SCRIPT_DIR"

    if [ -f "$MODEL_PT" ]; then
        print_info "Model already exists: $MODEL_PT"
        print_success "Using existing model"
    else
        print_info "Downloading from: $DOWNLOAD_URL"

        if command -v wget &> /dev/null; then
            wget -q --show-progress "$DOWNLOAD_URL"
        elif command -v curl &> /dev/null; then
            curl -L -o "$MODEL_PT" "$DOWNLOAD_URL"
        else
            print_warning "wget/curl not found, downloading via Python..."
            python3 << EOF
from ultralytics import YOLO
import sys
try:
    model = YOLO('${MODEL_PT}')
    print('Model downloaded successfully')
except Exception as e:
    print(f'Error downloading model: {e}', file=sys.stderr)
    sys.exit(1)
EOF
        fi

        if [ -f "$MODEL_PT" ]; then
            print_success "Model downloaded: $MODEL_PT ($(du -h "$MODEL_PT" | cut -f1))"
        else
            print_error "Failed to download model"
            exit 1
        fi
    fi

    # Step 3: Export model
    print_step "Step 3/6: Exporting to OpenCV-compatible format"
    print_info "This may take 10-30 seconds..."

    python3 export_yolo26_opencv.py --model "$MODEL_PT" --output "$MODEL_ONNX" --quiet

    if [ -f "$MODEL_ONNX" ]; then
        print_success "Export complete: $MODEL_ONNX ($(du -h "$MODEL_ONNX" | cut -f1))"
    else
        print_error "Export failed"
        exit 1
    fi

    # Step 4: Verify model format
    print_step "Step 4/6: Verifying model format"

    python3 << EOF
import onnx
import sys

try:
    model = onnx.load('${MODEL_ONNX}')
    output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]

    if len(output_shape) == 3 and output_shape[1] == 84:
        print('  ✓ Format: One-to-Many Head [1, 84, 8400]')
        print('  ✓ OpenCV DNN: Compatible')
        sys.exit(0)
    else:
        print(f'  ✗ Unexpected format: {output_shape}')
        sys.exit(1)
except Exception as e:
    print(f'  ✗ Verification failed: {e}')
    sys.exit(1)
EOF

    if [ $? -ne 0 ]; then
        print_error "Model format verification failed"
        exit 1
    fi

    # Step 5: Copy to weight directory
    print_step "Step 5/6: Installing model to weight directory"

    mkdir -p "$WEIGHT_DIR"
    cp "$MODEL_ONNX" "$WEIGHT_DIR/"

    print_success "Model installed: $WEIGHT_DIR/$MODEL_ONNX"

    # Step 6: Test with C++ application (optional)
    print_step "Step 6/6: Testing with C++ application"

    if [ ! -f "$BUILD_DIR/opencv_cpp_debug" ]; then
        print_warning "C++ application not built, skipping test"
        print_info "Build with: cd $BUILD_DIR && make"
    elif [ ! -f "$TEST_IMAGE" ]; then
        print_warning "Test image not found, skipping test"
    else
        print_info "Running inference on test image..."

        cd "$BUILD_DIR"
        OUTPUT=$(./opencv_cpp_debug -p "$TEST_IMAGE" -m "../weight/$MODEL_ONNX" 2>&1 | tail -5)

        if echo "$OUTPUT" | grep -q "Detected:"; then
            print_success "C++ inference successful!"
            echo "$OUTPUT" | grep "Detected:" | head -3
        else
            print_warning "Test completed (check output manually)"
        fi
    fi

    # Summary
    echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${GREEN}✓ Export Complete!${NC}                                              ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"

    echo -e "\n${GREEN}Model Ready:${NC}"
    echo -e "  Location: ${WEIGHT_DIR}/${MODEL_ONNX}"
    echo -e "  Format:   One-to-Many Head (OpenCV DNN compatible)"
    echo -e "  Size:     $(du -h "${WEIGHT_DIR}/${MODEL_ONNX}" | cut -f1)"

    echo -e "\n${GREEN}Usage in C++:${NC}"
    echo -e "  cv::dnn::Net net = cv::dnn::readNetFromONNX(\"${MODEL_ONNX}\");"

    echo -e "\n${GREEN}Test Command:${NC}"
    echo -e "  cd ${BUILD_DIR}"
    echo -e "  ./opencv_cpp_debug -i -d -p 'test_image.jpg' -m '../weight/${MODEL_ONNX}'"

    echo -e "\n${BLUE}Note:${NC} Use YOLOv8 post-processing, not YOLO26 one-to-one format"
    echo ""
}

# Trap errors
trap 'print_error "Script failed at line $LINENO"; exit 1' ERR

# Run main function
main "$@"
