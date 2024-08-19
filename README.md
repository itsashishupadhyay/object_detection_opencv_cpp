# object_detection_opencv_cpp



## how to get the yolo onnx weight

go to `external_components/yolov5` and install req libs
`python -m pip install -r requirements.txt`
once installed run 
```
python export.py \
--weights yolov5s.pt \
--img 640 \
--simplify \
--optimize \
--include onnx
```
validate that the yolo5s.onnx file is working by running
```
python detect.py --weights yolov5s.onnx --dnn
```
if successful move the `yolov5s.onnx` to `/weight`

## compile and run
to create a binary `debug` and `release` run
```
# debug
cmake -DCMAKE_BUILD_TYPE=Debug ../CMakelists.txt && make all 

# release
cmake -DCMAKE_BUILD_TYPE=Release ../CMakelists.txt && make all 

```
Run using
```
./opencv_cpp_debug -o -l '~/object_detection_opencv_cpp/weight/coco.names' -m '~/object_detection_opencv_cpp/weight/yolov5s.onnx' -p '~/object_detection_opencv_cpp/external_components/yolov5/data/images/bus.jpg'
```

use the help menu for binary
```
./opencv_cpp_debug -h
```