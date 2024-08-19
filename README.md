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