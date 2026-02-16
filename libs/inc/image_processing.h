#ifndef DISPLAY_IMAGE_H
#define DISPLAY_IMAGE_H

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

namespace DETECTION_IMAGE_PROCESSING {
class image_processing {
private:
  const float INPUT_WIDTH = 640.0;
  const float INPUT_HEIGHT = 640.0;
  const float SCORE_THRESHOLD = 0.5;
  const float NMS_THRESHOLD = 0.45;
  const float CONFIDENCE_THRESHOLD = 0.45;

  // Text parameters.
  const float FONT_SCALE = 0.7;
  const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
  const int THICKNESS = 1;

  // Colors.
  cv::Scalar BLACK = cv::Scalar(0, 0, 0);
  cv::Scalar BLUE = cv::Scalar(255, 178, 50);
  cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
  cv::Scalar RED = cv::Scalar(0, 0, 255);

  std::vector<std::string> class_list;
  cv::dnn::Net onnx_net;
  bool is_yolov8_model;
  bool is_yolo26_model;

  void draw_label(cv::Mat &input_image, std::string label, int left, int top);
  std::vector<cv::Mat> pre_process_yolo(cv::Mat &input_image,
                                        cv::dnn::Net &net);

  cv::Mat post_process_yolo(cv::Mat &input_image, std::vector<cv::Mat> &outputs,
                            const std::vector<std::string> &class_name);

  cv::Mat post_process_yolov8(cv::Mat &input_image,
                              std::vector<cv::Mat> &outputs,
                              const std::vector<std::string> &class_name);

  cv::Mat post_process_yolo26(cv::Mat &input_image,
                              std::vector<cv::Mat> &outputs,
                              const std::vector<std::string> &class_name);

  int detect_model_version(const std::vector<cv::Mat> &outputs);

public:
  cv::Mat get_image_from_file(std::string path2image);
  int display_image(cv::Mat &image, std::string displaymsg,
                    std::string put_text_on_image);
  cv::Mat run_yolo_obj_detection(cv::Mat &frame, std::string path2lables,
                                 std::string path2yolo_onnx);

  int IMAGE_TEST_BLOCK(std::string path2image);
  int detect_objects_in_image(std::string path2image,
                              std::string object_labes_path,
                              std::string onnx_file_path);
};

} // namespace DETECTION_IMAGE_PROCESSING

#ifdef __cplusplus
}
#endif

#endif // DISPLAY_IMAGE_H