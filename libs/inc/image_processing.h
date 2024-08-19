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

  void draw_label(cv::Mat &input_image, std::string label, int left, int top);
  std::vector<cv::Mat> pre_process_yolo(cv::Mat &input_image,
                                        cv::dnn::Net &net);

  cv::Mat post_process_yolo(cv::Mat &input_image, std::vector<cv::Mat> &outputs,
                            const std::vector<std::string> &class_name);

public:
  typedef struct canny_config {
    cv::Mat image; // Change from cv::Mat& to cv::Mat
    double threshold1;
    double threshold2;
    int apertureSize;
    bool L2gradient;
    bool dilate;
    bool erode;
    cv::Mat kernel;
    cv::Point anchor;
    int iterations;
    int borderType;
    cv::Scalar borderValue;

    // Constructor with default values
    canny_config(cv::Mat img = cv::Mat(), double thresh1 = 0,
                 double thresh2 = 0, int aperture = 3, bool l2grad = false,
                 bool dil = false, bool er = false, cv::Mat kern = cv::Mat(),
                 cv::Point anch = cv::Point(-1, -1), int iters = 1,
                 int border = cv::BORDER_REFLECT_101,
                 cv::Scalar borderVal = cv::morphologyDefaultBorderValue())
        : image(img), threshold1(thresh1), threshold2(thresh2),
          apertureSize(aperture), L2gradient(l2grad), dilate(dil), erode(er),
          kernel(kern), anchor(anch), iterations(iters), borderType(border),
          borderValue(borderVal) {}
  } canny_config;

  int display_image(cv::Mat image);
  cv::Mat get_image_from_file(std::string path2image);
  int display_image(cv::Mat &image, std::string displaymsg,
                    std::string put_text_on_image);
  cv::Mat image_greyscale(cv::Mat &image);
  cv::Mat blur_image(const cv::Mat &src, cv::Size kernelSize,
                     cv::Point anchorPoint);
  cv::Mat gaussian_blur_image(const cv::Mat &src, cv::Size ksize, double sigmaX,
                              double sigmaY);
  cv::Mat canny_edge_detector(const canny_config &config);
  cv::Mat get_top_perspective(cv::Mat &image,
                              std::vector<cv::Point2f> src_points,
                              std::vector<cv::Point2f> dst_points);
  cv::Mat run_yolo_obj_detection(cv::Mat &image, std::string path2lables,
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