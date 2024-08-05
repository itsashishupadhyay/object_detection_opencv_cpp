#ifndef DISPLAY_IMAGE_H
#define DISPLAY_IMAGE_H

#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

namespace DETECTION_IMAGE_PROCESSING {
class image_processing {
private:
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
  cv::Mat get_image_from_file(char **path2image);
  int display_image(cv::Mat &image, std::string displaymsg);
  cv::Mat image_greyscale(cv::Mat &image);
  cv::Mat blur_image(const cv::Mat &src, cv::Size kernelSize,
                     cv::Point anchorPoint);
  cv::Mat gaussian_blur_image(const cv::Mat &src, cv::Size ksize, double sigmaX,
                              double sigmaY);
  cv::Mat canny_edge_detector(const canny_config &config);
  int IMAGE_TEST_BLOCK(char **path2image);
};

} // namespace DETECTION_IMAGE_PROCESSING

#ifdef __cplusplus
}
#endif

#endif // DISPLAY_IMAGE_H