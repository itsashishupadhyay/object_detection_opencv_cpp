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
  int display_image(cv::Mat image);
  cv::Mat get_image_from_file(char **path2image);
  int display_image(cv::Mat &image, std::string displaymsg);
  cv::Mat image_greyscale(cv::Mat &image);
  cv::Mat blur_image(const cv::Mat &src, cv::Size kernelSize,
                     cv::Point anchorPoint);
  cv::Mat gaussian_blur_image(const cv::Mat &src, cv::Size ksize, double sigmaX,
                              double sigmaY);
  int IMAGE_TEST_BLOCK(char **path2image);
};

} // namespace DETECTION_IMAGE_PROCESSING

#ifdef __cplusplus
}
#endif

#endif // DISPLAY_IMAGE_H