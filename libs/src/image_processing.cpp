#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>

namespace DETECTION_IMAGE_PROCESSING {

int image_processing::display_image(char **path2image) {

  cv::Mat image;
  image = cv::imread(std::string(*path2image), cv::IMREAD_COLOR);

  if (image.empty()) {
    printf("No image data \n");
    return -1;
  }
  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);

  cv::waitKey(0); // wait for a keystroke in the window

  return 0;
}

} // namespace DETECTION_IMAGE_PROCESSING
