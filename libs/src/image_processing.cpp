#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>

namespace DETECTION_IMAGE_PROCESSING {

cv::Mat image_processing::get_image(char **path2image, bool greyscale = false) {
  cv::Mat image;
  image = cv::imread(std::string(*path2image), cv::IMREAD_COLOR);

  if (image.empty()) {
    printf("No image data \n");
  }

  if (greyscale) {
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
  }

  return image;
}

int image_processing::display_image(char **path2image) {
  cv::Mat image;
  image = get_image(path2image);

  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);

  cv::waitKey(0); // wait for a keystroke in the window

  return 0;
}

int image_processing::display_image_greyscale(char **path2image) {
  cv::Mat grey_image;
  grey_image = get_image(path2image, true);

  cv::namedWindow("Display Grey Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Grey Image", grey_image);

  cv::waitKey(0); // wait for a keystroke in the window

  return 0;
}

} // namespace DETECTION_IMAGE_PROCESSING
