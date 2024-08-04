#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>

namespace DETECTION_IMAGE_PROCESSING {

cv::Mat image_processing::get_image_from_file(char **path2image) {
  cv::Mat image;
  image = cv::imread(std::string(*path2image), cv::IMREAD_COLOR);

  if (image.empty()) {
    printf("No image data \n");
  }

  return image;
}

int image_processing::display_image(cv::Mat &image,
                                    std::string displaymsg = "") {

  std::string window_name;
  if (displaymsg.empty()) {
    window_name = "Display Image";
  } else {
    window_name = displaymsg;
  }

  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::imshow(window_name, image);
  cv::waitKey(0); // wait for a keystroke in the window

  return 0;
}

cv::Mat image_processing::image_greyscale(cv::Mat &image) {

  cv::Mat grey_image;
  cv::cvtColor(image, grey_image, cv::COLOR_BGR2GRAY);

  return grey_image;
}

/*
  * Function to blur an image
  * @param src: Source image
  * @param kernelSize: Size of the kernel to be used for blurring
  * @param anchorPoint: Anchor point of the kernel. Default is (-1, -1) which
  * means the anchor is at the kernel center
  * @return Blurred image
  *
  *
  NOTE: Size Parameter:
  The Size parameter in the blur() function specifies the size of the kernel
  (filter) to be used for smoothing. It is defined as Size(w, h), where:

      w is the width of the kernel in pixels (must be a positive odd integer)
      h is the height of the kernel in pixels (must be a positive odd integer)

  The kernel size determines the number of neighboring pixels to consider when
  calculating the average value for each pixel in the output image. A larger
  kernel size will result in more smoothing, while a smaller kernel size will
  result in less smoothing. Anchor Point Parameter: The anchor point parameter
  in the blur() function specifies the location of the anchor pixel (the pixel
  being evaluated) relative to the kernel. It is defined as Point(x, y), where:

      x is the x-coordinate of the anchor point (can be negative or positive)
      y is the y-coordinate of the anchor point (can be negative or positive)

  If the anchor point is set to Point(-1, -1), it means that the center of the
  kernel is aligned with the pixel being evaluated. This is the default behavior
  and is commonly used. If the anchor point is set to a positive value, it means
  that the kernel is shifted to the right (for x) or down (for y) by the
  specified number of pixels.

*/
cv::Mat image_processing::blur_image(const cv::Mat &src, cv::Size kernelSize,
                                     cv::Point anchorPoint = cv::Point(-1,
                                                                       -1)) {
  // Ensure kernel size is positive odd integers
  if (kernelSize.width <= 0 || kernelSize.height <= 0 ||
      kernelSize.width % 2 == 0 || kernelSize.height % 2 == 0) {
    kernelSize.width = std::max(1, kernelSize.width - kernelSize.width % 2 + 1);
    kernelSize.height =
        std::max(1, kernelSize.height - kernelSize.height % 2 + 1);
    std::cout << "Warning: Kernel size must be positive odd integers. Updating "
                 "to nearest positive odd values. "
              << "width: " << kernelSize.width
              << " height: " << kernelSize.height << std::endl;
  }

  // Ensure anchor point is valid
  if (anchorPoint.x < -1 || anchorPoint.y < -1) {
    std::cout << "Warning: Invalid anchor point. Defaulting to Point(-1, -1)."
              << std::endl;
    anchorPoint = cv::Point(-1, -1);
  }

  // Apply blur
  cv::Mat blurred_image;
  cv::blur(src, blurred_image, kernelSize, anchorPoint);

  return blurred_image;
}

/*
  * Function to apply Gaussian blur to an image
  * @param src: Source image
  * @param ksize: Size of the kernel to be used for blurring
  * @param sigmaX: Standard deviation of the Gaussian kernel in X direction
  * @param sigmaY: Standard deviation of the Gaussian kernel in Y direction
  * @return Gaussian blurred image
  *
  *
  NOTE: What do SigmaX and SigmaY represent?

    sigmaX: The standard deviation of the Gaussian kernel in the X direction. A
higher value of sigmaX means that the kernel will be more spread out in the X
direction, resulting in more blur in that direction. sigmaY: The standard
deviation of the Gaussian kernel in the Y direction. A higher value of sigmaY
means that the kernel will be more spread out in the Y direction, resulting in
more blur in that direction.

Effects of SigmaX and SigmaY on Gaussian Blur
Here are some key effects of varying sigmaX and sigmaY:

    Isotropic Blur: When sigmaX = sigmaY, the blur is isotropic, meaning it is
equal in all directions. This is the default behavior when sigmaX and sigmaY are
both 0. Anisotropic Blur: When sigmaX ≠ sigmaY, the blur is anisotropic, meaning
it is different in different directions. This can be useful for creating motion
blur or simulating camera shake. Increased Blur: As sigmaX and sigmaY increase,
the blur becomes more pronounced, and the image becomes more smoothed.
    Directional Blur: By setting sigmaX > sigmaY or vice versa, you can create
directional blur, where the blur is more pronounced in one direction than the
other.


*/

cv::Mat image_processing::gaussian_blur_image(const cv::Mat &src,
                                              cv::Size ksize, double sigmaX = 0,
                                              double sigmaY = 0) {
  // Ensure ksize is positive and odd
  if (ksize.width <= 0 || ksize.height <= 0 || ksize.width % 2 == 0 ||
      ksize.height % 2 == 0) {
    ksize.width = std::max(1, ksize.width - ksize.width % 2 + 1);
    ksize.height = std::max(1, ksize.height - ksize.height % 2 + 1);
    std::cout << "Warning: Kernel size must be positive and odd. Updating to "
                 "nearest positive odd values. "
              << "width: " << ksize.width << " height: " << ksize.height
              << std::endl;
  }

  // Apply Gaussian blur
  cv::Mat gaussian_blurred_image;
  cv::GaussianBlur(src, gaussian_blurred_image, ksize, sigmaX, sigmaY);

  return gaussian_blurred_image;
}

int image_processing::IMAGE_TEST_BLOCK(char **path2image) {

  cv::Mat image = get_image_from_file(path2image);
  display_image(image, "Original Image");
  cv::Mat grey_image = image_greyscale(image);
  display_image(grey_image, "Greyscale Image");
  cv::Mat blurred_image = blur_image(image, cv::Size(10, 10));
  display_image(blurred_image, "Blurred Image");
  cv::Mat gaussian_blurred_image =
      gaussian_blur_image(image, cv::Size(9, 0), 0, 0);
  display_image(gaussian_blurred_image, "Gaussian Blurred Image");
  return 0;
}

} // namespace DETECTION_IMAGE_PROCESSING
