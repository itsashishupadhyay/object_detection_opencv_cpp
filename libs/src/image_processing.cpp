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

/**
 * @brief Applies the Canny edge detector to an image.
 *
 * The Canny edge detector is a popular edge detection algorithm that uses the
 * gradient magnitude and direction to detect edges in an image.
 *
 * @param config A `canny_config` object containing the parameters for the Canny
 * edge detector.
 *
 * @return The edge map of the input image.
 *
 * @details
 * - The Canny edge detector first applies a Gaussian filter to the input image
 * to reduce noise.
 * - Then, it computes the gradient magnitude and direction using the Sobel
 * operator.
 * - The gradient magnitude is thresholded using the hysteresis procedure to
 * determine strong and weak edges.
 * - Finally, the edge map is constructed by connecting strong edges and
 * ignoring weak edges.
 *
 * The following steps are performed:
 *
 * 1. **Gaussian filtering**: The input image is filtered using a Gaussian
 * filter to reduce noise.
 * 2. **Gradient computation**: The gradient magnitude and direction are
 * computed using the Sobel operator.
 * 3. **Thresholding**: The gradient magnitude is thresholded using the
 * hysteresis procedure to determine strong and weak edges.
 * 4. **Edge map construction**: The edge map is constructed by connecting
 * strong edges and ignoring weak edges.
 * 5. **Dilation and erosion**: If `dilate` or `erode` is true, the edge map is
 * refined using dilation or erosion.
 *
 * @note
 * - The choice of `threshold1` and `threshold2` depends on the specific
 * application and the characteristics of the image. If the thresholds are too
 * low, the edge detector will be sensitive to noise. If the thresholds are too
 * high, the edge detector will miss edges.
 * - The choice of `apertureSize` depends on the specific application and the
 * characteristics of the image. A larger aperture size means that the Sobel
 * operator will consider a larger neighborhood of pixels when computing the
 * gradient.
 * - The choice of `L2gradient` depends on the specific application and the
 * characteristics of the image. The L2 norm is more accurate but also more
 * computationally expensive.
 * - The choice of `dilate` and `erode` depends on the specific application and
 * the characteristics of the image. Dilation and erosion can be used to refine
 * the edge map.
 * - The choice of `kernel`, `anchor`, `iterations`, `borderType`, and
 * `borderValue` depends on the specific application and the characteristics of
 * the image. These parameters are used for dilation and erosion.
 *
 * The `canny_config` object contains the following parameters:
 *
 * - `image`: The input image.
 * - `threshold1`: The first threshold for the hysteresis procedure.
 * - `threshold2`: The second threshold for the hysteresis procedure.
 * - `apertureSize`: The size of the Sobel operator used to compute the gradient
 * magnitude and direction.
 * - `L2gradient`: A flag indicating whether to use the L2 norm (true) or the L1
 * norm (false) to compute the gradient magnitude.
 * - `dilate`: A flag indicating whether to apply dilation to the edge map.
 * - `erode`: A flag indicating whether to apply erosion to the edge map.
 * - `kernel`: The kernel used for dilation or erosion.
 * - `anchor`: The anchor point for the kernel.
 * - `iterations`: The number of iterations for dilation or erosion.
 * - `borderType`: The border type for the edge map.
 * - `borderValue`: The border value for the edge map.
 */
cv::Mat image_processing::canny_edge_detector(const canny_config &config) {
  // Create an output edge map
  cv::Mat edges;

  // Apply Canny edge detector
  cv::Canny(config.image, edges, config.threshold1, config.threshold2,
            config.apertureSize, config.L2gradient);

  // Apply dilation or erosion if necessary
  if (config.dilate) {
    cv::dilate(edges, edges, config.kernel, config.anchor, config.iterations,
               config.borderType, config.borderValue);
  }
  if (config.erode) {
    cv::erode(edges, edges, config.kernel, config.anchor, config.iterations,
              config.borderType, config.borderValue);
  }

  return edges;
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

  canny_config config;
  config.image = image;
  config.threshold1 = 100;
  config.threshold2 = 200;
  config.apertureSize = 3;
  config.L2gradient = false;
  config.dilate = false;
  config.erode = false;
  config.kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  config.anchor = cv::Point(-1, -1);
  config.iterations = 1;
  config.borderType = cv::BORDER_REFLECT_101;
  config.borderValue = cv::morphologyDefaultBorderValue();

  cv::Mat edges = canny_edge_detector(config);
  display_image(edges, "Canny Edge Detector original");

  config.image = blurred_image;
  edges = canny_edge_detector(config);
  display_image(edges, "Canny Edge Detector on Blurred Image");

  config.image = gaussian_blurred_image;
  edges = canny_edge_detector(config);
  display_image(edges, "Canny Edge Detector on Gaussian Blurred Image");

  config.image = gaussian_blurred_image;
  config.dilate = true;

  edges = canny_edge_detector(config);
  display_image(edges,
                "Canny Edge Detector on Gaussian Blurred Image dilate true");

  config.erode = true;
  edges = canny_edge_detector(config);
  display_image(
      edges,
      "Canny Edge Detector on Gaussian Blurred Image dilate then erode ,true");

  return 0;
}

} // namespace DETECTION_IMAGE_PROCESSING
