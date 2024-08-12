#include "image_processing.h"
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>

namespace DETECTION_IMAGE_PROCESSING {

cv::Mat image_processing::get_image_from_file(char **path2image) {
  cv::Mat image;
  image = cv::imread(std::string(*path2image), cv::IMREAD_COLOR);

  if (image.empty()) {
    printf("No image data \n");
  } else {
    std::cout << "Image size is: " << image.size() << std::endl;
  }

  return image;
}

int image_processing::display_image(cv::Mat &image, std::string displaymsg = "",
                                    std::string put_text_on_image = "") {

  std::string window_name;
  if (displaymsg.empty()) {
    window_name = "Display Image";
  } else {
    window_name = displaymsg;
  }
  if (!put_text_on_image.empty()) {
    cv::putText(image, put_text_on_image, cv::Point(10, 50),
                cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(255, 255, 255), 1);
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

/**
  * @brief Function to blur an image
  * @param src: Source image
  * @param kernelSize: Size of the kernel to be used for blurring
  * @param anchorPoint: Anchor point of the kernel. Default is (-1, -1) which
  * means the anchor is at the kernel center
  * @return Blurred image
  *
  *
  * NOTE: Size Parameter:
  * The Size parameter in the blur() function specifies the size of the kernel
  * (filter) to be used for smoothing. It is defined as Size(w, h), where:
  *
  * w is the width of the kernel in pixels (must be a positive odd integer)
  * h is the height of the kernel in pixels (must be a positive odd integer)
  *
  * The kernel size determines the number of neighboring pixels to consider when
  * calculating the average value for each pixel in the output image. A larger
  * kernel size will result in more smoothing, while a smaller kernel size will
  * result in less smoothing. Anchor Point Parameter: The anchor point parameter
  * in the blur() function specifies the location of the anchor pixel (the pixel
  * being evaluated) relative to the kernel. It is defined as Point(x, y),
  where:
  *
  * x is the x-coordinate of the anchor point (can be negative or positive)
  * y is the y-coordinate of the anchor point (can be negative or positive)
  *
  * If the anchor point is set to Point(-1, -1), it means that the center of the
  * kernel is aligned with the pixel being evaluated. This is the default
  behavior
  * and is commonly used. If the anchor point is set to a positive value, it
  means
  * that the kernel is shifted to the right (for x) or down (for y) by the
  * specified number of pixels.

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

/**
 * @brief Function to apply Gaussian blur to an image
 * @param src: Source image
 * @param ksize: Size of the kernel to be used for blurring
 * @param sigmaX: Standard deviation of the Gaussian kernel in X direction
 * @param sigmaY: Standard deviation of the Gaussian kernel in Y direction
 * @return Gaussian blurred image
 * @note
 * - A Gaussian kernel is a specific type of kernel that is designed to blur an
 * image by averaging neighboring pixels. The Gaussian kernel is defined by its
 * size, which is typically an odd number (e.g., 3, 5, 7, etc.). The kernel is a
 * square matrix with a symmetrical, bell-shaped distribution of values.
 *
 * - What do SigmaX and SigmaY represent?
 *
 * - sigmaX: The standard deviation of the Gaussian kernel in the X direction. A
 * higher value of sigmaX means that the kernel will be more spread out in the X
 * direction, resulting in more blur in that direction. sigmaY: The standard
 * deviation of the Gaussian kernel in the Y direction. A higher value of sigmaY
 * means that the kernel will be more spread out in the Y direction, resulting
 * in more blur in that direction.
 *
 * - Effects of SigmaX and SigmaY on Gaussian Blur
 * - Here are some key effects of varying sigmaX and sigmaY:
 *
 * - Isotropic Blur: When sigmaX = sigmaY, the blur is isotropic, meaning it is
 * equal in all directions. This is the default behavior when sigmaX and sigmaY
 * are both 0. Anisotropic Blur: When sigmaX ≠ sigmaY, the blur is anisotropic,
 * meaning it is different in different directions. This can be useful for
 * creating motion blur or simulating camera shake. Increased Blur: As sigmaX
 * and sigmaY increase, the blur becomes more pronounced, and the image becomes
 * more smoothed. Directional Blur: By setting sigmaX > sigmaY or vice versa,
 * you can create directional blur, where the blur is more pronounced in one
 * direction than the other.
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

/**
 * @brief Applies a perspective transformation to an image to correct the
 * perspective.
 * @param image The input image.
 * @param src_points The source points for the perspective transformation.
 * @param dst_points The destination points for the perspective transformation.
 * @return The corrected image.
 * @details
 * - The function applies a perspective transformation to the input image to
 * correct the perspective.
 * - if source and destination points are not provided, the function calculates
 * the perspective points and appiles the transformation.
 */
cv::Mat image_processing::get_top_perspective(
    cv::Mat &image, std::vector<cv::Point2f> src_points = {},
    std::vector<cv::Point2f> dst_points = {}) {

  cv::Mat corrected_image;

  if (src_points.empty() || dst_points.empty()) {
    std::cout << "Source and destination points not provided caculating points "
                 "for perspective transformation"
              << std::endl;

    // Convert the image to greyscale
    cv::Mat grey_image = image_greyscale(image);
    cv::Mat gaussian_blurred_image, edges;
    canny_config config;

    gaussian_blurred_image =
        gaussian_blur_image(grey_image, cv::Size(3, 3), 0, 0);
    display_image(gaussian_blurred_image, "Gaussian Blurred Image");

    config.image = gaussian_blurred_image;
    config.threshold1 = 100;
    config.threshold2 = 200;
    config.apertureSize = 3;
    config.L2gradient = false;
    config.dilate = true;
    config.erode = false;
    config.kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    config.anchor = cv::Point(-1, -1);
    config.iterations = 1;
    config.borderType = cv::BORDER_REFLECT_101;
    config.borderValue = cv::morphologyDefaultBorderValue();
    edges = canny_edge_detector(config);
    display_image(edges, "Canny Edge Detector on Gaussian Blurred Image");

    // Find contours in the edge map
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(edges, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);

    // Find the largest contour
    double max_area = 0;
    int max_area_idx = -1;
    for (int i = 0; i < contours.size(); i++) {
      double area = cv::contourArea(contours[i]);
      if (area > max_area) {
        max_area = area;
        max_area_idx = i;
      }
    }

    // Get the largest contour
    std::vector<cv::Point> largest_contour = contours[max_area_idx];

    // Approximate the contour with a polygon using approxPolyDP
    // epsilon is set to 2% of the contour's perimeter
    double epsilon = 0.02 * cv::arcLength(largest_contour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(largest_contour, approx, epsilon, true);

    // Find the convex hull of the approximated polygon
    std::vector<cv::Point> hull;
    cv::convexHull(approx, hull);

    // Ensure the convex hull has 4 sides (a quadrilateral)
    if (hull.size() != 4) {
      // Adjust epsilon and repeat the process
      epsilon = 0.01 * cv::arcLength(largest_contour, true);
      cv::approxPolyDP(largest_contour, approx, epsilon, true);
      cv::convexHull(approx, hull);
    }

    // Draw the original contour
    cv::Mat contour_image = image.clone();
    cv::drawContours(contour_image, contours, max_area_idx,
                     cv::Scalar(0, 255, 0), 2);
    display_image(contour_image, "Original Contour");

    // Draw the approximated polygon
    cv::Mat approx_image = image.clone();
    cv::drawContours(approx_image, std::vector<std::vector<cv::Point>>{approx},
                     0, cv::Scalar(255, 0, 0), 2);
    display_image(approx_image, "Approximated Polygon");

    // Draw the convex hull (quadrilateral)
    cv::Mat hull_image = image.clone();
    cv::drawContours(hull_image, std::vector<std::vector<cv::Point>>{hull}, 0,
                     cv::Scalar(0, 0, 255), 2);
    display_image(hull_image, "Convex Hull (Quadrilateral)");

    // If the convex hull has 4 sides, use it as is
    std::vector<cv::Point> src_points;
    if (hull.size() == 4) {
      src_points = hull;
      std::cout << "Convex hull has 4 sides. Using it as source points."
                << std::endl;
    } else {
      // Find the largest area 4-sided quadrilateral within the convex hull
      std::cout << "Convex hull has " << hull.size()
                << " sides. Finding the largest area 4-sided quadrilateral "
                   "within the convex hull."
                << std::endl;
      double max_area = 0;
      std::vector<cv::Point> max_quad;

      // Iterate over all possible combinations of 4 points
      for (int i = 0; i < hull.size(); i++) {
        for (int j = i + 1; j < hull.size(); j++) {
          for (int k = j + 1; k < hull.size(); k++) {
            for (int l = k + 1; l < hull.size(); l++) {
              // Calculate the area of the current quadrilateral
              double area =
                  0.5 *
                  std::abs((hull[i].x * hull[j].y + hull[j].x * hull[k].y +
                            hull[k].x * hull[l].y + hull[l].x * hull[i].y) -
                           (hull[j].x * hull[i].y + hull[k].x * hull[j].y +
                            hull[l].x * hull[k].y + hull[i].x * hull[l].y));

              // Update the maximum area quadrilateral if needed
              if (area > max_area) {
                max_area = area;
                max_quad = {hull[i], hull[j], hull[k], hull[l]};
              }
            }
          }
        }
      }

      // Use the maximum area quadrilateral as the source points
      src_points = max_quad;
    }

    // // get the min rectangle that can be formed from the convex hull
    // cv::RotatedRect rect = cv::minAreaRect(hull);
    // cv::Point2f rect_points[4];
    // rect.points(rect_points);

    // Draw the rotated rectangle
    cv::Mat quad_image = image.clone();

    for (int i = 0; i < 4; i++) {
      cv::line(quad_image, src_points[i], src_points[(i + 1) % 4],
               cv::Scalar(0, 255, 0), 2);
    }
    display_image(quad_image, "Perspective Quad");

    cv::Point2f src_points_2f[4];
    for (int i = 0; i < 4; i++) {
      src_points_2f[i] = cv::Point2f(src_points[i].x, src_points[i].y);
    }

    cv::Point2f dst_points[4];
    // height and width of the Quadrilateral
    int width = std::max(cv::norm(src_points_2f[1] - src_points_2f[0]),
                         cv::norm(src_points_2f[2] - src_points_2f[3]));
    int height = std::max(cv::norm(src_points_2f[3] - src_points_2f[0]),
                          cv::norm(src_points_2f[2] - src_points_2f[1]));

    dst_points[0] = cv::Point2f(width, height); // Bottom-right
    dst_points[1] = cv::Point2f(0, height);     // Bottom-left
    dst_points[2] = cv::Point2f(0, 0);          // Top-left
    dst_points[3] = cv::Point2f(width, 0);      // Top-right

    // Get the perspective transform matrix
    cv::Mat matrix = cv::getPerspectiveTransform(src_points_2f, dst_points);

    // Warp the image
    cv::Mat warped_image;
    cv::warpPerspective(image, warped_image, matrix, cv::Size(width, height));

    // Display the warped image
    display_image(warped_image, "Warped Image");
    corrected_image = warped_image;
  } else {
    // TODO: implement the case where src_points and dst_points are provided
  }

  return corrected_image;
}

int image_processing::IMAGE_TEST_BLOCK(char **path2image) {

  cv::Mat image = get_image_from_file(path2image);
  display_image(image, "Original Image");
  // cv::Mat grey_image = image_greyscale(image);
  // display_image(grey_image, "Greyscale Image");
  // cv::Mat blurred_image = blur_image(image, cv::Size(10, 10));
  // display_image(blurred_image, "Blurred Image");
  // cv::Mat gaussian_blurred_image =
  //     gaussian_blur_image(image, cv::Size(9, 0), 0, 0);
  // display_image(gaussian_blurred_image, "Gaussian Blurred Image");

  // canny_config config;
  // config.image = image;
  // config.threshold1 = 100;
  // config.threshold2 = 200;
  // config.apertureSize = 3;
  // config.L2gradient = false;
  // config.dilate = false;
  // config.erode = false;
  // config.kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  // config.anchor = cv::Point(-1, -1);
  // config.iterations = 1;
  // config.borderType = cv::BORDER_REFLECT_101;
  // config.borderValue = cv::morphologyDefaultBorderValue();

  // cv::Mat edges = canny_edge_detector(config);
  // display_image(edges, "Canny Edge Detector original");

  // config.image = blurred_image;
  // edges = canny_edge_detector(config);
  // display_image(edges, "Canny Edge Detector on Blurred Image");

  // config.image = gaussian_blurred_image;
  // edges = canny_edge_detector(config);
  // display_image(edges, "Canny Edge Detector on Gaussian Blurred Image");

  // config.image = gaussian_blurred_image;
  // config.dilate = true;

  // edges = canny_edge_detector(config);
  // display_image(edges,
  //               "Canny Edge Detector on Gaussian Blurred Image dilate true");

  // config.erode = true;
  // edges = canny_edge_detector(config);
  // display_image(
  //     edges,
  //     "Canny Edge Detector on Gaussian Blurred Image dilate then erode
  //     ,true");

  cv::Mat corrected_image = get_top_perspective(image);
  // display_image(corrected_image, "Corrected Image");

  return 0;
}

} // namespace DETECTION_IMAGE_PROCESSING
