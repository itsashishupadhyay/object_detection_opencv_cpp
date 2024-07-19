#ifndef DISPLAY_IMAGE_H
#define DISPLAY_IMAGE_H

#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

namespace DETECTION_IMAGE_PROCESSING {
class image_processing {
private:
  cv::Mat get_image(char **path2image, bool greyscale);

public:
  int display_image(char **path2image);
  int display_image_greyscale(char **path2image);
};

} // namespace DETECTION_IMAGE_PROCESSING

#ifdef __cplusplus
}
#endif

#endif // DISPLAY_IMAGE_H