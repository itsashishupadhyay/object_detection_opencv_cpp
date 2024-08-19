#ifndef VIDEO_PROCESSING_H
#define VIDEO_PROCESSING_H

#include <string>

#ifdef __cplusplus
extern "C" {
#endif

namespace DETECTION_VIDEO_PROCESSING {
class video_processing {
public:
  int display_video(std::string path2video);
  int display_webcam();
  int run_object_detetion(std::string path2video, std::string path2label,
                          std::string path2model);
};

} // namespace DETECTION_VIDEO_PROCESSING

#ifdef __cplusplus
}
#endif

#endif // VIDEO_PROCESSING_H