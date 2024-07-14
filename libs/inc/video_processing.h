#ifndef VIDEO_PROCESSING_H
#define VIDEO_PROCESSING_H

#ifdef __cplusplus
extern "C" {
#endif

namespace DETECTION_VIDEO_PROCESSING {
class video_processing {
public:
  int display_video(char **path2video);
  int display_webcam();
};

} // namespace DETECTION_VIDEO_PROCESSING

#ifdef __cplusplus
}
#endif

#endif // VIDEO_PROCESSING_H