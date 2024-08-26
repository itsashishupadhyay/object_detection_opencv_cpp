#include "video_processing.h"
#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>

namespace DETECTION_VIDEO_PROCESSING {

int video_processing::display_video(std::string path2video) {

  cv::VideoCapture cap(path2video);

  double fps = cap.get(cv::CAP_PROP_FPS);
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
#ifndef NDEBUG
  printf("Path to video: %s\n", path2video.c_str());
  printf("Frame rate: %.2f\n", fps);
  printf("Video width: %d\n", width);
  printf("Video height: %d\n", height);
  printf("Total frames: %d\n", totalFrames);
#endif
  while (true) {
    cv::Mat frame;
    cap.read(frame);
    if (frame.empty()) {
      printf("No frame data \n");
      break;
    }
    cv::namedWindow("Display Video", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Video", frame);
    cv::waitKey(
        1000 / fps); // wait dependent on how long a frame needs to be displayed

    if (cv::waitKey(1) == 27) { // Exit if ESC key is pressed
      break;
    }
  }

  return 0;
}

int video_processing::run_object_detetion(std::string path2video,
                                          std::string path2label,
                                          std::string path2model) {

  printf("Path to video: %s\n", path2video.c_str());
  cv::VideoCapture cap(path2video);

  double fps = cap.get(cv::CAP_PROP_FPS);
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
#ifndef NDEBUG
  printf("Frame rate: %.2f\n", fps);
  printf("Video width: %d\n", width);
  printf("Video height: %d\n", height);
  printf("Total frames: %d\n", totalFrames);
#endif
  cv::Mat obj_detected_frame;
  DETECTION_IMAGE_PROCESSING::image_processing each_frame;
  cv::namedWindow("Object detected Video", cv::WINDOW_AUTOSIZE);
  while (true) {
    cv::Mat frame;
    cap.read(frame);
    if (frame.empty()) {
      printf("No frame data \n");
      break;
    }

    double start_time = cv::getTickCount();

    obj_detected_frame =
        each_frame.run_yolo_obj_detection(frame, path2label, path2model);

    cv::imshow("Object Video", obj_detected_frame);

    double end_time = cv::getTickCount();
    double processing_time = (end_time - start_time) / cv::getTickFrequency();

#ifndef NDEBUG
    printf("Processing time: %.2f\n", processing_time);
#endif

    // if (wait_time > 0) {
    //   cv::waitKey(wait_time);
    // }

    if (cv::waitKey(1) == 27) { // Exit if ESC key is pressed
      break;
    }
  }
  return 0;
}

int video_processing::display_webcam() {
  cv::VideoCapture cap(0); // Open the default camera

  if (!cap.isOpened()) {
    printf("Failed to open the camera\n");
    return -1;
  }

  cv::namedWindow("Webcam", cv::WINDOW_NORMAL);

  while (true) {
    cv::Mat frame;
    cap.read(frame); // Read a new frame from the camera

    if (frame.empty()) {
      printf("Failed to capture frame\n");
      break;
    }

    cv::imshow("Webcam", frame); // Display the frame

    if (cv::waitKey(1) == 27) { // Exit if ESC key is pressed
      break;
    }
  }

  cap.release();           // Release the camera
  cv::destroyAllWindows(); // Close all windows

  return 0;
}

int video_processing::run_object_detetion_webcam(std::string path2label,
                                                 std::string path2model) {
  cv::VideoCapture cap(0); // Open the default camera
  if (!cap.isOpened()) {
    printf("Failed to open the camera\n");
    return -1;
  }
  cv::namedWindow("Webcam Object Detection", cv::WINDOW_NORMAL);

  cv::Mat obj_detected_frame;
  DETECTION_IMAGE_PROCESSING::image_processing each_frame;

  while (true) {
    cv::Mat frame;
    cap.read(frame); // Read a new frame from the camera
    if (frame.empty()) {
      printf("Failed to capture frame\n");
      break;
    }
    obj_detected_frame =
        each_frame.run_yolo_obj_detection(frame, path2label, path2model);
    cv::imshow("Webcam Object detection", obj_detected_frame);

    if (cv::waitKey(1) == 27) { // Exit if ESC key is pressed
      break;
    }
  }

  cap.release();           // Release the camera
  cv::destroyAllWindows(); // Close all windows

  return 0;
}

} // namespace DETECTION_VIDEO_PROCESSING