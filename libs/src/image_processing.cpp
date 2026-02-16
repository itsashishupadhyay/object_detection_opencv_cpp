#include "image_processing.h"
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <stdio.h>

namespace DETECTION_IMAGE_PROCESSING {

cv::Mat image_processing::get_image_from_file(std::string path2image) {
  cv::Mat image;
  image = cv::imread(path2image, cv::IMREAD_COLOR);

  if (image.empty()) {
    printf("No image data \n");
  }
#ifndef NDEBUG
  else {
    std::cout << "Image size is: " << image.size() << std::endl;
  }
#endif

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

void image_processing::draw_label(cv::Mat &input_image, std::string label,
                                  int left, int top) {

  // Display the label at the top of the bounding box.
  int baseLine;
  cv::Size label_size =
      cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
  top = std::max(top, label_size.height);
  // Top left corner.
  cv::Point tlc = cv::Point(left, top);
  // Bottom right corner.
  cv::Point brc =
      cv::Point(left + label_size.width, top + label_size.height + baseLine);
  // Draw white rectangle.
  rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
  // Put the label on the black rectangle.
  putText(input_image, label, cv::Point(left, top + label_size.height),
          FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);

  return;
}

std::vector<cv::Mat> image_processing::pre_process_yolo(cv::Mat &image,
                                                        cv::dnn::Net &net) {
  // Convert to blob.
  cv::Mat blob;
  cv::dnn::blobFromImage(image, blob, 1. / 255.,
                         cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(),
                         true, false);

  net.setInput(blob);

  // Forward propagate.
  std::vector<cv::Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

  return outputs;
}

cv::Mat image_processing::post_process_yolo(
    cv::Mat &input_image, std::vector<cv::Mat> &outputs,
    const std::vector<std::string> &class_name) {
  // Initialize vectors to hold respective outputs while unwrapping detections.
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  // Resizing factor.
  float x_factor = input_image.cols / INPUT_WIDTH;
  float y_factor = input_image.rows / INPUT_HEIGHT;
  float *data = (float *)outputs[0].data;
  const int dimensions = 85;
  // 25200 for default size 640.
  const int rows = 25200;
  // Iterate through 25200 detections.
  for (int i = 0; i < rows; ++i) {
    float confidence = data[4];
    // Discard bad detections and continue.
    if (confidence >= CONFIDENCE_THRESHOLD) {
      float *classes_scores = data + 5;
      // Create a 1x85 Mat and store class scores of 80 classes.
      cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
      // Perform minMaxLoc and acquire the index of best class  score.
      cv::Point class_id;
      double max_class_score;
      minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
      // Continue if the class score is above the threshold.
      if (max_class_score > SCORE_THRESHOLD) {
        // Store class ID and confidence in the pre-defined respective vectors.
        confidences.push_back(confidence);
        class_ids.push_back(class_id.x);
        // Center.
        float cx = data[0];
        float cy = data[1];
        // Box dimension.
        float w = data[2];
        float h = data[3];
        // Bounding box coordinates.
        int left = int((cx - 0.5 * w) * x_factor);
        int top = int((cy - 0.5 * h) * y_factor);
        int width = int(w * x_factor);
        int height = int(h * y_factor);
        // Store good detections in the boxes vector.
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
    // Jump to the next row.
    data += 85;
  }

#ifndef NDEBUG
  std::cout << "YOLOv5: " << boxes.size()
            << " boxes before NMS (img: " << input_image.cols << "x"
            << input_image.rows << ")" << std::endl;
#endif

  // Perform Non-Maximum Suppression and draw predictions.
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
                    indices);

#ifndef NDEBUG
  std::cout << "YOLOv5: " << indices.size() << " boxes after NMS" << std::endl;
#endif

  for (int i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    cv::Rect box = boxes[idx];
    int left = box.x;
    int top = box.y;
    int width = box.width;
    int height = box.height;
    // Draw bounding box.
    cv::rectangle(input_image, cv::Point(left, top),
                  cv::Point(left + width, top + height), BLUE, 3 * THICKNESS);
    // Get the label for the class name and its confidence.
    std::string label = cv::format("%.2f", confidences[idx]);
    label = class_name[class_ids[idx]] + ":" + label;
#ifndef NDEBUG
    std::cout << "Detected: " << label << " at [" << left << "," << top << " "
              << width << "x" << height << "]" << std::endl;
#endif
    // Draw class labels.
    draw_label(input_image, label, left, top);
  }
  return input_image;
}

int image_processing::detect_model_version(
    const std::vector<cv::Mat> &outputs) {
  // Detect model version based on output shape
  // YOLOv5: shape [1, 25200, 85] - dimensions at index 2
  // YOLOv8: shape [1, 84, 8400] - dimensions at index 1
  // YOLO26: shape [1, 300, 6] - NMS-free one-to-one head
  if (outputs.empty()) {
    return 0; // Default to YOLOv5
  }

  cv::Mat output = outputs[0];
  std::vector<int> shape;
  for (int i = 0; i < output.dims; i++) {
    shape.push_back(output.size[i]);
  }

#ifndef NDEBUG
  std::cout << "Model output shape: [";
  for (int i = 0; i < shape.size(); i++) {
    std::cout << shape[i];
    if (i < shape.size() - 1)
      std::cout << ", ";
  }
  std::cout << "]" << std::endl;
#endif

  if (shape.size() >= 3) {
    // YOLO26 NMS-free format: [1, 300, 6]
    if (shape[1] == 300 && shape[2] == 6) {
#ifndef NDEBUG
      std::cout << "Detected YOLO26 model format (NMS-free)" << std::endl;
#endif
      return 2;
    }
    // YOLOv8 format: [1, 84, 8400]
    else if (shape[1] == 84 && shape[2] == 8400) {
#ifndef NDEBUG
      std::cout << "Detected YOLOv8 model format" << std::endl;
#endif
      return 1;
    }
    // YOLOv5 format: [1, 25200, 85]
    else if (shape[1] == 25200 && shape[2] == 85) {
#ifndef NDEBUG
      std::cout << "Detected YOLOv5 model format" << std::endl;
#endif
      return 0;
    }
  }

  // Default to YOLOv5 if detection fails
  return 0;
}

cv::Mat image_processing::post_process_yolov8(
    cv::Mat &input_image, std::vector<cv::Mat> &outputs,
    const std::vector<std::string> &class_name) {
  // YOLOv8 output format: [1, 84, 8400]
  // 84 = 4 (bbox) + 80 (classes), no objectness score
  // Layout is transposed compared to YOLOv5

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  float x_factor = input_image.cols / INPUT_WIDTH;
  float y_factor = input_image.rows / INPUT_HEIGHT;

  cv::Mat output = outputs[0];

  // Transpose from [1, 84, 8400] to [8400, 84] for easier processing
  cv::Mat output_transposed;
  if (output.dims == 3) {
    // Reshape to [84, 8400]
    cv::Mat reshaped = output.reshape(1, output.size[1]);
    // Transpose to [8400, 84]
    cv::transpose(reshaped, output_transposed);
  } else {
    output_transposed = output;
  }

  int rows = output_transposed.rows; // 8400

#ifndef NDEBUG
  std::cout << "YOLOv8 processing " << rows << " detections" << std::endl;
#endif

  for (int i = 0; i < rows; ++i) {
    float *data = output_transposed.ptr<float>(i);

    // First 4 values are bbox coordinates
    float cx = data[0];
    float cy = data[1];
    float w = data[2];
    float h = data[3];

    // Next 80 values are class scores (no objectness score in YOLOv8)
    cv::Mat scores(1, class_name.size(), CV_32FC1, data + 4);
    cv::Point class_id;
    double max_class_score;
    cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

    // Use max_class_score as confidence (YOLOv8 doesn't have separate
    // objectness)
    if (max_class_score > SCORE_THRESHOLD) {
      confidences.push_back(max_class_score);
      class_ids.push_back(class_id.x);

      // Calculate bounding box coordinates
      int left = int((cx - 0.5 * w) * x_factor);
      int top = int((cy - 0.5 * h) * y_factor);
      int width = int(w * x_factor);
      int height = int(h * y_factor);

      boxes.push_back(cv::Rect(left, top, width, height));
    }
  }

  // Perform Non-Maximum Suppression
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
                    indices);

  for (int i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    cv::Rect box = boxes[idx];
    int left = box.x;
    int top = box.y;
    int width = box.width;
    int height = box.height;

    // Draw bounding box
    cv::rectangle(input_image, cv::Point(left, top),
                  cv::Point(left + width, top + height), BLUE, 3 * THICKNESS);

    // Create label
    std::string label = cv::format("%.2f", confidences[idx]);
    label = class_name[class_ids[idx]] + ":" + label;

#ifndef NDEBUG
    std::cout << "Detected: " << label << std::endl;
#endif

    draw_label(input_image, label, left, top);
  }

  return input_image;
}

cv::Mat image_processing::post_process_yolo26(
    cv::Mat &input_image, std::vector<cv::Mat> &outputs,
    const std::vector<std::string> &class_name) {
  // YOLO26 NMS-free output format: [1, 300, 6]
  // 300 = maximum detections (already filtered, no NMS needed!)
  // 6 = [cx, cy, w, h, class_confidence, class_id] or similar format
  // Key advantage: No NMS post-processing needed - 43% faster!

  float x_factor = input_image.cols / INPUT_WIDTH;
  float y_factor = input_image.rows / INPUT_HEIGHT;

  cv::Mat output = outputs[0];

  // Reshape to [300, 6] for easier processing
  cv::Mat output_reshaped;
  if (output.dims == 3) {
    output_reshaped = output.reshape(1, output.size[1]);
  } else {
    output_reshaped = output;
  }

  int num_detections = output_reshaped.rows; // Should be 300

#ifndef NDEBUG
  std::cout << "YOLO26 NMS-free processing " << num_detections
            << " detections (no post-processing needed!)" << std::endl;
#endif

  int valid_detections = 0;

  // Store processed detections to avoid duplicates
  struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
  };
  std::vector<Detection> processed_detections;

  // Track unique confidence values to detect parsing issues
  std::set<float> unique_confidences;

  for (int i = 0; i < num_detections; ++i) {
    float *data = output_reshaped.ptr<float>(i);

#ifndef NDEBUG
    // Debug: Print ALL 6 raw values for first few detections with any
    // confidence
    static int raw_debug_count = 0;
    if (data[0] != 0.0f || data[1] != 0.0f || data[2] != 0.0f) {
      if (raw_debug_count < 10) {
        std::cout << "RAW [" << raw_debug_count << "]: "
                  << "data[0]=" << data[0] << " data[1]=" << data[1]
                  << " data[2]=" << data[2] << " data[3]=" << data[3]
                  << " data[4]=" << data[4] << " data[5]=" << data[5]
                  << std::endl;
        raw_debug_count++;
      }
    }
#endif

    // YOLO26/YOLO11 end-to-end output format: [1, 300, 6]
    // Ultralytics YOLO11 end-to-end format is: [x1, y1, x2, y2, confidence,
    // class_id] BUT coordinates are in INPUT resolution (640x640)
    float x1 = data[0];
    float y1 = data[1];
    float x2 = data[2];
    float y2 = data[3];
    float confidence = data[4];
    int class_id = static_cast<int>(data[5] + 0.5); // Round to nearest int

    // Ensure x1 < x2 and y1 < y2 (swap if needed)
    if (x1 > x2)
      std::swap(x1, x2);
    if (y1 > y2)
      std::swap(y1, y2);

    // Validate that we have a reasonable bounding box
    float box_width = x2 - x1;
    float box_height = y2 - y1;
    bool valid_box = (box_width > 1.0f) && (box_height > 1.0f) &&
                     (box_width < INPUT_WIDTH) && (box_height < INPUT_HEIGHT);

    // Filter by confidence threshold
    // Note: YOLO26 already applies internal filtering, so most detections
    // below threshold will have confidence near 0
    if (confidence > SCORE_THRESHOLD && confidence <= 1.0f && class_id >= 0 &&
        class_id < class_name.size() && valid_box) {

#ifndef NDEBUG
      // Debug: Print raw coordinates of valid detections
      static int valid_debug_count = 0;
      if (valid_debug_count < 3) {
        std::cout << "VALID Detection #" << valid_debug_count << ": "
                  << "x1=" << x1 << " y1=" << y1 << " x2=" << x2 << " y2=" << y2
                  << " conf=" << confidence << " class=" << class_id
                  << std::endl;
        valid_debug_count++;
      }
#endif

      // Scale coordinates from 640x640 input space to actual image size
      int left = int(x1 * x_factor);
      int top = int(y1 * y_factor);
      int width = int((x2 - x1) * x_factor);
      int height = int((y2 - y1) * y_factor);

      // Ensure bounding box is within image bounds
      left = std::max(0, std::min(left, input_image.cols - 1));
      top = std::max(0, std::min(top, input_image.rows - 1));
      width = std::max(1, std::min(width, input_image.cols - left));
      height = std::max(1, std::min(height, input_image.rows - top));

      cv::Rect current_box(left, top, width, height);

      // Check for duplicates: skip if we've already processed a very similar
      // detection
      bool is_duplicate = false;
      for (const auto &prev : processed_detections) {
        if (prev.class_id == class_id) {
          // Check if confidences are very close (within 0.001)
          bool conf_similar = std::abs(prev.confidence - confidence) < 0.001;

          // Calculate IoU (Intersection over Union)
          cv::Rect intersection = prev.box & current_box;
          float intersection_area = intersection.area();
          float union_area =
              prev.box.area() + current_box.area() - intersection_area;
          float iou = (union_area > 0) ? (intersection_area / union_area) : 0;

          // If confidences are identical and any overlap exists, it's likely a
          // duplicate Otherwise use standard IoU threshold
          if ((conf_similar && iou > 0.1) || iou > 0.5) {
            is_duplicate = true;
            break;
          }
        }
      }

      if (!is_duplicate) {
        valid_detections++;
        processed_detections.push_back({class_id, confidence, current_box});
        unique_confidences.insert(confidence);

        // Draw bounding box
        cv::rectangle(input_image, cv::Point(left, top),
                      cv::Point(left + width, top + height), BLUE,
                      3 * THICKNESS);

        // Create label
        std::string label = cv::format("%.2f", confidence);
        label = class_name[class_id] + ":" + label;

#ifndef NDEBUG
        std::cout << "Detected: " << label << " [NMS-free]" << std::endl;
#endif

        draw_label(input_image, label, left, top);
      }
#ifndef NDEBUG
      else {
        std::cout << "Skipped duplicate: " << class_name[class_id] << ":"
                  << cv::format("%.2f", confidence) << std::endl;
      }
#endif
    }
  }

#ifndef NDEBUG
  std::cout << "YOLO26: " << valid_detections << " valid detections (out of "
            << num_detections << " processed)" << std::endl;

  // Warn if all confidence values are identical (indicates parsing issue)
  if (unique_confidences.size() == 1 && valid_detections > 1) {
    std::cout << "WARNING: All detections have identical confidence ("
              << *unique_confidences.begin()
              << "). This may indicate incorrect output format parsing!"
              << std::endl;
    std::cout
        << "         The model might not be properly exported as end-to-end, "
        << std::endl;
    std::cout << "         or the output format interpretation is incorrect."
              << std::endl;
  }
#endif

  return input_image;
}

cv::Mat
image_processing::run_yolo_obj_detection(cv::Mat &frame,
                                         std::string path2lables = "",
                                         std::string path2yolo_onnx = "") {

  if (class_list.empty() && onnx_net.empty()) {
#ifndef NDEBUG
    std::cout << "class_list and onnex net are empty" << std::endl;
#endif
    if (path2lables.empty()) {
      auto currentPath = std::filesystem::current_path();
      path2lables = (currentPath / "weight" / "coco.names").string();
    }
    if (path2yolo_onnx.empty()) {
      auto currentPath = std::filesystem::current_path();
      path2yolo_onnx = (currentPath / "weight" / "yolov5s.onnx").string();
    }

    std::ifstream ifs(path2lables);
    std::string line;
    while (getline(ifs, line)) {
      this->class_list.push_back(line);
    }
    // Load model.
    this->onnx_net = cv::dnn::readNetFromONNX(path2yolo_onnx);
  }
  cv::Mat obj_detected_frame = frame.clone();
  std::vector<cv::Mat> detections; // Process the image.
  detections = pre_process_yolo(frame, onnx_net);

  // Detect model version on first run
  static bool version_detected = false;
  static int model_version = 0; // 0=YOLOv5, 1=YOLOv8, 2=YOLO26
  if (!version_detected) {
    model_version = detect_model_version(detections);
    this->is_yolov8_model = (model_version == 1);
    this->is_yolo26_model = (model_version == 2);
    version_detected = true;
  }

  // Use appropriate post-processing based on model version
  cv::Mat yolo_img;
  if (this->is_yolo26_model) {
    // YOLO26: NMS-free end-to-end inference (43% faster!)
    yolo_img = post_process_yolo26(obj_detected_frame, detections, class_list);
  } else if (this->is_yolov8_model) {
    // YOLOv8: Transposed format without objectness
    yolo_img = post_process_yolov8(obj_detected_frame, detections, class_list);
  } else {
    // YOLOv5: Traditional format with objectness score
    yolo_img = post_process_yolo(obj_detected_frame, detections, class_list);
  }
#ifndef NDEBUG
  // Put efficiency information.
  // The function getPerfProfile returns the overall time for     inference(t)
  // and the timings for each of the layers(in layersTimes).
  std::vector<double> layersTimes;
  double freq = cv::getTickFrequency() / 1000;
  double t = onnx_net.getPerfProfile(layersTimes) / freq;
  std::string label = cv::format("Inference time : %.2f ms", t);
  putText(yolo_img, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);
#endif
  return yolo_img;
}

int image_processing::detect_objects_in_image(
    std::string path2image, std::string object_labes_path = "",
    std::string onnx_file_path = "") {
  cv::Mat image = get_image_from_file(path2image);
  cv::Mat yolo_image =
      run_yolo_obj_detection(image, object_labes_path, onnx_file_path);
  display_image(yolo_image, "yolo processed image");
  return 0;
}

int image_processing::IMAGE_TEST_BLOCK(std::string path2image) {

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
  //     edges, "Canny Edge Detector on Gaussian Blurred Image dilate then
  //     erode");

  // cv::Mat corrected_image = get_top_perspective(image);
  // display_image(corrected_image, "Corrected Image");

  // cv::Mat yolo_image = run_yolo_obj_detection(image);
  // display_image(yolo_image, "yolo processed image");

  return 0;
}

} // namespace DETECTION_IMAGE_PROCESSING
