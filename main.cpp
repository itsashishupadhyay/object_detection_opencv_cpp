#include "image_processing.h"
#include "navigation_decision.h"
#include "navigation_geometry.h"
#include "video_processing.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <vector>

void help_menu() {
  std::cout << "Usage: ./program_name [options]\n";
  std::cout << "Options:\n";
  std::cout << "  -h, --help        Display this help message\n";
  std::cout << "  -i, --image       Process an image\n";
  std::cout << "  -w, --webcam      Process the webcam\n";
  std::cout << "  -v, --video       Process a video\n";
  std::cout
      << "  -d, --detect      Process an object detection on video/image\n";
  std::cout << "  -p, --path        Specify the path to the image or video\n";
  std::cout << "  -l, --label       Specify the path to the label file\n";
  std::cout << "  -m, --model       Specify the path to the ONNX model file\n";
  std::cout << "  --mission <name>  Mission name (e.g. cassini)\n";
  std::cout << "  --instrument <n>  Instrument name (e.g. issna)\n";
  std::cout
      << "  --nav-fix         Compute NavFix per §9 and emit JSON per image\n";
  std::cout << "  --nav-decision    Compute NavDecision per §10 and emit JSON "
               "per image\n";
  std::cout << "  --overlay-dir <d> When used with --nav-decision, also write "
               "an annotated PNG\n";
  std::cout << "                    (<image_id>_overlay.png) into directory "
               "<d>.\n";
  std::cout << "\n";
  std::cout << "Examples:\n";
  std::cout << "  ./program_name -i -p /path/to/image.jpg\n";
  std::cout << "  ./program_name -i -d -p /path/to/image.jpg -l "
               "/path/to/label.txt -m /path/to/model.onnx\n";
  std::cout << "  ./program_name --nav-decision --mission cassini "
               "--instrument issna -p /path/to/image.png -l "
               "weight/cassini_issna_planets.names -m "
               "weight/cassini_issna_planets.onnx\n";
}

namespace {

// Simple lowercase substring check used for the weight-file mission sanity
// check (e.g. weight path must contain "cassini_issna" when --mission cassini
// --instrument issna is passed).
bool contains_ci(const std::string &hay, const std::string &needle) {
  if (needle.empty())
    return true;
  auto it = std::search(
      hay.begin(), hay.end(), needle.begin(), needle.end(),
      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
  return it != hay.end();
}

// Render a per-image overlay: draws every detection's bbox on a copy of the
// input frame, stamps the top class + confidence + NavDecision status, and
// writes to <overlay_dir>/<image_id>_overlay.png. Refuses silently (but logs
// to stderr) if anything goes wrong — the overlay is a cosmetic output, it
// must not affect the NavDecision JSON contract.
static void
render_nav_overlay(const cv::Mat &frame,
                   const std::vector<NAVIGATION::RawDetection> &dets,
                   const std::string &status_label, const std::string &image_id,
                   const std::filesystem::path &overlay_path) {
  cv::Mat canvas;
  if (frame.channels() == 1) {
    cv::cvtColor(frame, canvas, cv::COLOR_GRAY2BGR);
  } else {
    canvas = frame.clone();
  }

  cv::Scalar box_color;
  if (status_label == "NOMINAL") {
    box_color = cv::Scalar(0, 220, 0); // green
  } else if (status_label == "DEGRADED") {
    box_color = cv::Scalar(0, 200, 220); // amber
  } else {
    box_color = cv::Scalar(0, 0, 220); // red
  }

  // Find the top detection once for the header label.
  const NAVIGATION::RawDetection *top = nullptr;
  for (const auto &d : dets) {
    if (!top || d.confidence > top->confidence) {
      top = &d;
    }
  }

  // Draw every detection (thin box) so viewers can see runner-ups.
  for (const auto &d : dets) {
    const bool is_top = (&d == top);
    cv::rectangle(canvas, d.bbox_px, box_color, is_top ? 3 : 1);
    std::ostringstream lbl;
    lbl << d.class_name << " " << std::fixed;
    lbl.precision(2);
    lbl << d.confidence;
    int base = 0;
    cv::Size ts =
        cv::getTextSize(lbl.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
    int y0 = std::max(d.bbox_px.y - 4, ts.height + 4);
    cv::rectangle(canvas, cv::Point(d.bbox_px.x, y0 - ts.height - 4),
                  cv::Point(d.bbox_px.x + ts.width + 4, y0 + 2), box_color,
                  cv::FILLED);
    cv::putText(canvas, lbl.str(), cv::Point(d.bbox_px.x + 2, y0 - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
                cv::LINE_AA);
  }

  // Header strip with status + image_id so it's self-labeling.
  std::ostringstream hdr;
  hdr << status_label << "  " << image_id;
  if (top) {
    hdr << "  top=" << top->class_name << "(" << std::fixed;
    hdr.precision(2);
    hdr << top->confidence << ")";
  } else {
    hdr << "  no-detections";
  }
  int hbase = 0;
  cv::Size hs =
      cv::getTextSize(hdr.str(), cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &hbase);
  cv::rectangle(canvas, cv::Point(0, 0),
                cv::Point(hs.width + 12, hs.height + 10), box_color,
                cv::FILLED);
  cv::putText(canvas, hdr.str(), cv::Point(6, hs.height + 4),
              cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 1,
              cv::LINE_AA);

  try {
    if (!cv::imwrite(overlay_path.string(), canvas)) {
      std::cerr << "[overlay] imwrite failed for " << overlay_path << "\n";
    }
  } catch (const cv::Exception &e) {
    std::cerr << "[overlay] imwrite threw: " << e.what() << "\n";
  }
}

int run_nav_pipeline(const std::string &path2file,
                     const std::string &path2label,
                     const std::string &path2onnxmodel,
                     const std::string &mission, const std::string &instrument,
                     bool want_decision, const std::string &overlay_dir) {
  namespace fs = std::filesystem;

  if (mission.empty() || instrument.empty()) {
    std::cerr << "Error: --nav-fix / --nav-decision require --mission and "
                 "--instrument.\n";
    return 2;
  }
  if (path2file.empty() || path2onnxmodel.empty() || path2label.empty()) {
    std::cerr << "Error: -p, -m, and -l are required for nav modes.\n";
    return 2;
  }

  // Sanity check: the weight filename must reference this mission+instrument.
  // This is the §2 refusal-on-mismatch rule.
  const std::string weight_name = fs::path(path2onnxmodel).filename().string();
  if (!contains_ci(weight_name, mission) ||
      !contains_ci(weight_name, instrument)) {
    std::cerr << "Error: weight filename '" << weight_name
              << "' does not reference mission='" << mission << "' instrument='"
              << instrument
              << "'. REFUSING per brief §2 (per-mission weights contract).\n";
    return 3;
  }

  // Resolve workspace-relative artifact paths. We assume the binary is invoked
  // from the project root (build/ subdirectory also works because the current
  // build places the binary under build/ but CSVs are loaded from the absolute
  // or CWD-relative paths we construct here).
  const fs::path cwd = fs::current_path();
  fs::path instruments_csv = cwd / "artifacts" / "instruments.csv";
  fs::path state_csv = cwd / "artifacts" / "spacecraft_state.csv";
  fs::path manifest_csv =
      cwd / "artifacts" / (mission + "_" + instrument) / "image_manifest.csv";

  // If those files aren't at cwd, try one level up (binary run from build/).
  if (!fs::exists(instruments_csv)) {
    fs::path up = cwd.parent_path() / "artifacts" / "instruments.csv";
    if (fs::exists(up)) {
      instruments_csv = up;
      state_csv = cwd.parent_path() / "artifacts" / "spacecraft_state.csv";
      manifest_csv = cwd.parent_path() / "artifacts" /
                     (mission + "_" + instrument) / "image_manifest.csv";
    }
  }

  // Derive image_id from the file name (stem).
  const std::string image_id = fs::path(path2file).stem().string();

  // Load CSV records.
  NAVIGATION::InstrumentRecord instr;
  bool instr_ok = NAVIGATION::load_instrument_record(
      instruments_csv.string(), mission, instrument, instr);
  if (!instr_ok) {
    std::cerr << "Error: instruments.csv does not contain a verified row for ("
              << mission << ", " << instrument
              << "). REFUSING. csv=" << instruments_csv << "\n";
    return 4;
  }

  NAVIGATION::SpacecraftState state;
  bool state_ok = NAVIGATION::load_spacecraft_state(state_csv.string(), mission,
                                                    image_id, state);

  NAVIGATION::ManifestRow manifest;
  bool manifest_ok =
      NAVIGATION::load_manifest_row(manifest_csv.string(), image_id, manifest);

  // Cross-check: if manifest has mission/instrument and they disagree with
  // the flags, refuse.
  if (manifest_ok) {
    if (!manifest.mission.empty() && manifest.mission != mission) {
      std::cerr << "Error: manifest row mission='" << manifest.mission
                << "' disagrees with --mission='" << mission
                << "'. REFUSING per §2.\n";
      return 5;
    }
    if (!manifest.instrument.empty() && manifest.instrument != instrument) {
      std::cerr << "Error: manifest row instrument='" << manifest.instrument
                << "' disagrees with --instrument='" << instrument
                << "'. REFUSING per §2.\n";
      return 5;
    }
  }

  // Load image and run detection.
  DETECTION_IMAGE_PROCESSING::image_processing img;
  cv::Mat frame = img.get_image_from_file(path2file);
  if (frame.empty()) {
    std::cerr << "Error: cannot read image " << path2file << "\n";
    return 6;
  }
  auto detections = img.detect_raw(frame, path2label, path2onnxmodel);

  // Prepare output dir.
  fs::path out_dir =
      instruments_csv.parent_path() / (mission + "_" + instrument);
  fs::path fixes_dir = out_dir / "fixes";
  fs::path decisions_dir = out_dir / "decisions";
  if (want_decision)
    fs::create_directories(decisions_dir);
  else
    fs::create_directories(fixes_dir);

  int nominal = 0, degraded = 0, refused = 0;

  if (want_decision) {
    NAVIGATION::NavDecision dec = NAVIGATION::build_nav_decision(
        image_id, mission, instrument, frame.size(), detections, instr,
        instr_ok, state, state_ok, manifest, manifest_ok);
    std::string json = NAVIGATION::nav_decision_to_json(dec);

    fs::path out_path = decisions_dir / (image_id + ".json");
    std::ofstream ofs(out_path);
    ofs << json << "\n";
    ofs.close();

    if (dec.status == NAVIGATION::DecisionStatus::NOMINAL) {
      std::cout << json << "\n";
      ++nominal;
    } else if (dec.status == NAVIGATION::DecisionStatus::DEGRADED) {
      std::cerr << "[DEGRADED — DO NOT USE FOR NAVIGATION] " << image_id
                << "\n";
      std::cout << json << "\n";
      ++degraded;
    } else {
      std::cout << json << "\n";
      ++refused;
    }

    // Optional annotated PNG — never fatal.
    if (!overlay_dir.empty()) {
      fs::path od(overlay_dir);
      std::error_code ec;
      fs::create_directories(od, ec);
      const char *status_str =
          (dec.status == NAVIGATION::DecisionStatus::NOMINAL)    ? "NOMINAL"
          : (dec.status == NAVIGATION::DecisionStatus::DEGRADED) ? "DEGRADED"
                                                                 : "REFUSED";
      render_nav_overlay(frame, detections, status_str, image_id,
                         od / (image_id + "_overlay.png"));
    }
  } else {
    // --nav-fix: emit one fix per detection (usually just the top).
    if (detections.empty()) {
      std::cerr << "[nav-fix] no detections for " << image_id << "\n";
      ++refused;
    } else {
      // Use highest-confidence detection.
      auto best = std::max_element(detections.begin(), detections.end(),
                                   [](const NAVIGATION::RawDetection &a,
                                      const NAVIGATION::RawDetection &b) {
                                     return a.confidence < b.confidence;
                                   });
      auto fix_opt = NAVIGATION::compute_nav_fix(*best, instr, image_id,
                                                 frame.size(), state_ok);
      if (!fix_opt.has_value()) {
        std::cerr << "[nav-fix] REFUSED for " << image_id << "\n";
        ++refused;
      } else {
        std::string json = NAVIGATION::nav_fix_to_json(fix_opt.value());
        fs::path out_path = fixes_dir / (image_id + ".json");
        std::ofstream ofs(out_path);
        ofs << json << "\n";
        ofs.close();
        std::cout << json << "\n";
        ++nominal;
      }
    }
  }

  std::cerr << "NOMINAL: " << nominal << " | DEGRADED: " << degraded
            << " | REFUSED: " << refused << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  std::string path2file;
  std::string path2label = "";
  std::string path2onnxmodel = "";
  std::string mission;
  std::string instrument;
  bool helpFlag = false;
  bool imageFlag = false;
  bool videoFlag = false;
  bool pathFlag = false;
  bool webcamFlag = false;
  bool object_detection = false;
  bool nav_fix_flag = false;
  bool nav_decision_flag = false;
  std::string overlay_dir;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      helpFlag = true;
    } else if (arg == "-i" || arg == "--image") {
      imageFlag = true;
    } else if (arg == "-v" || arg == "--video") {
      videoFlag = true;
    } else if (arg == "-w" || arg == "--webcam") {
      webcamFlag = true;
    } else if (arg == "-d" || arg == "--detect") {
      object_detection = true;
    } else if (arg == "-p" || arg == "--path") {
      pathFlag = true;
      if (i + 1 < argc) {
        path2file = argv[i + 1];
        i++;
      } else {
        std::cout << "Error: Path argument requires a value.\n";
        return 1;
      }
    } else if (arg == "-l" || arg == "--label") {
      if (i + 1 < argc) {
        path2label = argv[i + 1];
        i++;
      } else {
        std::cout << "Error: Label argument requires a value.\n";
        return 1;
      }
    } else if (arg == "-m" || arg == "--model") {
      if (i + 1 < argc) {
        path2onnxmodel = argv[i + 1];
        i++;
      } else {
        std::cout << "Error: Model argument requires a value.\n";
        return 1;
      }
    } else if (arg == "--mission") {
      if (i + 1 < argc) {
        mission = argv[i + 1];
        i++;
      } else {
        std::cout << "Error: --mission requires a value.\n";
        return 1;
      }
    } else if (arg == "--instrument") {
      if (i + 1 < argc) {
        instrument = argv[i + 1];
        i++;
      } else {
        std::cout << "Error: --instrument requires a value.\n";
        return 1;
      }
    } else if (arg == "--nav-fix") {
      nav_fix_flag = true;
    } else if (arg == "--nav-decision") {
      nav_decision_flag = true;
    } else if (arg == "--overlay-dir") {
      if (i + 1 < argc) {
        overlay_dir = argv[i + 1];
        i++;
      } else {
        std::cout << "Error: --overlay-dir requires a value.\n";
        return 1;
      }
    } else {
      std::cout << "Error: Unknown argument '" << arg << "'.\n";
      return 1;
    }
  }

  if (helpFlag) {
    help_menu();
    return 0;
  }

  // Navigation modes route through a separate pipeline.
  if (nav_fix_flag || nav_decision_flag) {
    return run_nav_pipeline(path2file, path2label, path2onnxmodel, mission,
                            instrument, nav_decision_flag, overlay_dir);
  }

  if (!imageFlag && !videoFlag && !webcamFlag) {
    std::cout << "Error: Either -i/--image or -v/--video or -w/--webcam flag "
                 "must be specified.\n";
    return 1;
  }

  if (imageFlag) {
    if (!pathFlag) {
      std::cout << "Error: -i/--image flag requires -p/--path argument.\n";
      return 1;
    }
    DETECTION_IMAGE_PROCESSING::image_processing my_image;
    if (object_detection) {
      my_image.detect_objects_in_image(path2file, path2label, path2onnxmodel);
      return 0;
    }
    my_image.IMAGE_TEST_BLOCK(path2file);
    return 0;
  }

  if (videoFlag) {
    if (!pathFlag) {
      std::cout << "Error: -i/--image flag requires -p/--path argument.\n";
      return 1;
    }
    DETECTION_VIDEO_PROCESSING::video_processing my_video;
    if (object_detection) {
      my_video.run_object_detetion(path2file, path2label, path2onnxmodel);
      return 0;
    }

    my_video.display_video(path2file);
    return 0;
  }

  if (webcamFlag) {
    DETECTION_VIDEO_PROCESSING::video_processing my_webcam;
    if (object_detection) {
      my_webcam.run_object_detetion_webcam(path2label, path2onnxmodel);
      return 0;
    }
    my_webcam.display_webcam();
    return 0;
  }

  return 0;
}
