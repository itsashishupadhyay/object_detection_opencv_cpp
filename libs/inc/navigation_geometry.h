#ifndef NAVIGATION_GEOMETRY_H
#define NAVIGATION_GEOMETRY_H

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace NAVIGATION {

// Raw detection emitted by the YOLO post-process path, without any drawing.
struct RawDetection {
  int class_id;
  std::string class_name;
  float confidence;
  cv::Rect bbox_px; // in original image pixel coordinates
};

// Verified instrument record, loaded from artifacts/instruments.csv.
struct InstrumentRecord {
  std::string mission;
  std::string instrument;
  int array_px_x = 0;
  int array_px_y = 0;
  double pixel_pitch_um = 0.0;
  double focal_length_mm = 0.0;
  double fov_deg_x = 0.0;
  double fov_deg_y = 0.0;
  double ifov_arcsec_per_px = 0.0;
  std::string boresight_frame;
  std::string source_citation;
  std::string verified_at;
  bool verified = false; // true only if row successfully parsed
};

// Spacecraft state row loaded from artifacts/spacecraft_state.csv, keyed by
// image_id.
struct SpacecraftState {
  std::string mission;
  std::string image_id;
  std::string timestamp_utc;
  double position_km[3] = {0, 0, 0};
  double velocity_kms[3] = {0, 0, 0};
  double attitude_q[4] = {0, 0, 0, 0}; // w, x, y, z
  std::string position_frame;
  std::string attitude_frame;
  std::string target_body_from_metadata;
  std::string spice_kernels_used;
  bool verified = false;
};

// Row from artifacts/<mission>_<instrument>/image_manifest.csv.
struct ManifestRow {
  std::string image_id;
  std::string mission;
  std::string instrument;
  std::string body;
  std::string filter;
  std::string timestamp_utc;
  std::string target_body_from_metadata;
  std::string local_path;
  bool instrument_verified = false;
  bool state_verified = false;
  bool found = false;
};

// §9 NavFix struct.
struct NavFix {
  std::string image_id;
  std::string mission;
  std::string instrument;
  std::string body;
  cv::Rect bbox_px;
  cv::Point2f center_offset_px;  // from optical center
  cv::Point2f center_offset_deg; // angular
  double range_km = 0.0;
  cv::Point3d xyz_camera_frame_km{0, 0, 0};
  double range_uncertainty_km = 0.0;
  bool instrument_verified = false;
  bool state_verified = false;
  // Which range regime fired: "small_angle" or "full_tan".
  std::string range_regime;
  // Body radius we used, in km.
  double body_radius_km_used = 0.0;
  // Whether body_radius_km_used came from the hardcoded table or a placeholder.
  bool body_radius_is_placeholder = false;
};

// Return a hardcoded equatorial radius in km for the given body label (the
// label used by the model's class_names). Sets is_placeholder=true if the
// body is unknown and the caller gets the 50 km placeholder.
double body_radius_km(const std::string &body_label, bool &is_placeholder);

// Compute a NavFix from one detection given a verified instrument record.
// Returns nullopt (and logs reason to stderr) if instrument_verified == false.
std::optional<NavFix> compute_nav_fix(const RawDetection &det,
                                      const InstrumentRecord &instr,
                                      const std::string &image_id,
                                      const cv::Size &image_size,
                                      bool state_verified);

// Simple CSV loaders. All return false and leave out-arg untouched on failure.
bool load_instrument_record(const std::string &csv_path,
                            const std::string &mission,
                            const std::string &instrument,
                            InstrumentRecord &out);

bool load_spacecraft_state(const std::string &csv_path,
                           const std::string &mission,
                           const std::string &image_id, SpacecraftState &out);

bool load_manifest_row(const std::string &csv_path, const std::string &image_id,
                       ManifestRow &out);

// Hand-written JSON writer for a NavFix (no external library).
std::string nav_fix_to_json(const NavFix &fix);

} // namespace NAVIGATION

#endif // NAVIGATION_GEOMETRY_H
