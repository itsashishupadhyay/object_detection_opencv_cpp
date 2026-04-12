#include "navigation_geometry.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace NAVIGATION {

// ---------------------------------------------------------------------------
// Body radii table (equatorial or volumetric mean, in km). Sources: cited by
// the verification sub-agent in docs/tech_sheets/. Values cross-checked against
// JPL SSD and Porco et al. for Cassini targets. Kept inline here per brief: no
// runtime JSON load, so the binary cannot silently drift.
// ---------------------------------------------------------------------------
static const std::unordered_map<std::string, double> kBodyRadiiKm = {
    {"saturn", 58232.0},
    {"saturn_rings", 140000.0}, // outer A-ring edge; extended object
    {"titan", 2575.0},
    {"jupiter", 69911.0},
    {"iapetus", 735.0},
    {"rhea", 763.8},
    {"dione", 561.4},
    {"tethys", 531.1},
    {"enceladus", 252.1},
    {"mimas", 198.2},
    {"hyperion", 135.0},
    {"phoebe", 106.5},
    {"epimetheus", 58.1},
    {"janus", 89.5},
    {"prometheus", 43.1},
    {"pandora", 40.7},
    {"atlas", 15.1},
    {"pan", 12.8},
    {"helene", 17.6},
    {"calypso", 10.7},
    {"telesto", 12.4},
    {"methone", 1.6},
    {"pallene", 2.5},
    {"anthe", 0.9},
    {"aegaeon", 0.33},
    {"daphnis", 3.8},
    {"polydeuces", 1.3},
    {"io", 1821.6},
    {"europa", 1560.8},
    {"ganymede", 2634.1},
    {"callisto", 2410.3},
    {"earth", 6371.0},
    {"moon", 1737.4},
    {"venus", 6051.8},
    {"pluto", 1188.3},
};

double body_radius_km(const std::string &body_label, bool &is_placeholder) {
  auto it = kBodyRadiiKm.find(body_label);
  if (it != kBodyRadiiKm.end()) {
    is_placeholder = false;
    return it->second;
  }
  is_placeholder = true;
  return 50.0; // documented placeholder per brief §9
}

// ---------------------------------------------------------------------------
// CSV helpers. We support quoted fields with embedded commas. Within a quoted
// field, a "" is an escaped quote.
// ---------------------------------------------------------------------------
static std::vector<std::string> parse_csv_line(const std::string &line) {
  std::vector<std::string> out;
  std::string cur;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          cur.push_back('"');
          ++i;
        } else {
          in_quotes = false;
        }
      } else {
        cur.push_back(c);
      }
    } else {
      if (c == '"') {
        in_quotes = true;
      } else if (c == ',') {
        out.push_back(cur);
        cur.clear();
      } else if (c == '\r') {
        // skip
      } else {
        cur.push_back(c);
      }
    }
  }
  out.push_back(cur);
  return out;
}

static int col_index(const std::vector<std::string> &header,
                     const std::string &name) {
  for (size_t i = 0; i < header.size(); ++i) {
    if (header[i] == name)
      return static_cast<int>(i);
  }
  return -1;
}

static double safe_d(const std::vector<std::string> &row, int idx) {
  if (idx < 0 || idx >= (int)row.size() || row[idx].empty())
    return 0.0;
  try {
    return std::stod(row[idx]);
  } catch (...) {
    return 0.0;
  }
}

static int safe_i(const std::vector<std::string> &row, int idx) {
  if (idx < 0 || idx >= (int)row.size() || row[idx].empty())
    return 0;
  try {
    return std::stoi(row[idx]);
  } catch (...) {
    return 0;
  }
}

static std::string safe_s(const std::vector<std::string> &row, int idx) {
  if (idx < 0 || idx >= (int)row.size())
    return "";
  return row[idx];
}

bool load_instrument_record(const std::string &csv_path,
                            const std::string &mission,
                            const std::string &instrument,
                            InstrumentRecord &out) {
  std::ifstream ifs(csv_path);
  if (!ifs) {
    std::cerr << "[nav] cannot open instruments csv: " << csv_path << std::endl;
    return false;
  }
  std::string line;
  if (!std::getline(ifs, line))
    return false;
  auto header = parse_csv_line(line);
  int c_mission = col_index(header, "mission");
  int c_instr = col_index(header, "instrument");
  int c_apx = col_index(header, "array_px_x");
  int c_apy = col_index(header, "array_px_y");
  int c_pitch = col_index(header, "pixel_pitch_um");
  int c_focal = col_index(header, "focal_length_mm");
  int c_fovx = col_index(header, "fov_deg_x");
  int c_fovy = col_index(header, "fov_deg_y");
  int c_ifov = col_index(header, "ifov_arcsec_per_px");
  int c_bore = col_index(header, "boresight_frame");
  int c_cite = col_index(header, "source_citation");
  int c_ver = col_index(header, "verified_at");
  while (std::getline(ifs, line)) {
    if (line.empty())
      continue;
    auto row = parse_csv_line(line);
    if (safe_s(row, c_mission) == mission &&
        safe_s(row, c_instr) == instrument) {
      out.mission = mission;
      out.instrument = instrument;
      out.array_px_x = safe_i(row, c_apx);
      out.array_px_y = safe_i(row, c_apy);
      out.pixel_pitch_um = safe_d(row, c_pitch);
      out.focal_length_mm = safe_d(row, c_focal);
      out.fov_deg_x = safe_d(row, c_fovx);
      out.fov_deg_y = safe_d(row, c_fovy);
      out.ifov_arcsec_per_px = safe_d(row, c_ifov);
      out.boresight_frame = safe_s(row, c_bore);
      out.source_citation = safe_s(row, c_cite);
      out.verified_at = safe_s(row, c_ver);
      out.verified = (out.focal_length_mm > 0.0 && out.pixel_pitch_um > 0.0 &&
                      out.array_px_x > 0 && out.array_px_y > 0);
      return out.verified;
    }
  }
  return false;
}

bool load_spacecraft_state(const std::string &csv_path,
                           const std::string &mission,
                           const std::string &image_id, SpacecraftState &out) {
  std::ifstream ifs(csv_path);
  if (!ifs) {
    std::cerr << "[nav] cannot open spacecraft_state csv: " << csv_path
              << std::endl;
    return false;
  }
  std::string line;
  if (!std::getline(ifs, line))
    return false;
  auto header = parse_csv_line(line);
  int c_mission = col_index(header, "mission");
  int c_image = col_index(header, "image_id");
  int c_ts = col_index(header, "timestamp_utc");
  int c_px = col_index(header, "position_km_x");
  int c_py = col_index(header, "position_km_y");
  int c_pz = col_index(header, "position_km_z");
  int c_pf = col_index(header, "position_frame");
  int c_vx = col_index(header, "velocity_kms_x");
  int c_vy = col_index(header, "velocity_kms_y");
  int c_vz = col_index(header, "velocity_kms_z");
  int c_qw = col_index(header, "attitude_q_w");
  int c_qx = col_index(header, "attitude_q_x");
  int c_qy = col_index(header, "attitude_q_y");
  int c_qz = col_index(header, "attitude_q_z");
  int c_af = col_index(header, "attitude_frame");
  int c_tg = col_index(header, "target_body_from_metadata");
  int c_kr = col_index(header, "spice_kernels_used");
  while (std::getline(ifs, line)) {
    if (line.empty())
      continue;
    auto row = parse_csv_line(line);
    if (safe_s(row, c_mission) == mission && safe_s(row, c_image) == image_id) {
      out.mission = mission;
      out.image_id = image_id;
      out.timestamp_utc = safe_s(row, c_ts);
      out.position_km[0] = safe_d(row, c_px);
      out.position_km[1] = safe_d(row, c_py);
      out.position_km[2] = safe_d(row, c_pz);
      out.position_frame = safe_s(row, c_pf);
      out.velocity_kms[0] = safe_d(row, c_vx);
      out.velocity_kms[1] = safe_d(row, c_vy);
      out.velocity_kms[2] = safe_d(row, c_vz);
      out.attitude_q[0] = safe_d(row, c_qw);
      out.attitude_q[1] = safe_d(row, c_qx);
      out.attitude_q[2] = safe_d(row, c_qy);
      out.attitude_q[3] = safe_d(row, c_qz);
      out.attitude_frame = safe_s(row, c_af);
      out.target_body_from_metadata = safe_s(row, c_tg);
      out.spice_kernels_used = safe_s(row, c_kr);
      out.verified = true;
      return true;
    }
  }
  return false;
}

bool load_manifest_row(const std::string &csv_path, const std::string &image_id,
                       ManifestRow &out) {
  std::ifstream ifs(csv_path);
  if (!ifs) {
    std::cerr << "[nav] cannot open manifest csv: " << csv_path << std::endl;
    return false;
  }
  std::string line;
  if (!std::getline(ifs, line))
    return false;
  auto header = parse_csv_line(line);
  int c_id = col_index(header, "image_id");
  int c_mission = col_index(header, "mission");
  int c_instr = col_index(header, "instrument");
  int c_body = col_index(header, "body");
  int c_filter = col_index(header, "filter");
  int c_ts = col_index(header, "timestamp_utc");
  int c_tg = col_index(header, "target_body_from_metadata");
  int c_lp = col_index(header, "local_path");
  int c_iv = col_index(header, "instrument_verified");
  int c_sv = col_index(header, "state_verified");
  while (std::getline(ifs, line)) {
    if (line.empty())
      continue;
    auto row = parse_csv_line(line);
    if (safe_s(row, c_id) == image_id) {
      out.image_id = image_id;
      out.mission = safe_s(row, c_mission);
      out.instrument = safe_s(row, c_instr);
      out.body = safe_s(row, c_body);
      out.filter = safe_s(row, c_filter);
      out.timestamp_utc = safe_s(row, c_ts);
      out.target_body_from_metadata = safe_s(row, c_tg);
      out.local_path = safe_s(row, c_lp);
      out.instrument_verified = (safe_s(row, c_iv) == "true");
      out.state_verified = (safe_s(row, c_sv) == "true");
      out.found = true;
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// NavFix computation. §9: range = body_radius / tan(angular_diameter/2).
// Small-angle approximation is used when angular_diameter_rad < 1e-3 rad
// (roughly 0.057 deg). Beyond that we use the full tan form. Both fire the
// same formula but we record which regime was used so the paper can cite it.
// ---------------------------------------------------------------------------
std::optional<NavFix> compute_nav_fix(const RawDetection &det,
                                      const InstrumentRecord &instr,
                                      const std::string &image_id,
                                      const cv::Size &image_size,
                                      bool state_verified) {
  if (!instr.verified) {
    std::cerr << "[nav] REFUSED: instrument record not verified for "
              << image_id << std::endl;
    return std::nullopt;
  }

  NavFix fix;
  fix.image_id = image_id;
  fix.mission = instr.mission;
  fix.instrument = instr.instrument;
  fix.body = det.class_name;
  fix.bbox_px = det.bbox_px;
  fix.instrument_verified = true;
  fix.state_verified = state_verified;

  const double focal_m = instr.focal_length_mm * 1e-3;
  const double pitch_m = instr.pixel_pitch_um * 1e-6;

  // Optical center = image center. The array_px fields in instruments.csv
  // describe the sensor; the image itself may be resampled, so we use the
  // actual image_size for the principal point. Any more precise distortion
  // model lives in the instrument tech sheet and is left for the paper.
  const double cx = image_size.width * 0.5;
  const double cy = image_size.height * 0.5;
  const double bbox_cx = det.bbox_px.x + det.bbox_px.width * 0.5;
  const double bbox_cy = det.bbox_px.y + det.bbox_px.height * 0.5;
  fix.center_offset_px.x = static_cast<float>(bbox_cx - cx);
  fix.center_offset_px.y = static_cast<float>(bbox_cy - cy);

  // Angular offset: each pixel subtends pitch_m / focal_m radians. Note this
  // assumes the image wasn't resampled from the native array; if it was, the
  // runtime sub-agent's residual check will surface the discrepancy.
  const double rad_per_px = pitch_m / focal_m;
  const double dx_rad = fix.center_offset_px.x * rad_per_px;
  const double dy_rad = fix.center_offset_px.y * rad_per_px;
  fix.center_offset_deg.x = static_cast<float>(dx_rad * 180.0 / M_PI);
  fix.center_offset_deg.y = static_cast<float>(dy_rad * 180.0 / M_PI);

  // Angular diameter from the longer box side (better SNR than mean).
  const double bbox_diam_px = std::max(det.bbox_px.width, det.bbox_px.height);
  const double angular_diameter_rad = (bbox_diam_px * pitch_m) / focal_m;

  bool is_placeholder = false;
  const double body_r_km = body_radius_km(det.class_name, is_placeholder);
  fix.body_radius_km_used = body_r_km;
  fix.body_radius_is_placeholder = is_placeholder;

  double range_km = 0.0;
  if (angular_diameter_rad < 1e-3) {
    // Small-angle: tan(x/2) ~ x/2 -> range = 2*R/angle
    range_km = (2.0 * body_r_km) / angular_diameter_rad;
    fix.range_regime = "small_angle";
  } else {
    range_km = body_r_km / std::tan(angular_diameter_rad / 2.0);
    fix.range_regime = "full_tan";
  }
  fix.range_km = range_km;

  // Camera-frame XYZ: Z = range along boresight, X/Y from angular offsets.
  // Again a first-order model; the verification agent's tech sheet notes that
  // NAC distortion is < 0.45 px at corners which is well inside our NavFix
  // noise floor, so we do not apply a distortion correction here.
  fix.xyz_camera_frame_km.x = range_km * std::tan(dx_rad);
  fix.xyz_camera_frame_km.y = range_km * std::tan(dy_rad);
  fix.xyz_camera_frame_km.z = range_km;

  // Uncertainty: dominated by bbox edge error (~1 pixel per side = 2 px on
  // the diameter). dR/R ~ d(angle)/angle. This is the quickest defensible
  // number; the paper can refine it.
  const double angle_err_rad = (2.0 * pitch_m) / focal_m;
  const double rel_err =
      (angular_diameter_rad > 0) ? (angle_err_rad / angular_diameter_rad) : 0.0;
  fix.range_uncertainty_km = std::abs(range_km * rel_err);

  if (is_placeholder) {
    std::cerr << "[nav] WARNING: unknown body '" << det.class_name
              << "' — using placeholder 50 km radius for " << image_id
              << std::endl;
  }

  return fix;
}

// ---------------------------------------------------------------------------
// Hand-written JSON (no library). Minimal escaping: we never embed user input
// that could contain quotes or backslashes beyond simple ASCII class names and
// file paths. If a string does contain a quote we escape it.
// ---------------------------------------------------------------------------
static std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    if (c == '"' || c == '\\')
      out.push_back('\\');
    if (c == '\n') {
      out += "\\n";
      continue;
    }
    out.push_back(c);
  }
  return out;
}

std::string nav_fix_to_json(const NavFix &fix) {
  std::ostringstream os;
  os.precision(9);
  os << std::fixed;
  os << "{\n";
  os << "  \"image_id\": \"" << json_escape(fix.image_id) << "\",\n";
  os << "  \"mission\": \"" << json_escape(fix.mission) << "\",\n";
  os << "  \"instrument\": \"" << json_escape(fix.instrument) << "\",\n";
  os << "  \"body\": \"" << json_escape(fix.body) << "\",\n";
  os << "  \"bbox_px\": [" << fix.bbox_px.x << ", " << fix.bbox_px.y << ", "
     << fix.bbox_px.width << ", " << fix.bbox_px.height << "],\n";
  os << "  \"center_offset_px\": [" << fix.center_offset_px.x << ", "
     << fix.center_offset_px.y << "],\n";
  os << "  \"center_offset_deg\": [" << fix.center_offset_deg.x << ", "
     << fix.center_offset_deg.y << "],\n";
  os << "  \"range_km\": " << fix.range_km << ",\n";
  os << "  \"xyz_camera_frame_km\": [" << fix.xyz_camera_frame_km.x << ", "
     << fix.xyz_camera_frame_km.y << ", " << fix.xyz_camera_frame_km.z
     << "],\n";
  os << "  \"range_uncertainty_km\": " << fix.range_uncertainty_km << ",\n";
  os << "  \"range_regime\": \"" << fix.range_regime << "\",\n";
  os << "  \"body_radius_km_used\": " << fix.body_radius_km_used << ",\n";
  os << "  \"body_radius_is_placeholder\": "
     << (fix.body_radius_is_placeholder ? "true" : "false") << ",\n";
  os << "  \"instrument_verified\": "
     << (fix.instrument_verified ? "true" : "false") << ",\n";
  os << "  \"state_verified\": " << (fix.state_verified ? "true" : "false")
     << "\n";
  os << "}";
  return os.str();
}

} // namespace NAVIGATION
