#include "navigation_decision.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>

namespace NAVIGATION {

const char *status_to_str(DecisionStatus s) {
  switch (s) {
  case DecisionStatus::NOMINAL:
    return "NOMINAL";
  case DecisionStatus::DEGRADED:
    return "DEGRADED";
  case DecisionStatus::REFUSED:
    return "REFUSED";
  }
  return "REFUSED";
}

static std::string to_lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Pick the single detection with the highest confidence. If the vector is
// empty, returns nullopt.
static std::optional<RawDetection>
top_detection(const std::vector<RawDetection> &dets) {
  if (dets.empty())
    return std::nullopt;
  auto it = std::max_element(dets.begin(), dets.end(),
                             [](const RawDetection &a, const RawDetection &b) {
                               return a.confidence < b.confidence;
                             });
  return *it;
}

NavDecision build_nav_decision(
    const std::string &image_id, const std::string &mission,
    const std::string &instrument, const cv::Size &image_size,
    const std::vector<RawDetection> &detections, const InstrumentRecord &instr,
    bool instr_ok, const SpacecraftState &state, bool state_ok,
    const ManifestRow &manifest, bool manifest_ok, double confidence_floor) {
  NavDecision dec;
  dec.image_id = image_id;
  dec.mission = mission;
  dec.instrument = instrument;
  dec.instrument_record_source =
      instr_ok ? instr.source_citation : std::string("");
  dec.spice_kernels_used =
      state_ok ? state.spice_kernels_used : std::string("");

  // §10.3 REFUSED short-circuit: no instrument at all, or no detections.
  if (!instr_ok) {
    dec.status = DecisionStatus::REFUSED;
    dec.reasoning = "instrument record not verified — cannot compute NavFix";
    return dec;
  }

  auto top = top_detection(detections);
  if (!top.has_value()) {
    dec.status = DecisionStatus::REFUSED;
    dec.reasoning = "no detections above score threshold";
    return dec;
  }

  // Try to compute a NavFix. Even degraded records get a range_km if we can.
  auto fix_opt =
      compute_nav_fix(top.value(), instr, image_id, image_size, state_ok);
  if (!fix_opt.has_value()) {
    dec.status = DecisionStatus::REFUSED;
    dec.reasoning = "compute_nav_fix refused (likely instrument check)";
    return dec;
  }

  // Check NOMINAL preconditions.
  std::vector<std::string> missing;
  if (!state_ok)
    missing.push_back("spacecraft_state");
  if (!manifest_ok)
    missing.push_back("image_manifest_row");
  if (top->confidence < confidence_floor) {
    std::ostringstream os;
    os << "detection_confidence_" << top->confidence << "_below_floor_"
       << confidence_floor;
    missing.push_back(os.str());
  }

  // Class-vs-metadata match. target_body_from_metadata comes from manifest,
  // falls back to spacecraft_state row. We compare case-insensitively.
  std::string expected_body;
  if (manifest_ok && !manifest.target_body_from_metadata.empty())
    expected_body = manifest.target_body_from_metadata;
  else if (state_ok && !state.target_body_from_metadata.empty())
    expected_body = state.target_body_from_metadata;
  if (!expected_body.empty()) {
    if (to_lower_copy(expected_body) != to_lower_copy(top->class_name)) {
      missing.push_back(std::string("class_metadata_mismatch_expected_") +
                        expected_body + "_got_" + top->class_name);
    }
  } else {
    missing.push_back("no_target_body_metadata");
  }

  if (missing.empty()) {
    dec.status = DecisionStatus::NOMINAL;
    dec.fix = fix_opt;
    dec.action = "HOLD";
    dec.delta_v_mps[0] = 0;
    dec.delta_v_mps[1] = 0;
    dec.delta_v_mps[2] = 0;
    dec.time_to_closest_approach_s = std::nullopt;
    dec.predicted_miss_distance_km = std::nullopt;
    dec.decision_confidence = static_cast<double>(top->confidence);
    dec.reasoning =
        "NOMINAL: instrument verified, SPICE state verified, class matches "
        "metadata. No planned trajectory provided, delta_v defaulted to zero.";
    return dec;
  }

  // DEGRADED: keep the fix but strip XYZ / action per §10.2.
  dec.status = DecisionStatus::DEGRADED;
  NavFix stripped = fix_opt.value();
  stripped.xyz_camera_frame_km = cv::Point3d(0, 0, 0);
  dec.fix = stripped;
  dec.missing_inputs = missing;
  dec.warning = "DO NOT USE FOR NAVIGATION";
  dec.decision_confidence = static_cast<double>(top->confidence);
  std::ostringstream why;
  why << "DEGRADED: ";
  for (size_t i = 0; i < missing.size(); ++i) {
    if (i)
      why << "; ";
    why << missing[i];
  }
  dec.reasoning = why.str();
  return dec;
}

// ---------------------------------------------------------------------------
// JSON writer. Mirrors the schema in the task contract.
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

static std::string opt_num(const std::optional<double> &v) {
  if (!v.has_value())
    return "null";
  std::ostringstream os;
  os.precision(6);
  os << std::fixed << v.value();
  return os.str();
}

std::string nav_decision_to_json(const NavDecision &dec) {
  std::ostringstream os;
  os.precision(6);
  os << std::fixed;
  os << "{\n";
  os << "  \"status\": \"" << status_to_str(dec.status) << "\",\n";
  os << "  \"image_id\": \"" << json_escape(dec.image_id) << "\",\n";
  os << "  \"mission\": \"" << json_escape(dec.mission) << "\",\n";
  os << "  \"instrument\": \"" << json_escape(dec.instrument) << "\",\n";

  if (dec.fix.has_value()) {
    const NavFix &f = dec.fix.value();
    os << "  \"fix\": {\n";
    os << "    \"body\": \"" << json_escape(f.body) << "\",\n";
    os << "    \"bbox_px\": [" << f.bbox_px.x << ", " << f.bbox_px.y << ", "
       << f.bbox_px.width << ", " << f.bbox_px.height << "],\n";
    os << "    \"center_offset_px\": [" << f.center_offset_px.x << ", "
       << f.center_offset_px.y << "],\n";
    os << "    \"center_offset_deg\": [" << f.center_offset_deg.x << ", "
       << f.center_offset_deg.y << "],\n";
    os << "    \"range_km\": " << f.range_km << ",\n";
    if (dec.status == DecisionStatus::NOMINAL) {
      os << "    \"xyz_camera_frame_km\": [" << f.xyz_camera_frame_km.x << ", "
         << f.xyz_camera_frame_km.y << ", " << f.xyz_camera_frame_km.z
         << "],\n";
    } else {
      os << "    \"xyz_camera_frame_km\": null,\n";
    }
    os << "    \"range_uncertainty_km\": " << f.range_uncertainty_km << ",\n";
    os << "    \"range_regime\": \"" << f.range_regime << "\",\n";
    os << "    \"body_radius_km_used\": " << f.body_radius_km_used << ",\n";
    os << "    \"body_radius_is_placeholder\": "
       << (f.body_radius_is_placeholder ? "true" : "false") << ",\n";
    os << "    \"instrument_verified\": "
       << (f.instrument_verified ? "true" : "false") << ",\n";
    os << "    \"state_verified\": " << (f.state_verified ? "true" : "false")
       << "\n";
    os << "  },\n";
  } else {
    os << "  \"fix\": null,\n";
  }

  if (dec.status == DecisionStatus::NOMINAL) {
    os << "  \"action\": \"" << dec.action << "\",\n";
    os << "  \"delta_v_mps\": [" << dec.delta_v_mps[0] << ", "
       << dec.delta_v_mps[1] << ", " << dec.delta_v_mps[2] << "],\n";
    os << "  \"time_to_closest_approach_s\": "
       << opt_num(dec.time_to_closest_approach_s) << ",\n";
    os << "  \"predicted_miss_distance_km\": "
       << opt_num(dec.predicted_miss_distance_km) << ",\n";
  } else {
    os << "  \"action\": null,\n";
    os << "  \"delta_v_mps\": null,\n";
    os << "  \"time_to_closest_approach_s\": null,\n";
    os << "  \"predicted_miss_distance_km\": null,\n";
  }

  os << "  \"decision_confidence\": " << dec.decision_confidence << ",\n";
  os << "  \"instrument_record_source\": \""
     << json_escape(dec.instrument_record_source) << "\",\n";
  os << "  \"spice_kernels_used\": \"" << json_escape(dec.spice_kernels_used)
     << "\",\n";
  os << "  \"range_residual_vs_spice_km\": "
     << opt_num(dec.range_residual_vs_spice_km) << ",\n";

  if (dec.status == DecisionStatus::DEGRADED) {
    os << "  \"missing_inputs\": [";
    for (size_t i = 0; i < dec.missing_inputs.size(); ++i) {
      if (i)
        os << ", ";
      os << "\"" << json_escape(dec.missing_inputs[i]) << "\"";
    }
    os << "],\n";
    os << "  \"warning\": \"" << json_escape(dec.warning) << "\",\n";
  }

  os << "  \"reasoning\": \"" << json_escape(dec.reasoning) << "\"\n";
  os << "}";
  return os.str();
}

} // namespace NAVIGATION
