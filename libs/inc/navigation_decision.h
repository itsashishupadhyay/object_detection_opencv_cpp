#ifndef NAVIGATION_DECISION_H
#define NAVIGATION_DECISION_H

#include "navigation_geometry.h"
#include <optional>
#include <string>
#include <vector>

namespace NAVIGATION {

enum class DecisionStatus { NOMINAL, DEGRADED, REFUSED };

// §10 NavDecision.
struct NavDecision {
  DecisionStatus status = DecisionStatus::REFUSED;
  std::string image_id;
  std::string mission;
  std::string instrument;

  // The NavFix is only populated when it could be computed at all. REFUSED
  // cases leave this empty.
  std::optional<NavFix> fix;

  // Populated on NOMINAL only.
  std::string action = "HOLD";
  double delta_v_mps[3] = {0, 0, 0};
  std::optional<double> time_to_closest_approach_s;
  std::optional<double> predicted_miss_distance_km;
  double decision_confidence = 0.0;

  // Provenance, always populated when we have instrument / state records.
  std::string instrument_record_source;
  std::string spice_kernels_used;

  // Residual vs SPICE reconstruction — filled later by runtime sub-agent.
  // Always null on emit from this module.
  std::optional<double> range_residual_vs_spice_km;

  // Human-readable explanation.
  std::string reasoning;

  // DEGRADED bookkeeping.
  std::vector<std::string> missing_inputs;
  std::string warning;
};

// Build a NavDecision from a single top detection + the already-loaded
// verification records. confidence_floor defaults to 0.35 per §10.1.
NavDecision build_nav_decision(const std::string &image_id,
                               const std::string &mission,
                               const std::string &instrument,
                               const cv::Size &image_size,
                               const std::vector<RawDetection> &detections,
                               const InstrumentRecord &instr, bool instr_ok,
                               const SpacecraftState &state, bool state_ok,
                               const ManifestRow &manifest, bool manifest_ok,
                               double confidence_floor = 0.35);

// JSON serializer for a NavDecision, matching the schema in the brief.
std::string nav_decision_to_json(const NavDecision &dec);

const char *status_to_str(DecisionStatus s);

} // namespace NAVIGATION

#endif // NAVIGATION_DECISION_H
