// Configuration of the robodog_nmpc solver: the contents of config/config.yaml as a plain struct.
#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace robodog_nmpc {

// One member per key of config.yaml (robodog_nmpc: ros__parameters: ...); see there for meaning and units.
// Foot order of all per-foot lists: FR, FL, RR, RL.
struct Config {
  struct Robot {
    double mass{};
    std::array<double, 3> inertia{};         // Ixx, Iyy, Izz (body frame)
    double nominal_height{};
    std::array<double, 8> hip_offsets_xy{};  // x, y per foot
  } robot;

  struct Horizon {
    int N{};
    double dt{};
  } horizon;

  struct Friction {
    std::string model;                       // polygon | cone | pyramid
    int polygon_sides{};
    double mu{}, f_min{}, f_max{};
  } friction;

  struct Cost {
    std::array<double, 3> w_velocity{};
    double w_yaw_rate{}, w_height{};
    std::array<double, 2> w_roll_pitch{};
    double w_force{}, terminal_weight_scale{};
  } cost;

  struct Solver {
    std::string nlp_solver_type, qp_solver, hessian_approx, integrator_type;
    int sim_method_num_stages{}, sim_method_num_steps{};
    std::string globalization;
    int nlp_solver_max_iter{}, qp_solver_iter_max{};
  } solver;

  struct Codegen {
    std::string solver_name, export_dir;
  } codegen;
};

// Thrown by loadConfig(); what() lists every problem found, one per line.
class ConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Read config.yaml and validate it with the same rules as validate_config() in scripts/srbd_ocp.py:
// every group and key present, no unknown or duplicate keys, types, ranges, list lengths, f_min < f_max.
// Additionally, all numbers must be finite. Throws ConfigError.
Config loadConfig(const std::string& path);

// Stale-solver check: compares the [codegen] settings the solver in this library was generated with
// (solver/srbd_nmpc_codegen_config.h) with `config`. Returns one line per difference, e.g.
// "horizon.dt: config.yaml 0.02, generated solver 0.025"; empty if they match.
std::vector<std::string> codegenMismatches(const Config& config);

namespace detail {
// One entry of the codegen-config header. Exposed for tests only.
struct CodegenEntry {
  std::string group, key;
  std::variant<long, double, std::string> value;
};
std::vector<CodegenEntry> generatedCodegenEntries();
std::vector<std::string> compareCodegen(const Config& config, const std::vector<CodegenEntry>& entries);
}  // namespace detail

}  // namespace robodog_nmpc
