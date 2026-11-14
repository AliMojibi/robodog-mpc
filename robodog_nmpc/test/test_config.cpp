// Tests of loadConfig(): the real config.yaml loads with the right values, and every kind of invalid file
// is rejected with a clear message. The invalid-value cases and message fragments are the same as in
// test_generate_solver.py (Python validator), so both validators are checked against one list.
#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "robodog_nmpc/config.hpp"

using robodog_nmpc::Config;
using robodog_nmpc::ConfigError;
using robodog_nmpc::loadConfig;

namespace {

std::string readFile(const std::string& path) {
  std::ifstream f(path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

const std::string& baseText() {
  static const std::string text = readFile(CONFIG_YAML);
  return text;
}

// The config text with the value of `key` (first line "<indent>key: ...") replaced; comments are kept.
std::string withValue(const std::string& key, const std::string& value, std::string text = baseText()) {
  const std::regex line("(\\n\\s+" + key + ":)[^#\\n]*");
  std::smatch m;
  if (!std::regex_search(text, m, line)) throw std::runtime_error("test bug: key not found: " + key);
  return text.replace(m.position(0), m.length(0), m.str(1) + " " + value + "   ");
}

std::string writeTemp(const std::string& text) {
  static int counter = 0;
  const auto path = std::filesystem::temp_directory_path() /
                    ("robodog_nmpc_test_config_" + std::to_string(::getpid()) + "_" + std::to_string(counter++) + ".yaml");
  std::ofstream(path) << text;
  return path.string();
}

// Loading `text` must fail with a message containing every fragment; returns the message.
std::string expectRejected(const std::string& text, const std::vector<std::string>& fragments) {
  const std::string path = writeTemp(text);
  std::string msg;
  try {
    loadConfig(path);
    ADD_FAILURE() << "invalid config was accepted";
  } catch (const ConfigError& e) {
    msg = e.what();
  }
  std::filesystem::remove(path);
  for (const auto& f : fragments) EXPECT_NE(msg.find(f), std::string::npos) << "missing '" << f << "' in:\n" << msg;
  return msg;
}

Config load(const std::string& text) {
  const std::string path = writeTemp(text);
  Config c = loadConfig(path);
  std::filesystem::remove(path);
  return c;
}

}  // namespace

TEST(Config, RealFileLoadsWithAllValues) {
  const Config c = loadConfig(CONFIG_YAML);
  EXPECT_EQ(c.robot.mass, 43.0);
  EXPECT_EQ(c.robot.inertia, (std::array<double, 3>{0.41, 2.1, 2.1}));
  EXPECT_EQ(c.robot.nominal_height, 0.45);
  EXPECT_EQ(c.robot.hip_offsets_xy, (std::array<double, 8>{0.30, -0.15, 0.30, 0.15, -0.30, -0.15, -0.30, 0.15}));
  EXPECT_EQ(c.horizon.N, 16);
  EXPECT_EQ(c.horizon.dt, 0.025);
  EXPECT_EQ(c.friction.model, "polygon");
  EXPECT_EQ(c.friction.polygon_sides, 16);
  EXPECT_EQ(c.friction.mu, 0.6);
  EXPECT_EQ(c.friction.f_min, 10.0);
  EXPECT_EQ(c.friction.f_max, 666.0);
  EXPECT_EQ(c.cost.w_velocity, (std::array<double, 3>{1.0, 1.0, 1.0}));
  EXPECT_EQ(c.cost.w_yaw_rate, 1.0);
  EXPECT_EQ(c.cost.w_height, 50.0);
  EXPECT_EQ(c.cost.w_roll_pitch, (std::array<double, 2>{1.0, 1.0}));
  EXPECT_EQ(c.cost.w_force, 1.0e-6);
  EXPECT_EQ(c.cost.terminal_weight_scale, 1.0);
  EXPECT_EQ(c.solver.nlp_solver_type, "SQP_RTI");
  EXPECT_EQ(c.solver.qp_solver, "PARTIAL_CONDENSING_HPIPM");
  EXPECT_EQ(c.solver.hessian_approx, "GAUSS_NEWTON");
  EXPECT_EQ(c.solver.integrator_type, "ERK");
  EXPECT_EQ(c.solver.sim_method_num_stages, 4);
  EXPECT_EQ(c.solver.sim_method_num_steps, 1);
  EXPECT_EQ(c.solver.globalization, "FIXED_STEP");
  EXPECT_EQ(c.solver.nlp_solver_max_iter, 50);
  EXPECT_EQ(c.solver.qp_solver_iter_max, 50);
  EXPECT_EQ(c.codegen.solver_name, "srbd_nmpc");
  EXPECT_EQ(c.codegen.export_dir, "solver");
}

TEST(Config, ValidChangesAreRead) {
  EXPECT_EQ(load(withValue("mass", "30.5")).robot.mass, 30.5);
  EXPECT_EQ(load(withValue("mass", "30")).robot.mass, 30.0);              // an integer is a valid real
  EXPECT_EQ(load(withValue("model", "pyramid")).friction.model, "pyramid");
  EXPECT_EQ(load(withValue("f_min", "0.0")).friction.f_min, 0.0);           // >= 0 includes 0
  EXPECT_EQ(load(withValue("sim_method_num_stages", "1")).solver.sim_method_num_stages, 1);
}

TEST(Config, InvalidValuesSameAsPython) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {withValue("model", "polyogn"), "friction.model: must be one of"},
      {withValue("w_velocity", "[1.0, 1.0]"), "cost.w_velocity: must be a list of 3 entries"},
      {withValue("inertia", "[-0.41, 2.1, 2.1]"), "robot.inertia: entry 0: must be a number > 0"},
      {withValue("N", "0"), "horizon.N: must be an integer >= 1"},
      {withValue("N", "16.5"), "horizon.N: must be an integer >= 1"},
      {withValue("polygon_sides", "2"), "friction.polygon_sides: must be an integer >= 3"},
      {withValue("f_min", "700.0"), "f_min (700.0) must be smaller than f_max (666.0)"},
      {withValue("w_force", "0.0"), "cost.w_force: must be a number > 0"},
      {withValue("nlp_solver_type", "SQP_FAST"), "solver.nlp_solver_type: must be one of"},
      {withValue("solver_name", "srbd-nmpc"), "codegen.solver_name: must be a valid C identifier"},
      {withValue("mass", "True"), "robot.mass: must be a number > 0"},
  };
  for (const auto& [text, fragment] : cases) {
    SCOPED_TRACE(fragment);
    expectRejected(text, {"invalid config.yaml", fragment});
  }
}

TEST(Config, MoreInvalidValues) {
  expectRejected(withValue("mass", ""), {"robot.mass: must be a number > 0, got null"});
  expectRejected(withValue("mass", "heavy"), {"robot.mass: must be a number > 0, got heavy"});
  expectRejected(withValue("N", "true"), {"horizon.N: must be an integer >= 1"});
  expectRejected(withValue("N", "99999999999"), {"horizon.N: must be an integer >= 1"});   // int overflow
  expectRejected(withValue("sim_method_num_stages", "5"), {"solver.sim_method_num_stages: must be an integer in 1..4"});
  expectRejected(withValue("f_min", "-1.0"), {"friction.f_min: must be a number >= 0"});
  expectRejected(withValue("f_min", "666.0"), {"f_min (666.0) must be smaller than f_max (666.0)"});  // equal
  expectRejected(withValue("w_velocity", "1.0"), {"cost.w_velocity: must be a list of 3 entries, got 1.0"});
  expectRejected(withValue("w_roll_pitch", "[1.0, x]"), {"cost.w_roll_pitch: entry 1: must be a number >= 0"});
  expectRejected(withValue("export_dir", "' '"), {"codegen.export_dir: must be a non-empty folder name"});
}

TEST(Config, NonFiniteNumbersRejected) {  // PyYAML and yaml-cpp both read .inf / .nan as numbers
  expectRejected(withValue("mass", ".inf"), {"robot.mass: must be a number > 0, got .inf"});
  expectRejected(withValue("w_height", ".nan"), {"cost.w_height: must be a number >= 0, got .nan"});
  expectRejected(withValue("inertia", "[0.41, .inf, 2.1]"), {"robot.inertia: entry 1: must be a number > 0"});
}

TEST(Config, DuplicateKeysRejected) {  // PyYAML keeps the last value, yaml-cpp the first: never accept them
  std::string text = baseText();
  text.replace(text.find("      mu: 0.6"), 0, "      mu: 0.5\n");
  expectRejected(text, {"friction.mu: appears more than once"});
  text = baseText();
  text.replace(text.find("    horizon:"), 0, "    horizon:\n      N: 10\n      dt: 0.01\n");
  expectRejected(text, {"horizon: appears more than once"});
}

TEST(Config, TypoAndMissingKeys) {
  std::string text = baseText();
  text.replace(text.find("w_height:"), 9, "w_heigth:");
  expectRejected(text, {"cost.w_heigth: unknown key (typo?)", "cost.w_height: missing"});
  text = baseText();
  text.replace(text.find("    horizon:"), 12, "    horizonx:");
  expectRejected(text, {"unknown group 'horizonx' (typo?)", "missing group 'horizon'"});
}

TEST(Config, AllErrorsReportedAtOnce) {
  const std::string msg = expectRejected(withValue("dt", "0.0", withValue("N", "-3", withValue("inertia", "[1.0, 2.0]"))),
                                         {"horizon.N:", "horizon.dt:", "robot.inertia:"});
  EXPECT_EQ(std::count(msg.begin(), msg.end(), '\n'), 3) << msg;   // header line + exactly 3 problems
}

TEST(Config, FileAndLayoutErrors) {
  try {
    loadConfig("/nonexistent/config.yaml");
    ADD_FAILURE() << "missing file was accepted";
  } catch (const ConfigError& e) {
    EXPECT_NE(std::string(e.what()).find("/nonexistent/config.yaml: cannot open the file"), std::string::npos);
  }
  std::string text = baseText();
  text.replace(text.find("    horizon:"), 12, "   horizon:  [");
  expectRejected(text, {"not valid YAML"});
  text = baseText();
  text.replace(text.find("ros__parameters"), 15, "parameters");
  expectRejected(text, {"expected the layout 'robodog_nmpc: ros__parameters: ...'"});
  expectRejected("", {"expected the layout"});
  expectRejected("robodog_nmpc:\n  ros__parameters: [1, 2]\n", {"must contain the parameter groups"});
}

TEST(Config, FormatDifferenceToPython) {
  // Documented in config.yaml: write 1.0e-6. yaml-cpp also reads 1e-6 correctly; PyYAML would read it as
  // text and generate_solver.py rejects it, so a wrong value can never be used silently.
  EXPECT_EQ(load(withValue("w_force", "1e-6")).cost.w_force, 1.0e-6);
}
