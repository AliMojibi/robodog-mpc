// Loading and validation of config.yaml. Same rules and messages as validate_config() in scripts/srbd_ocp.py,
// plus: numbers must be finite, keys must not be repeated (PyYAML keeps the last, yaml-cpp the first).
#include "robodog_nmpc/config.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <functional>
#include <map>
#include <regex>
#include <set>
#include <vector>

namespace robodog_nmpc {
namespace {

// A check validates one YAML value; on success it stores the value in the Config and returns "",
// otherwise it returns the problem.
using Check = std::function<std::string(const YAML::Node&)>;
using Condition = std::function<bool(double)>;

struct Key {
  std::string name;
  Check check;
};
struct Group {
  std::string name;
  std::vector<Key> keys;
};

std::string text(const YAML::Node& n) {  // a value as written in the file, for error messages
  if (n.IsScalar()) return n.Scalar();
  if (n.IsNull()) return "null";
  YAML::Emitter e;
  e << YAML::Flow << n;
  return e.c_str();
}

Check number(double& out, Condition cond, const std::string& range) {
  return [&out, cond, range](const YAML::Node& n) -> std::string {
    double v;
    if (!YAML::convert<double>::decode(n, v) || !std::isfinite(v) || !cond(v))
      return "must be a number" + (range.empty() ? "" : " " + range) + ", got " + text(n);
    out = v;
    return "";
  };
}

Check integer(int& out, std::function<bool(int)> cond, const std::string& range) {
  return [&out, cond, range](const YAML::Node& n) -> std::string {
    int v;
    if (!YAML::convert<int>::decode(n, v) || !cond(v)) return "must be an integer " + range + ", got " + text(n);
    out = v;
    return "";
  };
}

template <std::size_t N>
Check list(std::array<double, N>& out, Condition cond, const std::string& range) {
  return [&out, cond, range](const YAML::Node& n) -> std::string {
    if (!n.IsSequence() || n.size() != N)
      return "must be a list of " + std::to_string(N) + " entries, got " + text(n);
    for (std::size_t i = 0; i < N; ++i) {
      std::string msg = number(out[i], cond, range)(n[i]);
      if (!msg.empty()) return "entry " + std::to_string(i) + ": " + msg;
    }
    return "";
  };
}

Check choice(std::string& out, const std::vector<std::string>& options) {
  return [&out, options](const YAML::Node& n) -> std::string {
    std::string all;
    for (const auto& o : options) {
      if (n.IsScalar() && n.Scalar() == o) {
        out = o;
        return "";
      }
      all += (all.empty() ? "'" : ", '") + o + "'";
    }
    return "must be one of [" + all + "], got " + text(n);
  };
}

Check identifier(std::string& out) {
  return [&out](const YAML::Node& n) -> std::string {
    if (!n.IsScalar() || !std::regex_match(n.Scalar(), std::regex("[A-Za-z_][A-Za-z0-9_]*")))
      return "must be a valid C identifier (letters, digits, _), got " + text(n);
    out = n.Scalar();
    return "";
  };
}

Check folder(std::string& out) {
  return [&out](const YAML::Node& n) -> std::string {
    if (!n.IsScalar() || n.Scalar().find_first_not_of(" \t") == std::string::npos)
      return "must be a non-empty folder name, got " + text(n);
    out = n.Scalar();
    return "";
  };
}

// The entries of a map by key; repeated keys are reported (with `prefix`, e.g. "robot.") instead of stored.
std::map<std::string, YAML::Node> entries(const YAML::Node& map, const std::string& prefix,
                                          std::set<std::string>& duplicates) {
  std::map<std::string, YAML::Node> out;
  for (const auto& kv : map) {
    const std::string key = kv.first.as<std::string>("");
    if (!out.emplace(key, kv.second).second) duplicates.insert(prefix + key);
  }
  return out;
}

}  // namespace

Config loadConfig(const std::string& path) {
  YAML::Node doc;
  try {
    doc = YAML::LoadFile(path);
  } catch (const YAML::BadFile&) {
    throw ConfigError(path + ": cannot open the file");
  } catch (const YAML::ParserException& e) {
    throw ConfigError(path + ": not valid YAML:\n" + e.what());
  }
  const YAML::Node& root = doc;  // const: operator[] must never insert into the document
  if (!root.IsMap() || !root["robodog_nmpc"].IsMap() || !root["robodog_nmpc"]["ros__parameters"])
    throw ConfigError(path + ": expected the layout 'robodog_nmpc: ros__parameters: ...'");
  const YAML::Node params = root["robodog_nmpc"]["ros__parameters"];
  if (!params.IsMap())
    throw ConfigError("config: 'robodog_nmpc: ros__parameters:' must contain the parameter groups");

  Config c;
  const Condition positive = [](double v) { return v > 0; };
  const Condition nonneg = [](double v) { return v >= 0; };
  const Condition any = [](double) { return true; };
  const auto at_least = [](int lo) { return [lo](int v) { return v >= lo; }; };

  const std::vector<Group> schema = {
      {"robot",
       {{"mass", number(c.robot.mass, positive, "> 0")},
        {"inertia", list(c.robot.inertia, positive, "> 0")},
        {"nominal_height", number(c.robot.nominal_height, positive, "> 0")},
        {"hip_offsets_xy", list(c.robot.hip_offsets_xy, any, "")}}},
      {"horizon",
       {{"N", integer(c.horizon.N, at_least(1), ">= 1")},
        {"dt", number(c.horizon.dt, positive, "> 0")}}},
      {"friction",
       {{"model", choice(c.friction.model, {"polygon", "cone", "pyramid"})},
        {"polygon_sides", integer(c.friction.polygon_sides, at_least(3), ">= 3")},
        {"mu", number(c.friction.mu, positive, "> 0")},
        {"f_min", number(c.friction.f_min, nonneg, ">= 0")},
        {"f_max", number(c.friction.f_max, positive, "> 0")}}},
      {"cost",
       {{"w_velocity", list(c.cost.w_velocity, nonneg, ">= 0")},
        {"w_yaw_rate", number(c.cost.w_yaw_rate, nonneg, ">= 0")},
        {"w_height", number(c.cost.w_height, nonneg, ">= 0")},
        {"w_roll_pitch", list(c.cost.w_roll_pitch, nonneg, ">= 0")},
        {"w_force", number(c.cost.w_force, positive, "> 0")},  // > 0: keeps the QP Hessian regular
        {"terminal_weight_scale", number(c.cost.terminal_weight_scale, nonneg, ">= 0")}}},
      {"solver",
       {{"nlp_solver_type", choice(c.solver.nlp_solver_type, {"SQP_RTI", "SQP"})},
        {"qp_solver", choice(c.solver.qp_solver, {"PARTIAL_CONDENSING_HPIPM", "FULL_CONDENSING_HPIPM",
                                                  "FULL_CONDENSING_QPOASES", "FULL_CONDENSING_DAQP"})},
        {"hessian_approx", choice(c.solver.hessian_approx, {"GAUSS_NEWTON", "EXACT"})},
        {"integrator_type", choice(c.solver.integrator_type, {"ERK", "IRK"})},
        {"sim_method_num_stages",
         integer(c.solver.sim_method_num_stages, [](int v) { return 1 <= v && v <= 4; }, "in 1..4")},
        {"sim_method_num_steps", integer(c.solver.sim_method_num_steps, at_least(1), ">= 1")},
        {"globalization", choice(c.solver.globalization, {"FIXED_STEP", "MERIT_BACKTRACKING"})},
        {"nlp_solver_max_iter", integer(c.solver.nlp_solver_max_iter, at_least(1), ">= 1")},
        {"qp_solver_iter_max", integer(c.solver.qp_solver_iter_max, at_least(1), ">= 1")}}},
      {"codegen",
       {{"solver_name", identifier(c.codegen.solver_name)},
        {"export_dir", folder(c.codegen.export_dir)}}},
  };

  // Same order as Python: unknown groups, then per group: missing group / unknown keys / missing keys / values.
  std::vector<std::string> errors;
  std::set<std::string> duplicates;
  const auto groups = entries(params, "", duplicates);
  std::set<std::string> known_groups;
  for (const auto& g : schema) known_groups.insert(g.name);
  for (const auto& [name, node] : groups)
    if (!known_groups.count(name)) errors.push_back("unknown group '" + name + "' (typo?)");

  for (const auto& group : schema) {
    const auto g = groups.find(group.name);
    if (g == groups.end() || !g->second.IsMap()) {
      errors.push_back("missing group '" + group.name + "'");
      continue;
    }
    const auto values = entries(g->second, group.name + ".", duplicates);
    std::set<std::string> known_keys;
    for (const auto& k : group.keys) known_keys.insert(k.name);
    for (const auto& [name, node] : values)
      if (!known_keys.count(name)) errors.push_back(group.name + "." + name + ": unknown key (typo?)");
    for (const auto& key : group.keys) {
      const auto v = values.find(key.name);
      const std::string msg = v == values.end() ? "missing" : key.check(v->second);
      if (!msg.empty()) errors.push_back(group.name + "." + key.name + ": " + msg);
    }
  }
  for (const auto& d : duplicates) errors.push_back(d + ": appears more than once");

  if (errors.empty() && c.friction.f_min >= c.friction.f_max) {
    const auto fr = entries(groups.at("friction"), "", duplicates);
    errors.push_back("friction: f_min (" + text(fr.at("f_min")) + ") must be smaller than f_max (" +
                     text(fr.at("f_max")) + ")");
  }
  if (!errors.empty()) {
    std::string msg = "invalid config.yaml:";
    for (const auto& e : errors) msg += "\n  - " + e;
    throw ConfigError(msg);
  }
  return c;
}

}  // namespace robodog_nmpc
