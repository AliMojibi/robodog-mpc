// Stale-solver check (design decision C): the [codegen] settings compiled into the generated solver against
// config.yaml. Which keys are compared is decided by the generated header alone (generate_solver.py: CODEGEN_KEYS).
#include <charconv>
#include <map>

#include "acados_solver_srbd_nmpc.h"
#include "robodog_nmpc/config.hpp"
#include "srbd_nmpc_codegen_config.h"

// The C++ code relies on this problem structure; a solver generated from a different model needs code changes.
static_assert(SRBD_NMPC_NX == 12, "generated solver: expected 12 states x = [Theta, p, omega, v]");
static_assert(SRBD_NMPC_NU == 12, "generated solver: expected 12 inputs u = [f_FR, f_FL, f_RR, f_RL]");
static_assert(SRBD_NMPC_NP == 17, "generated solver: expected 17 parameters p = [p_foot (12), m, Ixx, Iyy, Izz, mu]");

namespace robodog_nmpc {
namespace {

using Value = std::variant<long, double, std::string>;

// Every scalar setting of a Config by "group.key" (not only [codegen]: the header decides what is compared).
std::map<std::string, Value> scalars(const Config& c) {
  return {
      {"robot.mass", c.robot.mass},
      {"robot.nominal_height", c.robot.nominal_height},
      {"horizon.N", long{c.horizon.N}},
      {"horizon.dt", c.horizon.dt},
      {"friction.model", c.friction.model},
      {"friction.polygon_sides", long{c.friction.polygon_sides}},
      {"friction.mu", c.friction.mu},
      {"friction.f_min", c.friction.f_min},
      {"friction.f_max", c.friction.f_max},
      {"cost.w_yaw_rate", c.cost.w_yaw_rate},
      {"cost.w_height", c.cost.w_height},
      {"cost.w_force", c.cost.w_force},
      {"cost.terminal_weight_scale", c.cost.terminal_weight_scale},
      {"solver.nlp_solver_type", c.solver.nlp_solver_type},
      {"solver.qp_solver", c.solver.qp_solver},
      {"solver.hessian_approx", c.solver.hessian_approx},
      {"solver.integrator_type", c.solver.integrator_type},
      {"solver.sim_method_num_stages", long{c.solver.sim_method_num_stages}},
      {"solver.sim_method_num_steps", long{c.solver.sim_method_num_steps}},
      {"solver.globalization", c.solver.globalization},
      {"solver.nlp_solver_max_iter", long{c.solver.nlp_solver_max_iter}},
      {"solver.qp_solver_iter_max", long{c.solver.qp_solver_iter_max}},
      {"codegen.solver_name", c.codegen.solver_name},
      {"codegen.export_dir", c.codegen.export_dir},
  };
}

std::string show(const Value& v) {
  if (const auto* i = std::get_if<long>(&v)) return std::to_string(*i);
  if (const auto* d = std::get_if<double>(&v)) {
    char buf[32];
    const auto end = std::to_chars(buf, buf + sizeof buf, *d).ptr;   // shortest exact form, e.g. 0.025
    return std::string(buf, end);
  }
  return std::get<std::string>(v);
}

}  // namespace

namespace detail {

std::vector<CodegenEntry> generatedCodegenEntries() {
  std::vector<CodegenEntry> entries;
#define ROBODOG_INT(group, key, value) entries.push_back({group, key, Value(long{value})});
#define ROBODOG_DOUBLE(group, key, value) entries.push_back({group, key, Value(double{value})});
#define ROBODOG_STRING(group, key, value) entries.push_back({group, key, Value(std::string(value))});
  SRBD_NMPC_CODEGEN_CONFIG(ROBODOG_INT, ROBODOG_DOUBLE, ROBODOG_STRING)
#undef ROBODOG_INT
#undef ROBODOG_DOUBLE
#undef ROBODOG_STRING
  return entries;
}

std::vector<std::string> compareCodegen(const Config& config, const std::vector<CodegenEntry>& entries) {
  const auto values = scalars(config);
  std::vector<std::string> mismatches;
  for (const auto& e : entries) {
    const std::string name = e.group + "." + e.key;
    const auto it = values.find(name);
    if (it == values.end())
      mismatches.push_back(name + ": setting of the generated solver unknown to this library "
                                  "(library and scripts/generate_solver.py out of sync)");
    else if (it->second.index() != e.value.index())
      mismatches.push_back(name + ": different type in config.yaml (" + show(it->second) +
                           ") and generated solver (" + show(e.value) + ")");
    else if (it->second != e.value)
      mismatches.push_back(name + ": config.yaml " + show(it->second) + ", generated solver " + show(e.value));
  }
  return mismatches;
}

}  // namespace detail

std::vector<std::string> codegenMismatches(const Config& config) {
  return detail::compareCodegen(config, detail::generatedCodegenEntries());
}

}  // namespace robodog_nmpc
