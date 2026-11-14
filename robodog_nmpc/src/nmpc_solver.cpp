// NmpcSolver: wraps the generated acados solver (solver/acados_solver_srbd_nmpc.h).
// The runtime setup mirrors build_ocp() in scripts/srbd_ocp.py and the closed loop of derivation step 11.
#include "robodog_nmpc/nmpc_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Geometry>

#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_srbd_nmpc.h"

namespace robodog_nmpc {
namespace {

constexpr int N = SRBD_NMPC_N;
constexpr int NX = SRBD_NMPC_NX;
constexpr int NU = SRBD_NMPC_NU;
constexpr int NP = SRBD_NMPC_NP;
constexpr int NY = SRBD_NMPC_NY;
constexpr int NYN = SRBD_NMPC_NYN;
static_assert(NY == 19 && NYN == 7,
              "cost: y = [v_heading (3), yaw rate, height, roll, pitch, forces (12)], terminal y_e = first 7");

constexpr double kBig = 1e10;  // "no bound" (= ACADOS_INFTY); 1e9 would be a real bound and slows HPIPM down
constexpr double kGravity = 9.81;

bool finite(const Eigen::Vector3d& v) { return v.allFinite(); }
bool finite(const State& s) {
  return finite(s.rpy) && finite(s.position) && finite(s.angular_velocity) && finite(s.linear_velocity);
}
bool finite(const Command& c) { return std::isfinite(c.vx) && std::isfinite(c.vy) && std::isfinite(c.yaw_rate); }
bool finite(const FootPositions& f) { return std::all_of(f.begin(), f.end(), [](const auto& v) { return finite(v); }); }

}  // namespace

void detail::predictFeet(const Config& config, const State& state, const Command& cmd, const Contact* schedule,
                         int n, const FootPositions& current, FootPositions* out) {
  const double dt = config.horizon.dt;
  double ground = current[0].z();  // flat ground at the height of the lowest foot now (a stance foot)
  for (const auto& f : current) ground = std::min(ground, f.z());

  for (int leg = 0; leg < kNumFeet; ++leg) {
    // the CoM on the commanded path (heading frame command, constant over the horizon), node by node
    Eigen::Vector2d com = state.position.head<2>();
    double yaw = state.rpy.z();
    int touchdown = -1;  // interval at which the current (predicted) stance phase of this foot began
    Eigen::Vector3d foothold = current[leg];
    for (int k = 0; k <= n; ++k) {
      const int j = std::min(k, n - 1);  // node n belongs to the last interval
      if (k > 0 && k < n && schedule[k][leg] && !schedule[k - 1][leg]) {
        // touchdown at interval k: Raibert foothold from the CoM and yaw predicted for that time
        int stance = 0;
        while (k + stance < n && schedule[k + stance][leg]) ++stance;
        const Eigen::Rotation2Dd R(yaw);
        const Eigen::Vector2d hip(config.robot.hip_offsets_xy[2 * leg], config.robot.hip_offsets_xy[2 * leg + 1]);
        const Eigen::Vector2d v_ref = R * Eigen::Vector2d(cmd.vx, cmd.vy);
        const Eigen::Vector2d xy = com + R * hip + v_ref * (stance * dt) / 2.0;
        foothold = Eigen::Vector3d(xy.x(), xy.y(), ground);
        touchdown = k;
      }
      // swing, or stance since now: current position; later stance phase: its predicted foothold
      out[k][leg] = (schedule[j][leg] && touchdown >= 0) ? foothold : current[leg];
      if (k < n) {  // advance the commanded path by one interval
        com += Eigen::Rotation2Dd(yaw) * Eigen::Vector2d(cmd.vx, cmd.vy) * dt;
        yaw += cmd.yaw_rate * dt;
      }
    }
  }
}

struct NmpcSolver::Impl {
  srbd_nmpc_solver_capsule* capsule = nullptr;
  ocp_nlp_config* nlp_config = nullptr;
  ocp_nlp_dims* nlp_dims = nullptr;
  ocp_nlp_in* nlp_in = nullptr;
  ocp_nlp_out* nlp_out = nullptr;
  ocp_nlp_solver* nlp_solver = nullptr;

  Config config;

  // Inputs of the current run(), filled by the public run() forms. Fixed size: no allocation in run().
  std::array<Contact, N> contact{};        // per interval k = 0..N-1
  std::array<FootPositions, N + 1> feet{};  // per node k = 0..N

  // Last successful plan: warm start of the next solve and fallback forces after a failure (decision G).
  std::array<std::array<double, NX>, N + 1> X_plan{};
  std::array<std::array<double, NU>, N> U_plan{};  // raw solver output (warm start identical to the Python loop)
  std::array<Contact, N> contact_plan{};           // the contact schedule the plan was computed for
  bool have_plan = false;
  int steps_since_plan = 0;   // run() calls since the plan was computed
  bool reset_needed = false;  // the last solve failed: clear the iterate (incl. multipliers) first
  Solution solution;
  std::string error;
  bool initialized = false;

  Impl() { error.reserve(1024); }  // short messages set later (in run()) then need no allocation
  ~Impl() { release(); }

  void release() {
    if (capsule) {
      // Safe also after a failed create: its only early return (N differs from the generated N) cannot happen
      // with srbd_nmpc_acados_create(); every other failure occurs after all memory has been allocated.
      srbd_nmpc_acados_free(capsule);
      srbd_nmpc_acados_free_capsule(capsule);
    }
    capsule = nullptr;
    nlp_config = nullptr;
    nlp_dims = nullptr;
    nlp_in = nullptr;
    nlp_out = nullptr;
    nlp_solver = nullptr;
    initialized = false;
    have_plan = false;
    steps_since_plan = 0;
    reset_needed = false;
    solution = Solution{};
  }

  Status fail(Status status, const std::string& message) {
    release();
    error = message;
    return status;
  }

  // Weights, parameters, force bounds and the standing reference from config (the [runtime] settings).
  void applyRuntimeConfig() {
    const auto& c = config;
    std::array<double, 7> w_state{c.cost.w_velocity[0], c.cost.w_velocity[1], c.cost.w_velocity[2],
                                  c.cost.w_yaw_rate, c.cost.w_height, c.cost.w_roll_pitch[0], c.cost.w_roll_pitch[1]};

    std::array<double, NY * NY> W{};  // acados uses 1/2 ||y - y_ref||^2_W  ->  W = 2 diag(weights)
    for (int i = 0; i < NY; ++i) W[i * NY + i] = 2.0 * (i < NYN ? w_state[i] : c.cost.w_force);
    std::array<double, NYN * NYN> W_e{};
    for (int i = 0; i < NYN; ++i) W_e[i * NYN + i] = 2.0 * w_state[i] * c.cost.terminal_weight_scale;

    std::array<double, NP> p{};  // [p_foot (12), m, Ixx, Iyy, Izz, mu]; default feet: under the hips, z = 0
    for (int leg = 0; leg < kNumFeet; ++leg) {
      p[3 * leg] = c.robot.hip_offsets_xy[2 * leg];
      p[3 * leg + 1] = c.robot.hip_offsets_xy[2 * leg + 1];
    }
    p[12] = c.robot.mass;
    p[13] = c.robot.inertia[0];
    p[14] = c.robot.inertia[1];
    p[15] = c.robot.inertia[2];
    p[16] = c.friction.mu;

    std::array<double, NU> lbu{}, ubu{};  // all feet in stance
    for (int leg = 0; leg < kNumFeet; ++leg) {
      lbu[3 * leg] = lbu[3 * leg + 1] = -kBig;
      ubu[3 * leg] = ubu[3 * leg + 1] = kBig;
      lbu[3 * leg + 2] = c.friction.f_min;
      ubu[3 * leg + 2] = c.friction.f_max;
    }

    std::array<double, NY> yref{};  // stand still at the nominal height
    yref[4] = c.robot.nominal_height;

    for (int k = 0; k <= N; ++k) srbd_nmpc_acados_update_params(capsule, k, p.data(), NP);
    for (int k = 0; k < N; ++k) {
      ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, k, "W", W.data());
      ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, k, "yref", yref.data());
      ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, k, "lbu", lbu.data());
      ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, k, "ubu", ubu.data());
    }
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e.data());
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref.data());  // first NYN entries used
  }

  // Force of one foot from a solver input vector. Swing feet: the interior-point QP solver meets the bounds
  // 0 <= f <= 0 only up to ~1e-16, so they are returned as exact zeros.
  static Eigen::Vector3d forceOf(const std::array<double, NU>& u, const Contact& contact, int leg) {
    if (!contact[leg]) return Eigen::Vector3d::Zero();
    return Eigen::Vector3d(u[3 * leg], u[3 * leg + 1], u[3 * leg + 2]);
  }

  // Forces after InvalidInput / SolverFailed: the last plan's step for "now", feet in swing now get zero force.
  Status fallback(Status status, const char* message, const Contact* contact_now) {
    for (int leg = 0; leg < kNumFeet; ++leg) {
      const int k = std::min(steps_since_plan, N - 1);
      solution.forces[leg] = forceOf(U_plan[k], contact_plan[k], leg);   // exact 0 where the plan had swing
      if (!have_plan || (contact_now && !(*contact_now)[leg])) solution.forces[leg].setZero();
    }
    solution.status = status;
    error = message;  // literal shorter than the reserved capacity: no allocation
    return status;
  }

  // One SQP-RTI iteration with the inputs in `contact` and `feet` (same steps as mpc_step() in step 11).
  Status solve(const State& st, const Command& cmd) {
    const auto& c = config;
    std::array<double, NX> x0{};
    for (int i = 0; i < 3; ++i) {
      x0[i] = st.rpy[i];
      x0[3 + i] = st.position[i];
      x0[6 + i] = st.angular_velocity[i];
      x0[9 + i] = st.linear_velocity[i];
    }

    // parameters per node: [p_foot (12), m, Ixx, Iyy, Izz, mu]
    std::array<double, NP> p{};
    p[12] = c.robot.mass;
    p[13] = c.robot.inertia[0];
    p[14] = c.robot.inertia[1];
    p[15] = c.robot.inertia[2];
    p[16] = c.friction.mu;
    for (int k = 0; k <= N; ++k) {
      for (int leg = 0; leg < kNumFeet; ++leg)
        for (int i = 0; i < 3; ++i) p[3 * leg + i] = feet[k][leg][i];
      srbd_nmpc_acados_update_params(capsule, k, p.data(), NP);
    }

    // contact bounds per interval (swing: f = 0; stance: f_min <= f_z <= f_max) and the cost reference
    std::array<double, NY> yref{};
    yref[0] = cmd.vx;
    yref[1] = cmd.vy;
    yref[3] = cmd.yaw_rate;
    yref[4] = c.robot.nominal_height;
    for (int k = 0; k < N; ++k) {
      std::array<double, NU> lbu{}, ubu{};
      for (int leg = 0; leg < kNumFeet; ++leg) {
        if (!contact[k][leg]) continue;  // swing: all three components bounded to 0
        lbu[3 * leg] = lbu[3 * leg + 1] = -kBig;
        ubu[3 * leg] = ubu[3 * leg + 1] = kBig;
        lbu[3 * leg + 2] = c.friction.f_min;
        ubu[3 * leg + 2] = c.friction.f_max;
      }
      ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, k, "lbu", lbu.data());
      ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, k, "ubu", ubu.data());
      ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, k, "yref", yref.data());
    }
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref.data());

    // initial guess
    if (reset_needed) {  // after a failure the iterate may hold NaN; keep parameters and weights (0 = no reset)
      srbd_nmpc_acados_reset(capsule, 1, 0, 0, 0);
      reset_needed = false;
    }
    if (have_plan) {  // warm start: the last plan shifted forward in time
      for (int k = 0; k <= N; ++k) {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, k, "x",
                        X_plan[std::min(k + steps_since_plan, N)].data());
        if (k < N)
          ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, k, "u",
                          U_plan[std::min(k + steps_since_plan, N - 1)].data());
      }
    } else {  // cold start: the body stays where it is, the stance feet share the weight
      for (int k = 0; k <= N; ++k) {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, k, "x", x0.data());
        if (k == N) break;
        const int n_stance = static_cast<int>(std::count(contact[k].begin(), contact[k].end(), true));
        std::array<double, NU> u{};
        for (int leg = 0; leg < kNumFeet; ++leg)
          if (contact[k][leg]) u[3 * leg + 2] = c.robot.mass * kGravity / n_stance;
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, k, "u", u.data());
      }
    }
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", x0.data());  // x_0 = measured
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", x0.data());

    const int status = srbd_nmpc_acados_solve(capsule);
    double time_tot = 0.0;
    ocp_nlp_get(nlp_solver, "time_tot", &time_tot);
    ocp_nlp_get(nlp_solver, "qp_iter", &solution.qp_iterations);
    solution.acados_status = status;
    solution.solve_time_ms = 1e3 * time_tot;

    std::array<double, NU> u0{};
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "u", u0.data());
    const bool u0_finite = std::all_of(u0.begin(), u0.end(), [](double v) { return std::isfinite(v); });
    if (status != 0 || !u0_finite) {
      reset_needed = true;
      return fallback(Status::SolverFailed, "acados: the solve failed (see Solution::acados_status); "
                                            "forces from the previous plan", &contact[0]);
    }

    for (int k = 0; k <= N; ++k) {
      ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, k, "x", X_plan[k].data());
      if (k < N) ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, k, "u", U_plan[k].data());
    }
    contact_plan = contact;
    have_plan = true;
    steps_since_plan = 0;
    for (int leg = 0; leg < kNumFeet; ++leg) solution.forces[leg] = forceOf(U_plan[0], contact[0], leg);
    solution.status = Status::Ok;
    error.clear();
    return Status::Ok;
  }

  // Common start of every run(): not initialized -> NotInitialized; otherwise one more step since the plan.
  bool begin() {
    if (!initialized) {
      solution.status = Status::NotInitialized;
      error = "run() called before a successful initialize()";
      return false;
    }
    ++steps_since_plan;
    solution.acados_status = -1;
    solution.qp_iterations = 0;
    solution.solve_time_ms = 0.0;
    return true;
  }
};

NmpcSolver::NmpcSolver() : impl_(std::make_unique<Impl>()) {}
NmpcSolver::~NmpcSolver() = default;

Status NmpcSolver::initialize(const std::string& config_path) {
  Impl& s = *impl_;
  s.release();  // a re-initialization starts from scratch

  try {
    s.config = loadConfig(config_path);
  } catch (const ConfigError& e) {
    return s.fail(Status::InvalidConfig, e.what());
  }

  const auto mismatches = codegenMismatches(s.config);
  if (!mismatches.empty()) {
    std::string msg = "config.yaml does not match the solver compiled into this library "
                      "(rerun scripts/generate_solver.py, then rebuild):";
    for (const auto& m : mismatches) msg += "\n  - " + m;
    return s.fail(Status::SolverMismatch, msg);
  }

  s.capsule = srbd_nmpc_acados_create_capsule();
  if (!s.capsule) return s.fail(Status::SolverFailed, "acados: out of memory (create_capsule)");
  if (const int status = srbd_nmpc_acados_create(s.capsule); status != 0)
    return s.fail(Status::SolverFailed, "acados: creating the solver failed (status " + std::to_string(status) + ")");
  s.nlp_config = srbd_nmpc_acados_get_nlp_config(s.capsule);
  s.nlp_dims = srbd_nmpc_acados_get_nlp_dims(s.capsule);
  s.nlp_in = srbd_nmpc_acados_get_nlp_in(s.capsule);
  s.nlp_out = srbd_nmpc_acados_get_nlp_out(s.capsule);
  s.nlp_solver = srbd_nmpc_acados_get_nlp_solver(s.capsule);

  s.applyRuntimeConfig();
  s.initialized = true;
  s.error.clear();
  return Status::Ok;
}

Status NmpcSolver::run(const State& state, const Command& cmd, const Contact& contact, const FootPositions& feet) {
  Impl& s = *impl_;
  if (!s.begin()) return Status::NotInitialized;
  if (!finite(state) || !finite(cmd) || !finite(feet))
    return s.fallback(Status::InvalidInput, "run(): state, command or foot positions contain NaN or Inf", &contact);
  s.contact.fill(contact);
  s.feet.fill(feet);
  return s.solve(state, cmd);
}

Status NmpcSolver::run(const State& state, const Command& cmd, const std::vector<Contact>& contact_schedule,
                       const FootPositions& feet) {
  Impl& s = *impl_;
  if (!s.begin()) return Status::NotInitialized;
  const Contact* now = contact_schedule.empty() ? nullptr : &contact_schedule[0];
  if (contact_schedule.size() != static_cast<std::size_t>(N))
    return s.fallback(Status::InvalidInput, "run(): contact_schedule must have horizonLength() entries", now);
  if (!finite(state) || !finite(cmd) || !finite(feet))
    return s.fallback(Status::InvalidInput, "run(): state, command or foot positions contain NaN or Inf", now);
  std::copy(contact_schedule.begin(), contact_schedule.end(), s.contact.begin());
  detail::predictFeet(s.config, state, cmd, s.contact.data(), N, feet, s.feet.data());
  return s.solve(state, cmd);
}

Status NmpcSolver::run(const State& state, const Command& cmd, const std::vector<Contact>& contact_schedule,
                       const std::vector<FootPositions>& foot_plan) {
  Impl& s = *impl_;
  if (!s.begin()) return Status::NotInitialized;
  const Contact* now = contact_schedule.empty() ? nullptr : &contact_schedule[0];
  if (contact_schedule.size() != static_cast<std::size_t>(N) || foot_plan.size() != static_cast<std::size_t>(N + 1))
    return s.fallback(Status::InvalidInput,
                      "run(): contact_schedule must have horizonLength() entries and foot_plan horizonLength() + 1",
                      now);
  if (!finite(state) || !finite(cmd) ||
      !std::all_of(foot_plan.begin(), foot_plan.end(), [](const FootPositions& f) { return finite(f); }))
    return s.fallback(Status::InvalidInput, "run(): state, command or foot positions contain NaN or Inf", now);
  std::copy(contact_schedule.begin(), contact_schedule.end(), s.contact.begin());
  std::copy(foot_plan.begin(), foot_plan.end(), s.feet.begin());
  return s.solve(state, cmd);
}

bool NmpcSolver::isInitialized() const { return impl_->initialized; }
const Solution& NmpcSolver::getSolution() const { return impl_->solution; }
const std::string& NmpcSolver::errorMessage() const { return impl_->error; }
const Config& NmpcSolver::config() const { return impl_->config; }
int NmpcSolver::horizonLength() const { return N; }

}  // namespace robodog_nmpc
