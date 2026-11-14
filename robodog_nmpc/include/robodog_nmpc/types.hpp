// Input and output types of the NmpcSolver class. Foot order everywhere: FR, FL, RR, RL.
#pragma once

#include <Eigen/Core>

#include <array>

namespace robodog_nmpc {

constexpr int kNumFeet = 4;

// Measured body state, world frame (z up).
struct State {
  Eigen::Vector3d rpy = Eigen::Vector3d::Zero();               // roll, pitch, yaw [rad]; Z-Y-X Euler, R = Rz Ry Rx
  Eigen::Vector3d position = Eigen::Vector3d::Zero();          // CoM position [m]
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();  // [rad/s], world frame
  Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();   // [m/s], world frame
};

// Operator command in the robot's HEADING frame (x forward, y left, rotated by yaw only).
struct Command {
  double vx{0.0};        // [m/s]
  double vy{0.0};        // [m/s]
  double yaw_rate{0.0};  // [rad/s]
};

using Contact = std::array<bool, kNumFeet>;                    // true = stance
using FootPositions = std::array<Eigen::Vector3d, kNumFeet>;   // world frame [m]
using Forces = std::array<Eigen::Vector3d, kNumFeet>;          // ground reaction forces, world frame [N]

enum class Status {
  Ok,
  NotInitialized,   // run() before a successful initialize()
  InvalidConfig,    // config.yaml missing or invalid
  SolverMismatch,   // config.yaml does not match the solver compiled into the library: regenerate
  InvalidInput,     // run() inputs rejected (e.g. NaN); nothing was solved
  SolverFailed,     // acados could not create the solver or reported an error in run()
};

constexpr const char* toString(Status s) {
  switch (s) {
    case Status::Ok: return "Ok";
    case Status::NotInitialized: return "NotInitialized";
    case Status::InvalidConfig: return "InvalidConfig";
    case Status::SolverMismatch: return "SolverMismatch";
    case Status::InvalidInput: return "InvalidInput";
    case Status::SolverFailed: return "SolverFailed";
  }
  return "Unknown";
}

// Result of the last run().
struct Solution {
  Forces forces{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Status status{Status::NotInitialized};
  int acados_status{-1};     // return value of the acados solve: 0 = success, 4 = QP failure, 1 = NaN, ...
  int qp_iterations{0};      // == solver.qp_solver_iter_max: the QP stopped at its iteration limit (acados still
                             // reports success then; the forces are usable but less accurate)
  double solve_time_ms{0.0};
};

}  // namespace robodog_nmpc
