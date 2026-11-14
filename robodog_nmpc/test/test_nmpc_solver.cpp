// Tests of the NmpcSolver class.
#include <gtest/gtest.h>

#include <Eigen/Geometry>  // cross()

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "robodog_nmpc/nmpc_solver.hpp"
#include "test_helpers.hpp"

using robodog_nmpc::NmpcSolver;
using robodog_nmpc::Status;
using test_helpers::TempConfig;
using test_helpers::withValue;

namespace {

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

}  // namespace

// ---------------- initialize() ----------------

TEST(Initialize, StateBeforeInitialize) {
  NmpcSolver mpc;
  EXPECT_FALSE(mpc.isInitialized());
  EXPECT_EQ(mpc.getSolution().status, Status::NotInitialized);
  EXPECT_EQ(mpc.errorMessage(), "");
  EXPECT_EQ(mpc.horizonLength(), 16);
}

TEST(Initialize, RealConfig) {
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok) << mpc.errorMessage();
  EXPECT_TRUE(mpc.isInitialized());
  EXPECT_EQ(mpc.errorMessage(), "");
  EXPECT_EQ(mpc.config().robot.mass, 43.0);
}

TEST(Initialize, MissingFile) {
  NmpcSolver mpc;
  EXPECT_EQ(mpc.initialize("/nonexistent/config.yaml"), Status::InvalidConfig);
  EXPECT_FALSE(mpc.isInitialized());
  EXPECT_TRUE(contains(mpc.errorMessage(), "/nonexistent/config.yaml: cannot open the file")) << mpc.errorMessage();
}

TEST(Initialize, InvalidConfigListsAllProblems) {
  TempConfig file(withValue("mass", "-1.0", withValue("model", "polyogn")));
  NmpcSolver mpc;
  EXPECT_EQ(mpc.initialize(file.path), Status::InvalidConfig);
  EXPECT_FALSE(mpc.isInitialized());
  EXPECT_TRUE(contains(mpc.errorMessage(), "robot.mass: must be a number > 0")) << mpc.errorMessage();
  EXPECT_TRUE(contains(mpc.errorMessage(), "friction.model: must be one of")) << mpc.errorMessage();
}

TEST(Initialize, StaleSolverIsRefused) {
  TempConfig file(withValue("dt", "0.02", withValue("N", "20")));
  NmpcSolver mpc;
  EXPECT_EQ(mpc.initialize(file.path), Status::SolverMismatch);
  EXPECT_FALSE(mpc.isInitialized());
  const std::string& msg = mpc.errorMessage();
  EXPECT_TRUE(contains(msg, "rerun scripts/generate_solver.py")) << msg;
  EXPECT_TRUE(contains(msg, "horizon.N: config.yaml 20, generated solver 16")) << msg;
  EXPECT_TRUE(contains(msg, "horizon.dt: config.yaml 0.02, generated solver 0.025")) << msg;
}

TEST(Initialize, RuntimeSettingsNeedNoRegeneration) {
  TempConfig file(withValue("mass", "30.0", withValue("mu", "0.4", withValue("w_height", "10.0"))));
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(file.path), Status::Ok) << mpc.errorMessage();
  EXPECT_EQ(mpc.config().robot.mass, 30.0);
  EXPECT_EQ(mpc.config().friction.mu, 0.4);
}

TEST(Initialize, ReinitializeAfterSuccessAndFailure) {
  TempConfig bad(withValue("N", "0"));
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  EXPECT_EQ(mpc.initialize(bad.path), Status::InvalidConfig);   // failure: the old solver is gone, too
  EXPECT_FALSE(mpc.isInitialized());
  EXPECT_EQ(mpc.getSolution().status, Status::NotInitialized);
  EXPECT_FALSE(mpc.errorMessage().empty());
  EXPECT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  EXPECT_TRUE(mpc.isInitialized());
  EXPECT_EQ(mpc.errorMessage(), "");                            // cleared by the successful call
}

TEST(Initialize, RepeatedAndSideBySide) {
  NmpcSolver mpc;
  for (int i = 0; i < 50; ++i) ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok) << "iteration " << i;
  auto other = std::make_unique<NmpcSolver>();                  // independent instance (and movable via unique_ptr)
  EXPECT_EQ(other->initialize(CONFIG_YAML), Status::Ok);
  EXPECT_TRUE(mpc.isInitialized());
  other.reset();                                                // destroying one leaves the other intact
  EXPECT_TRUE(mpc.isInitialized());
}

TEST(Status, Names) {
  EXPECT_STREQ(robodog_nmpc::toString(Status::Ok), "Ok");
  EXPECT_STREQ(robodog_nmpc::toString(Status::SolverMismatch), "SolverMismatch");
}

// ---------------- run() ----------------

namespace {

using robodog_nmpc::Command;
using robodog_nmpc::Config;
using robodog_nmpc::Contact;
using robodog_nmpc::FootPositions;
using robodog_nmpc::Forces;
using robodog_nmpc::State;
constexpr double kG = 9.81;

State standing(double height = 0.45) {
  State s;
  s.position = Eigen::Vector3d(0.0, 0.0, height);
  return s;
}

FootPositions feetUnderHips(const Config& c) {
  FootPositions f;
  for (int leg = 0; leg < 4; ++leg) f[leg] = Eigen::Vector3d(c.robot.hip_offsets_xy[2 * leg], c.robot.hip_offsets_xy[2 * leg + 1], 0.0);
  return f;
}

// run() `steps` times with the same inputs (the RTI iterations converge to the solution of this problem).
Forces solveRepeatedly(NmpcSolver& mpc, const State& s, const Command& cmd, const Contact& contact, int steps = 30) {
  for (int i = 0; i < steps; ++i) EXPECT_EQ(mpc.run(s, cmd, contact, feetUnderHips(mpc.config())), Status::Ok) << mpc.errorMessage();
  return mpc.getSolution().forces;
}

double sumFz(const Forces& f) { return f[0].z() + f[1].z() + f[2].z() + f[3].z(); }

}  // namespace

TEST(Run, Standing) {
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  const Forces f = solveRepeatedly(mpc, standing(), Command{}, {true, true, true, true});
  const double m = mpc.config().robot.mass;
  EXPECT_NEAR(sumFz(f), m * kG, 0.01 * m * kG);
  for (int leg = 0; leg < 4; ++leg) {
    EXPECT_NEAR(f[leg].z(), m * kG / 4, 0.02 * m * kG) << "leg " << leg;
    EXPECT_NEAR(f[leg].x(), 0.0, 1.0) << "leg " << leg;
    EXPECT_NEAR(f[leg].y(), 0.0, 1.0) << "leg " << leg;
  }
  EXPECT_EQ(mpc.getSolution().status, Status::Ok);
  EXPECT_EQ(mpc.getSolution().acados_status, 0);
  EXPECT_GT(mpc.getSolution().qp_iterations, 0);
  EXPECT_LT(mpc.getSolution().qp_iterations, mpc.config().solver.qp_solver_iter_max);
  EXPECT_GT(mpc.getSolution().solve_time_ms, 0.0);
}

TEST(Run, RuntimeMassReachesTheSolver) {
  TempConfig file(withValue("mass", "30.0"));
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(file.path), Status::Ok);
  EXPECT_NEAR(sumFz(solveRepeatedly(mpc, standing(), Command{}, {true, true, true, true})), 30.0 * kG, 0.01 * 30.0 * kG);
}

TEST(Run, RuntimeForceBoundsReachTheSolver) {
  TempConfig file(withValue("f_min", "150.0"));      // more than m g / 4 = 105 N: every stance foot is pushed up
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(file.path), Status::Ok);
  const Forces f = solveRepeatedly(mpc, standing(), Command{}, {true, true, true, true});
  for (int leg = 0; leg < 4; ++leg) EXPECT_GE(f[leg].z(), 150.0 - 1e-6) << "leg " << leg;
}

TEST(Run, TrotPairSwingZeroStanceInLimits) {
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  const Config& c = mpc.config();
  const Forces f = solveRepeatedly(mpc, standing(), Command{0.5, 0.0, 0.3}, {true, false, false, true});  // FR + RL
  for (int leg : {1, 2}) EXPECT_EQ(f[leg], Eigen::Vector3d::Zero()) << "swing leg " << leg << " must be exactly 0";
  for (int leg : {0, 3}) {
    EXPECT_GE(f[leg].z(), c.friction.f_min - 1e-6) << "leg " << leg;
    EXPECT_LE(f[leg].z(), c.friction.f_max + 1e-6) << "leg " << leg;
    EXPECT_LE(f[leg].head<2>().norm(), c.friction.mu * f[leg].z() + 1e-6) << "friction, leg " << leg;
  }
}

TEST(Run, CommandsProduceTheRightForces) {
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  const Contact all{true, true, true, true};
  Forces f = solveRepeatedly(mpc, standing(), Command{0.5, 0.0, 0.0}, all, 1);   // first step: accelerate forward
  EXPECT_GT(f[0].x() + f[1].x() + f[2].x() + f[3].x(), 1.0);
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  f = solveRepeatedly(mpc, standing(), Command{0.0, 0.0, 0.6}, all, 1);          // first step: start turning left
  const FootPositions feet = feetUnderHips(mpc.config());
  double yaw_torque = 0.0;
  for (int leg = 0; leg < 4; ++leg) yaw_torque += (feet[leg] - standing().position).cross(f[leg]).z();
  EXPECT_GT(yaw_torque, 0.1);
}

TEST(Run, FullFormEqualsSimpleForm) {
  NmpcSolver a, b;
  ASSERT_EQ(a.initialize(CONFIG_YAML), Status::Ok);
  ASSERT_EQ(b.initialize(CONFIG_YAML), Status::Ok);
  const Contact contact{true, false, false, true};
  const FootPositions feet = feetUnderHips(a.config());
  const std::vector<Contact> schedule(a.horizonLength(), contact);
  const std::vector<FootPositions> plan(a.horizonLength() + 1, feet);
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(a.run(standing(), Command{0.3, 0.1, 0.2}, contact, feet), Status::Ok);
    ASSERT_EQ(b.run(standing(), Command{0.3, 0.1, 0.2}, schedule, plan), Status::Ok);
    for (int leg = 0; leg < 4; ++leg) EXPECT_EQ(a.getSolution().forces[leg], b.getSolution().forces[leg]);
  }
}

TEST(Run, NotInitialized) {
  NmpcSolver mpc;
  FootPositions feet;
  for (auto& f : feet) f.setZero();
  EXPECT_EQ(mpc.run(standing(), Command{}, {true, true, true, true}, feet), Status::NotInitialized);
  EXPECT_EQ(mpc.getSolution().status, Status::NotInitialized);
  EXPECT_FALSE(mpc.errorMessage().empty());
}

TEST(Run, InvalidInputUsesThePreviousPlan) {
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  const Contact all{true, true, true, true};
  const FootPositions feet = feetUnderHips(mpc.config());
  State bad = standing();
  bad.linear_velocity.x() = std::nan("");
  EXPECT_EQ(mpc.run(bad, Command{}, all, feet), Status::InvalidInput);   // no plan yet: zero forces
  for (const auto& f : mpc.getSolution().forces) EXPECT_EQ(f, Eigen::Vector3d::Zero());

  solveRepeatedly(mpc, standing(), Command{}, all, 5);
  EXPECT_EQ(mpc.run(bad, Command{}, all, feet), Status::InvalidInput);
  EXPECT_EQ(mpc.getSolution().acados_status, -1);                        // nothing was solved
  EXPECT_FALSE(mpc.errorMessage().empty());
  EXPECT_NEAR(sumFz(mpc.getSolution().forces), mpc.config().robot.mass * kG, 0.02 * mpc.config().robot.mass * kG);
  EXPECT_EQ(mpc.run(bad, Command{}, {true, false, false, true}, feet), Status::InvalidInput);   // swing now -> 0
  EXPECT_EQ(mpc.getSolution().forces[1], Eigen::Vector3d::Zero());
  EXPECT_EQ(mpc.getSolution().forces[2], Eigen::Vector3d::Zero());
  EXPECT_GT(mpc.getSolution().forces[0].z(), 0.0);

  FootPositions bad_feet = feet;
  bad_feet[2].y() = std::numeric_limits<double>::infinity();
  EXPECT_EQ(mpc.run(standing(), Command{}, all, bad_feet), Status::InvalidInput);
  EXPECT_EQ(mpc.run(standing(), Command{std::nan(""), 0.0, 0.0}, all, feet), Status::InvalidInput);
  EXPECT_EQ(mpc.run(standing(), Command{}, std::vector<Contact>(3, all), std::vector<FootPositions>(17, feet)),
            Status::InvalidInput);
  EXPECT_EQ(mpc.run(standing(), Command{}, std::vector<Contact>(16, all), std::vector<FootPositions>(16, feet)),
            Status::InvalidInput);
  EXPECT_EQ(mpc.run(standing(), Command{}, std::vector<Contact>{}, std::vector<FootPositions>{}), Status::InvalidInput);

  EXPECT_EQ(mpc.run(standing(), Command{}, all, feet), Status::Ok);     // valid input again: back to normal
  EXPECT_EQ(mpc.errorMessage(), "");
}

// Harsh scenario used below: trot pairs switch every 4 steps, form (a) gives the solver no preview of the switch,
// and the measured state never moves. The Python acados interface fails at exactly the same steps (4, 36, ...)
// with bit-identical forces: this is solver behavior, and the fallback of decision G has to handle it.
Status harshStep(NmpcSolver& mpc, int i) {
  const bool a = (i / 4) % 2 == 0;
  return mpc.run(standing(), Command{0.5, 0.0, 0.3}, {a, !a, !a, a}, feetUnderHips(mpc.config()));
}

TEST(Run, SolverFailureFallbackAndRecovery) {
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  for (int i = 0; i < 4; ++i) ASSERT_EQ(harshStep(mpc, i), Status::Ok) << "step " << i;
  EXPECT_EQ(harshStep(mpc, 4), Status::SolverFailed);           // first step with the new pair FL + RR
  const auto& s = mpc.getSolution();
  EXPECT_EQ(s.acados_status, 4);                                 // ACADOS_QP_FAILURE
  EXPECT_FALSE(mpc.errorMessage().empty());
  // previous plan's next step (it had FR + RL in stance for the whole horizon), feet in swing now (FR, RL) -> 0
  for (const auto& f : s.forces) EXPECT_EQ(f, Eigen::Vector3d::Zero());
  EXPECT_EQ(harshStep(mpc, 5), Status::Ok);                      // recovers at the next step
  EXPECT_EQ(mpc.errorMessage(), "");
  EXPECT_EQ(mpc.getSolution().forces[0], Eigen::Vector3d::Zero());   // FR in swing
  EXPECT_GT(mpc.getSolution().forces[1].z(), 0.0);                   // FL in stance
}

TEST(Run, PredictedTrajectory) {
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  EXPECT_EQ(mpc.getSolution().plan_age, -1);
  ASSERT_EQ(mpc.getSolution().predicted_states.size(), 17u);   // sized by initialize()
  ASSERT_EQ(mpc.getSolution().predicted_forces.size(), 16u);
  solveRepeatedly(mpc, standing(), Command{}, {true, false, false, true}, 3);
  const auto& sol = mpc.getSolution();
  EXPECT_EQ(sol.plan_age, 0);
  EXPECT_EQ(sol.predicted_states[0].position, standing().position);       // x_0 = measured state
  for (int leg = 0; leg < 4; ++leg) EXPECT_EQ(sol.predicted_forces[0][leg], sol.forces[leg]);   // u_0 = output
  for (const auto& f : sol.predicted_forces) EXPECT_EQ(f[1], Eigen::Vector3d::Zero());          // FL in swing
  EXPECT_LT(sol.predicted_states[16].position.z(), 0.45);   // two feet only: the body sags a little over 0.4 s
}

// Closed loop in C++: the "robot" follows the model exactly (the next measured state is the plan's x_1, i.e. one
// RK4 step of the solver's own model). Not a substitute for a simulation, but realistic inputs for timing and a
// check that the commands are tracked. Scenario: stand, trot in place, forward, forward + turn, forward + sideways,
// trot in place. Feet: form (b); a foot lands on the foothold that was predicted for it.
TEST(Run, ClosedLoopTrackingAndTiming) {
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  const Config& c = mpc.config();
  const int n = mpc.horizonLength();
  const double dt = c.horizon.dt;
  struct Phase { double t, vx, vy, wz; };
  const std::vector<Phase> scenario = {{0.0, 0, 0, 0}, {1.5, 0.5, 0, 0}, {4.0, 0.5, 0, 0.6}, {6.0, 0.5, 0.2, 0}, {8.0, 0, 0, 0}};
  auto command = [&](int i) {
    Command cmd;
    for (const auto& p : scenario)
      if (p.t <= i * dt + 1e-9) cmd = Command{p.vx, p.vy, p.wz};
    return cmd;
  };
  auto contact = [&](int i) -> Contact {   // all feet down until 0.5 s, then trot, pairs switch every 0.1 s
    if (i < 20) return {true, true, true, true};
    const bool a = ((i - 20) / 4) % 2 == 0;
    return {a, !a, !a, a};
  };
  auto headingVelocity = [](const State& s) {   // Rz(yaw)^T v
    return Eigen::Rotation2Dd(-s.rpy.z()) * s.linear_velocity.head<2>();
  };

  State x = standing(c.robot.nominal_height);
  FootPositions feet = feetUnderHips(c);
  std::vector<FootPositions> predicted(n + 1, feet);
  std::vector<double> times;
  int failures = 0, qp_limit = 0;
  const int steps = 400;   // 10 s
  for (int i = 0; i < steps; ++i) {
    std::vector<Contact> schedule(n);
    for (int k = 0; k < n; ++k) schedule[k] = contact(i + k);
    for (int leg = 0; leg < 4; ++leg)   // touchdown now: the foot lands where it was predicted one step ago
      if (schedule[0][leg] && i > 0 && !contact(i - 1)[leg]) feet[leg] = predicted[1][leg];
    const Command cmd = command(i);
    robodog_nmpc::detail::predictFeet(c, x, cmd, schedule.data(), n, feet, predicted.data());

    const Status st = mpc.run(x, cmd, schedule, feet);
    const auto& sol = mpc.getSolution();
    ASSERT_TRUE(st == Status::Ok || st == Status::SolverFailed) << "step " << i << ": " << mpc.errorMessage();
    times.push_back(sol.solve_time_ms);
    failures += st == Status::SolverFailed;
    qp_limit += sol.qp_iterations >= c.solver.qp_solver_iter_max;
    if (st == Status::Ok) {
      for (int leg = 0; leg < 4; ++leg) {
        const Eigen::Vector3d& f = sol.forces[leg];
        if (!schedule[0][leg]) {
          ASSERT_EQ(f, Eigen::Vector3d::Zero()) << "step " << i << ", swing leg " << leg;
          continue;
        }
        ASSERT_GE(f.z(), c.friction.f_min - 1e-6) << "step " << i << ", leg " << leg;
        ASSERT_LE(f.z(), c.friction.f_max + 1e-6) << "step " << i << ", leg " << leg;
        ASSERT_LE(f.head<2>().norm(), c.friction.mu * f.z() + 1e-6) << "step " << i << ", leg " << leg;
      }
    }
    // the robot: the plan's next state (after a failure: the next state of the plan the fallback forces come from)
    x = sol.predicted_states[std::min(sol.plan_age + 1, n)];
    ASSERT_TRUE(x.position.allFinite() && x.linear_velocity.allFinite()) << "step " << i;
    EXPECT_NEAR(x.position.z(), c.robot.nominal_height, 0.03) << "step " << i;
    EXPECT_LT(x.rpy.head<2>().cwiseAbs().maxCoeff(), 0.1) << "roll/pitch, step " << i;

    const double t = (i + 1) * dt;   // checks at the end of each phase
    if (std::abs(t - 4.0) < 1e-9) {
      EXPECT_NEAR(headingVelocity(x).x(), 0.5, 0.05) << "forward";
      EXPECT_NEAR(headingVelocity(x).y(), 0.0, 0.05) << "forward";
    }
    if (std::abs(t - 6.0) < 1e-9) EXPECT_NEAR(x.angular_velocity.z(), 0.6, 0.06) << "turn";
    if (std::abs(t - 8.0) < 1e-9) {
      EXPECT_NEAR(headingVelocity(x).x(), 0.5, 0.05) << "forward + sideways";
      EXPECT_NEAR(headingVelocity(x).y(), 0.2, 0.05) << "forward + sideways";
    }
    if (std::abs(t - 10.0) < 1e-9) EXPECT_LT(x.linear_velocity.norm(), 0.05) << "stop";
  }
  std::sort(times.begin(), times.end());
  std::cout << "[ timing   ] closed loop, " << steps << " steps: median " << times[steps / 2] << " ms, 99% "
            << times[steps * 99 / 100] << " ms, max " << times.back() << " ms; " << failures << " solver failure(s), "
            << qp_limit << " QP(s) at the iteration limit" << std::endl;
  EXPECT_LT(times[steps / 2], 5.0);   // generous: ~1.5 ms expected on a desktop CPU
  EXPECT_LE(failures, steps / 20);    // the fallback covers single failures; many would mean a real problem
}

// ---------------- form (b): foothold prediction (Raibert) ----------------

namespace {

// Trot schedule over N = 16 intervals: FR + RL in stance for intervals 0-3 and 8-11, FL + RR for 4-7 and 12-15.
std::vector<Contact> trotSchedule(int n = 16, int phase = 4) {
  std::vector<Contact> s(n);
  for (int k = 0; k < n; ++k) {
    const bool a = (k / phase) % 2 == 0;
    s[k] = {a, !a, !a, a};
  }
  return s;
}

std::vector<FootPositions> predict(const Config& c, const State& s, const Command& cmd,
                                   const std::vector<Contact>& schedule, const FootPositions& current) {
  std::vector<FootPositions> out(schedule.size() + 1);
  robodog_nmpc::detail::predictFeet(c, s, cmd, schedule.data(), static_cast<int>(schedule.size()), current, out.data());
  return out;
}

void expectNear(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const std::string& what) {
  EXPECT_LT((a - b).norm(), 1e-12) << what << ": got " << a.transpose() << ", expected " << b.transpose();
}

}  // namespace

TEST(PredictFeet, TrotForwardByHand) {
  const Config c = robodog_nmpc::loadConfig(CONFIG_YAML);   // dt 0.025, hips (+-0.30, +-0.15)
  FootPositions now = feetUnderHips(c);
  now[1].z() = now[2].z() = 0.08;                           // FL and RR are in the air now
  const auto f = predict(c, standing(), Command{0.5, 0.0, 0.0}, trotSchedule(), now);
  for (int k : {0, 3, 4, 7, 12, 15, 16}) expectNear(f[k][0], now[0], "FR, node " + std::to_string(k));  // stance now / swing
  // FR touches down at interval 8: CoM x = 0.5 * 8 * 0.025 = 0.1; + hip 0.30 + v T/2 = 0.5 * 0.1 / 2 = 0.025
  for (int k = 8; k <= 11; ++k) expectNear(f[k][0], Eigen::Vector3d(0.425, -0.15, 0.0), "FR, node " + std::to_string(k));
  // FL touches down at 4 (CoM x 0.05) and at 12 (CoM x 0.15); ground = lowest foot now (0.0)
  for (int k = 4; k <= 7; ++k) expectNear(f[k][1], Eigen::Vector3d(0.375, 0.15, 0.0), "FL, node " + std::to_string(k));
  for (int k = 12; k <= 16; ++k) expectNear(f[k][1], Eigen::Vector3d(0.475, 0.15, 0.0), "FL, node " + std::to_string(k));
  for (int k : {0, 3, 8, 11}) expectNear(f[k][1], now[1], "FL in swing, node " + std::to_string(k));
}

TEST(PredictFeet, HeadingRotatesHipsAndCommand) {
  const Config c = robodog_nmpc::loadConfig(CONFIG_YAML);
  State s = standing();
  s.rpy.z() = M_PI / 2;   // facing +y: forward command moves +y, FR hip (0.30, -0.15) is at world (0.15, 0.30)
  const auto f = predict(c, s, Command{0.5, 0.0, 0.0}, trotSchedule(), feetUnderHips(c));
  expectNear(f[8][0], Eigen::Vector3d(0.15, 0.1 + 0.30 + 0.025, 0.0), "FR touchdown at 8");
}

TEST(PredictFeet, TruncatedStanceAndRaisedGround) {
  const Config c = robodog_nmpc::loadConfig(CONFIG_YAML);
  std::vector<Contact> schedule(16, Contact{true, false, true, true});
  schedule[14][1] = schedule[15][1] = true;               // FL touches down at 14, phase cut by the horizon: T = 0.05
  FootPositions now = feetUnderHips(c);
  for (auto& p : now) p.z() = 0.2;                        // robot on a 0.2 m platform
  now[1].z() = 0.3;                                       // FL in the air
  const auto f = predict(c, standing(0.65), Command{0.5, 0.0, 0.0}, schedule, now);
  expectNear(f[14][1], Eigen::Vector3d(0.5 * 14 * 0.025 + 0.30 + 0.5 * 0.05 / 2, 0.15, 0.2), "FL touchdown at 14");
  expectNear(f[16][1], f[14][1], "FL, last node");
  for (int k = 0; k <= 16; ++k) expectNear(f[k][0], now[0], "FR, stance the whole time");
}

TEST(PredictFeet, ConstantScheduleKeepsTheFeet) {
  const Config c = robodog_nmpc::loadConfig(CONFIG_YAML);
  const auto f = predict(c, standing(), Command{0.5, 0.2, 0.6}, std::vector<Contact>(16, Contact{true, false, false, true}),
                         feetUnderHips(c));
  for (int k = 0; k <= 16; ++k)
    for (int leg = 0; leg < 4; ++leg) EXPECT_EQ(f[k][leg], feetUnderHips(c)[leg]);
}

TEST(Run, FormBMatchesTheOtherForms) {
  NmpcSolver a, b, c;
  ASSERT_EQ(a.initialize(CONFIG_YAML), Status::Ok);
  ASSERT_EQ(b.initialize(CONFIG_YAML), Status::Ok);
  ASSERT_EQ(c.initialize(CONFIG_YAML), Status::Ok);
  const FootPositions feet = feetUnderHips(a.config());
  const Command cmd{0.4, 0.0, 0.3};
  // constant contact: (b) == (a)
  const Contact pair{true, false, false, true};
  ASSERT_EQ(a.run(standing(), cmd, pair, feet), Status::Ok);
  ASSERT_EQ(b.run(standing(), cmd, std::vector<Contact>(16, pair), feet), Status::Ok);
  for (int leg = 0; leg < 4; ++leg) EXPECT_EQ(a.getSolution().forces[leg], b.getSolution().forces[leg]);
  // gait schedule: (b) == (c) with the predicted foot plan
  ASSERT_EQ(b.initialize(CONFIG_YAML), Status::Ok);
  ASSERT_EQ(b.run(standing(), cmd, trotSchedule(), feet), Status::Ok);
  ASSERT_EQ(c.run(standing(), cmd, trotSchedule(), predict(c.config(), standing(), cmd, trotSchedule(), feet)), Status::Ok);
  for (int leg = 0; leg < 4; ++leg) EXPECT_EQ(b.getSolution().forces[leg], c.getSolution().forces[leg]);
}

TEST(Run, FormBInvalidInput) {
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  FootPositions feet = feetUnderHips(mpc.config());
  EXPECT_EQ(mpc.run(standing(), Command{}, trotSchedule(15), feet), Status::InvalidInput);
  EXPECT_EQ(mpc.run(standing(), Command{}, std::vector<Contact>{}, feet), Status::InvalidInput);
  feet[3].x() = std::nan("");
  EXPECT_EQ(mpc.run(standing(), Command{}, trotSchedule(), feet), Status::InvalidInput);
}

// ---------------- comparison with the Python acados interface ----------------

// test/reference/closed_loop.txt is written by test/generate_reference.py: a closed loop (stand, trot in place,
// forward, forward + turn) with an independent Python version of run() form (b) on the acados Python interface.
// The C++ class must reproduce every step: status, acados status, QP iterations and forces.
TEST(PythonReference, ClosedLoop) {
  std::ifstream file(REFERENCE_DIR "/closed_loop.txt");
  ASSERT_TRUE(file.good()) << "missing " REFERENCE_DIR "/closed_loop.txt (run test/generate_reference.py)";
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok) << mpc.errorMessage();
  const int n = mpc.horizonLength();

  int steps = 0, failures = 0;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream in(line);
    int step;
    State s;
    Command cmd;
    std::string schedule_hex;
    FootPositions feet;
    int status, acados_status, qp_iterations;
    std::array<double, 12> forces;
    in >> step;
    for (Eigen::Vector3d* v : {&s.rpy, &s.position, &s.angular_velocity, &s.linear_velocity}) in >> (*v)[0] >> (*v)[1] >> (*v)[2];
    in >> cmd.vx >> cmd.vy >> cmd.yaw_rate >> schedule_hex;
    for (auto& f : feet) in >> f[0] >> f[1] >> f[2];
    in >> status >> acados_status >> qp_iterations;
    for (double& f : forces) in >> f;
    ASSERT_FALSE(in.fail()) << "cannot parse line: " << line.substr(0, 60);
    ASSERT_EQ(static_cast<int>(schedule_hex.size()), n) << "step " << step;

    std::vector<Contact> schedule(n);
    for (int k = 0; k < n; ++k) {
      const int bits = std::stoi(schedule_hex.substr(k, 1), nullptr, 16);
      for (int leg = 0; leg < 4; ++leg) schedule[k][leg] = (bits >> leg) & 1;
    }

    const Status st = mpc.run(s, cmd, schedule, feet);
    const auto& sol = mpc.getSolution();
    ASSERT_EQ(static_cast<int>(st), status) << "step " << step << ": " << mpc.errorMessage();
    EXPECT_EQ(sol.acados_status, acados_status) << "step " << step;
    EXPECT_EQ(sol.qp_iterations, qp_iterations) << "step " << step;
    for (int i = 0; i < 12; ++i)
      ASSERT_NEAR(sol.forces[i / 3][i % 3], forces[i], 1e-9 * (1.0 + std::abs(forces[i])))
          << "step " << step << ", force component " << i;
    ++steps;
    failures += st == Status::SolverFailed;
  }
  EXPECT_GE(steps, 100) << "reference file too short";
  std::cout << "[ python   ] " << steps << " steps identical to the Python reference (" << failures
            << " solver failure(s) with fallback)" << std::endl;
}
