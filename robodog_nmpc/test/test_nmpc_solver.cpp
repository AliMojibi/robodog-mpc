// Tests of the NmpcSolver class.
#include <gtest/gtest.h>

#include <Eigen/Geometry>  // cross()

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
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

TEST(Run, Timing) {
  NmpcSolver mpc;
  ASSERT_EQ(mpc.initialize(CONFIG_YAML), Status::Ok);
  std::vector<double> t;
  int failures = 0;
  for (int i = 0; i < 1000; ++i) {
    const Status st = harshStep(mpc, i);
    ASSERT_TRUE(st == Status::Ok || st == Status::SolverFailed) << "step " << i;
    failures += st == Status::SolverFailed;
    for (const auto& f : mpc.getSolution().forces) ASSERT_TRUE(f.allFinite()) << "step " << i;
    t.push_back(mpc.getSolution().solve_time_ms);
  }
  std::sort(t.begin(), t.end());
  std::cout << "[ timing   ] 1000 steps (harsh scenario, " << failures << " solver failures): median " << t[500]
            << " ms, 99% " << t[990] << " ms, max " << t.back() << " ms" << std::endl;
  EXPECT_LT(t[500], 5.0);   // generous: ~1.5 ms expected on a desktop CPU
}
