#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# =====================================================================================
# robodog-mpc notes: what this example does
# =====================================================================================
# System: a pendulum (ball on a rod) mounted on a cart. We push the cart with a force F.
#
#   state   x = [ p, theta, v, theta_dot ]     (nx = 4)
#               p         cart position            [m]
#               theta     pendulum angle, 0 = upright, pi = hanging down   [rad]
#               v         cart velocity            [m/s]
#               theta_dot pendulum angular velocity [rad/s]
#   input   u = [ F ]                           (nu = 1)  horizontal force on the cart [N]
#   model   x_dot = f(x, u)   nonlinear (sin/cos of theta), see pendulum_model.py
#
# Task: start hanging down (theta = pi) and swing the pendulum up to theta = 0, cart back at p = 0.
#
# The optimal control problem (OCP) solved at EVERY step of the closed loop:
#
#   min over x_0..x_N, u_0..u_{N-1}
#         sum_{k=0}^{N-1}  1/2 * || y(x_k, u_k) - y_ref ||^2_W      (stage cost)
#       +                  1/2 * || y_e(x_N)    - y_ref_e ||^2_W_e  (terminal cost)
#
#   s.t.  x_0     = x_measured                  (start from the current state)
#         x_{k+1} = F(x_k, u_k)                 (dynamics; acados integrates f over one interval)
#         -Fmax  <= u_k <= Fmax                 (input bounds)
#
#   with  y(x, u) = [x; u],  y_ref = 0          -> drive all states and the force to 0
#         y_e(x)  = x,       y_ref_e = 0
#
# This is a nonlinear program (NLP). acados transcribes it with multiple shooting
# (x_k and u_k are all decision variables) and solves it with SQP: repeatedly linearize
# around the current guess and solve a QP with HPIPM.
#
# Closed loop (MPC): solve the OCP, apply only u_0 to the "real" pendulum (here a simulator),
# measure the new state, and solve again. Repeat Nsim times.
# =====================================================================================

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from pendulum_model import export_pendulum_ode_model
from utils import plot_pendulum
import numpy as np
import scipy.linalg
from casadi import vertcat

def setup_ocp_solver(x0, Fmax, N_horizon, Tf):
    # create ocp object to formulate the OCP
    # (AcadosOcp is only a DESCRIPTION of the problem; the solver is generated at the end)
    ocp = AcadosOcp()

    # set model: x_dot = f(x, u), written in CasADi (pendulum_model.py)
    model = export_pendulum_ode_model()
    ocp.model = model

    nx = model.x.rows()     # 4 states
    nu = model.u.rows()     # 1 input
    ny = nx + nu            # size of the stage cost vector y = [x; u]
    ny_e = nx               # size of the terminal cost vector y_e = x (no u at the last node)

    # set cost
    # NONLINEAR_LS: cost = 1/2 * || y(x,u) - y_ref ||^2_W, where y(x,u) is any CasADi expression.
    # (here y is simply [x; u], so LINEAR_LS would also work; for our robot y is nonlinear)
    ocp.cost.cost_type = 'NONLINEAR_LS'      # stage cost, nodes k = 0 .. N-1
    ocp.cost.cost_type_e = 'NONLINEAR_LS'    # terminal cost, node k = N

    # weights. The factor 2 cancels the 1/2 that acados puts in front of the cost.
    #                    p     theta   v     theta_dot
    Q_mat = 2*np.diag([1e3,   1e3,   1e-2,  1e-2])   # strongly penalize position and angle errors
    R_mat = 2*np.diag([1e-2])                        # small penalty on the force F

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)   # W = diag(Q, R), size ny x ny
    ocp.cost.W_e = Q_mat                                 # terminal weight, size ny_e x ny_e

    ocp.model.cost_y_expr = vertcat(model.x, model.u)    # y(x, u) = [x; u]
    ocp.model.cost_y_expr_e = model.x                    # y_e(x)  = x
    ocp.cost.yref = np.zeros((ny, ))                     # target: everything 0 (upright, cart at 0)
    ocp.cost.yref_e = np.zeros((ny_e, ))

    # set constraints
    # input bounds:  lbu <= u[idxbu] <= ubu   ->   -Fmax <= F <= Fmax
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])   # which entries of u are bounded (F is entry 0)

    # initial-state constraint x_0 = x0. This value is only a placeholder at code generation;
    # in the loop it is overwritten every step with the measured state (solve_for_x0 below).
    ocp.constraints.x0 = x0

    # set prediction horizon
    # Tf = horizon length in seconds, N = number of shooting intervals -> dt = Tf/N = 0.02 s
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = Tf

    # set ocp options
    # GAUSS_NEWTON: Hessian approximated by J^T W J (J = dy/d(x,u)); cheap and always convex QPs.
    # Not set here, so acados defaults are used:
    #   nlp_solver_type = 'SQP'                         (iterate SQP until converged)
    #   qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    #   integrator_type = 'ERK'                         (explicit Runge-Kutta 4)
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.qp_tol = 1e-8

    # generate C code for the solver into this folder, compile it, and load it
    ocp.code_export_directory = 'c_generated_code_ocp'
    ocp_solver = AcadosOcpSolver(ocp)

    return ocp, ocp_solver


def setup_integrator(ocp: AcadosOcp):
    # A separate acados SIMULATOR that plays the role of the real pendulum.
    # It uses the same model and the same dt as the OCP...
    sim = AcadosSim.from_ocp(ocp)

    # ...but 2 integration steps per interval instead of 1, so the "plant" is a bit more accurate
    # than the MPC's internal prediction model (a small model mismatch, as in reality).
    sim.solver_options.num_steps = 2 # make integrator more precise than the integrator used within the OCP
    sim.code_export_directory = 'c_generated_code_sim'
    integrator = AcadosSimSolver(sim)
    return integrator


def main():

    x0 = np.array([0.0, np.pi, 0.0, 0.0])   # start: cart at 0, pendulum hanging down, at rest
    Fmax = 80                                # force limit [N]

    Tf = .8          # prediction horizon [s]
    N_horizon = 40   # number of intervals -> dt = 0.02 s

    ocp, ocp_solver = setup_ocp_solver(x0, Fmax, N_horizon, Tf)
    integrator = setup_integrator(ocp)

    nx = ocp_solver.ocp.dims.nx
    nu = ocp_solver.ocp.dims.nu

    Nsim = 450                          # closed-loop steps: 100 * 0.02 s = 2 s of simulated time
    simX = np.zeros((Nsim+1, nx))       # state history
    simU = np.zeros((Nsim, nu))         # applied-input history

    simX[0,:] = x0

    t = np.zeros((Nsim))                # solver time per step

    # closed loop
    for i in range(Nsim):

        # solve ocp and get next control input
        # solve_for_x0 = set x_0 = current state, solve the whole OCP over the horizon,
        # and return only the FIRST input u_0 (receding horizon).
        # The previous solution stays inside the solver and is used as the initial guess
        # (warm start), which is why later solves are much faster than the first one.
        simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :])

        t[i] = ocp_solver.get_stats('time_tot')

        # simulate system: apply u_0 for one interval (dt) to the "real" pendulum
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

    # evaluate timings
    # scale to milliseconds
    t *= 1000
    print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')

    # plot results
    model = ocp_solver.ocp.model
    plot_pendulum(np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1), Fmax, simU, simX, latexify=False, time_label=model.t_label, x_labels=model.x_labels, u_labels=model.u_labels)

    ocp_solver = None


if __name__ == '__main__':
    main()
