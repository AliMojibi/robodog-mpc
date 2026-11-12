"""Step 10: the quadruped NMPC as an acados OCP (first real solve).

The optimal control problem solved here (one solve = one MPC plan over the horizon):

    min   sum_{k=0}^{N-1} 1/2 ||y(x_k, u_k) - y_ref||^2_W  +  1/2 ||y_e(x_N) - y_ref_e||^2_W_e
    s.t.  x_0     = x_measured                          (start from the current robot state)
          x_{k+1} = RK4( f(x_k, u_k, p_k), dt )         (SRBD dynamics, step 7)
          friction, per stance foot i:                  (step 8)
              polygon: inscribed 16-gon of the cone (DEFAULT)
                       cos(th_j) f_x + sin(th_j) f_y - mu cos(pi/16) f_z <= 0,  j = 0..15
                       linear in u -> fast, robust SQP; never exceeds mu; at most 1.9 % conservative
              cone   : sqrt(f_x^2 + f_y^2 + eps^2) - mu f_z <= 0     (exact, nonlinear; available,
                       but converges slowly / not at all when friction is the limit)
              pyramid: |f_x| <= mu f_z, |f_y| <= mu f_z               (paper, linear; lies OUTSIDE
                       the cone, allows up to 1.41 mu -> slipping; for comparison only)
          contact, per foot i:
              stance : f_min <= f_z <= f_max
              swing  : f = 0  (all three components; the cone is switched off for that foot)

    x = [Theta, p, omega, v]  (12)     u = [f_FR, f_FL, f_RR, f_RL]  (12, world-frame GRFs)
    p = [p_foot (12), m, Ixx, Iyy, Izz, mu]  (17)

Reuses:  step 7  x, u, P, x_dot   (model, P = [p_foot(12), m, Ixx, Iyy, Izz])
         step 8  polygon (16-gon), cone, mu, u_bounds, cone_upper;  D (pyramid);  F_MIN, F_MAX
         step 9  y, W, make_y_ref (least-squares cost)

What is fixed at code generation vs. what can change at every solve:
  fixed   : N, dt, the model structure, y(x,u), which friction model (polygon / cone / pyramid)
            (pyramid only: also mu, because it sits inside the matrix D)
  runtime : per stage k (no code regeneration needed)
              p_k      foot positions, mass, inertia, mu           -> solver.set(k, 'p', ...)
              contact  bounds on all 12 forces                     -> solver.constraints_set(k, 'lbu'/'ubu', ...)
                       cone on/off per foot                        -> solver.constraints_set(k, 'uh', ...)
              y_ref    operator command                            -> solver.set(k, 'yref', ...)
            and x_0 (the measured state)                           -> solver.set(0, 'lbx'/'ubx', ...)
"""
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from step7_casadi_model import x, u, P, x_dot, G0
from step8_constraints import (D, F_MIN, F_MAX, MU, BIG, cone, polygon, mu as MU_SYM, cone_fun,
                               u_bounds, cone_upper)
from step9_cost import y, W, make_y_ref

# ---------------- settings ----------------
N, TF = 16, 0.4                       # horizon: 16 intervals over 0.4 s -> dt = 0.025 s (paper: 10-16 steps, 0.33-0.5 s)
DT = TF / N
MASS, INERTIA = 43.0, [0.41, 2.1, 2.1]   # Cheetah 3, paper Table I
# FOOT ORDER used everywhere (forces u, foot positions, contact flags):  0 = FR, 1 = FL, 2 = RR, 3 = RL
#   FR = front right, FL = front left, RR = rear right, RL = rear left
#   e.g. a contact row [1, 0, 0, 1] means FR and RL on the ground, FL and RR in the air
FEET = ['FR', 'FL', 'RR', 'RL']
HX, HY, Z_DES = 0.30, 0.15, 0.45      # hip offsets from the CoM [m] and desired body height [m]
HIPS = np.array([[HX, -HY], [HX, HY], [-HX, -HY], [-HX, HY]])   # x-y of the hips: FR, FL, RR, RL
IDX_FZ = [2, 5, 8, 11]                # positions of f_z in u = [f_FR_x, f_FR_y, f_FR_z, f_FL_x, ...]
NY_E = 7                              # terminal cost uses y[0:7]: velocities, yaw rate, height, roll, pitch
                                      # (no forces, because there is no input at the last node x_N)


def build_solver(friction='polygon', nlp_solver_type='SQP', globalization='FIXED_STEP'):
    """Describe the OCP with AcadosOcp, then generate + compile + load the C solver.
    This is the slow part (a few seconds); it only has to be done once.
    friction: 'polygon' (inscribed 16-gon, default), 'cone' (exact, nonlinear) or 'pyramid' (paper)."""

    # ---- model: x_dot = f(x, u, p) ----
    # acados needs the CasADi symbols (x, u, p) and the expression for x_dot.
    # 'ERK' below integrates this with Runge-Kutta 4 to get x_{k+1} = F(x_k, u_k, p_k),
    # and CasADi generates the derivatives (A_k, B_k) automatically.
    model = AcadosModel()
    model.name = f'srbd_{friction}'      # used in the names of the generated C files/functions
    model.x, model.u = x, u
    model.p = ca.vertcat(P, MU_SYM)      # 16 model parameters + mu (used by polygon and cone)
    model.f_expl_expr = x_dot            # explicit ODE  x_dot = f(x, u, p)

    ocp = AcadosOcp()
    ocp.model = model
    # Every stage has its own copy of p. This value is only a placeholder needed at
    # code generation; the real values are set per stage in solve().
    ocp.parameter_values = np.zeros(model.p.shape[0])

    # ---- cost: 1/2 ||y - y_ref||^2_W  (factor 2 cancels acados' 1/2) ----
    # NONLINEAR_LS: y(x, u) is a CasADi expression (step 9). It is nonlinear because of
    # the heading-frame velocity Rz(psi)^T v. acados uses the Gauss-Newton Hessian J^T W J.
    W_num = ca.evalf(W).full()           # step 9 built W in CasADi; acados wants a numpy matrix
    ocp.cost.cost_type = 'NONLINEAR_LS'  # stage cost, nodes k = 0 .. N-1 (node 0 uses the same by default)
    ocp.model.cost_y_expr = y
    ocp.cost.W = 2 * W_num
    ocp.cost.yref = np.zeros(y.shape[0])  # placeholder; the command is set per stage in solve()

    ocp.cost.cost_type_e = 'NONLINEAR_LS'               # terminal cost, node k = N
    ocp.model.cost_y_expr_e = y[:NY_E]                  # same as y, but without the 12 force entries
    ocp.cost.W_e = 2 * W_num[:NY_E, :NY_E]
    ocp.cost.yref_e = np.zeros(NY_E)

    # ---- constraints ----
    # (a) contact: bounds on ALL 12 force components, lbu <= u[idxbu] <= ubu.
    #     The defaults here (all feet in stance) are overwritten per stage in solve():
    #     stance: f_x, f_y free, f_min <= f_z <= f_max;  swing: f = 0.
    ocp.constraints.idxbu = np.arange(12)
    lb, ub = u_bounds([1, 1, 1, 1])
    ocp.constraints.lbu = np.array(lb)
    ocp.constraints.ubu = np.array(ub)

    # (b) friction
    if friction in ('polygon', 'cone'):
        # acados "nonlinear constraints": lh <= h(x, u, p) <= uh, linearized in every SQP iteration.
        #   polygon: h = 64 rows (16 faces x 4 feet). Linear in u for a given mu, so the
        #            linearization is exact; mu comes from p, so it is set at runtime.
        #   cone   : h = 4 rows (one per foot), truly nonlinear.
        # lh = -inf, uh = 0. (cone only: uh = BIG for swing feet, set in solve().)
        # IMPORTANT: node 0 has its own h (con_h_expr_0), which is empty by default.
        # Without it, friction would NOT be enforced on u_0 -- the force that is actually applied.
        h = polygon if friction == 'polygon' else cone
        nh = h.shape[0]
        ocp.model.con_h_expr_0 = h
        ocp.constraints.lh_0 = -BIG * np.ones(nh)
        ocp.constraints.uh_0 = np.zeros(nh)
        ocp.model.con_h_expr = h                        # nodes 1 .. N-1
        ocp.constraints.lh = -BIG * np.ones(nh)
        ocp.constraints.uh = np.zeros(nh)
    elif friction == 'pyramid':
        # acados "general linear constraints": lg <= C x + D u <= ug, with C = 0, ug = 0 -> D u <= 0.
        # A swing foot has f = 0 from (a), which satisfies D u <= 0, so nothing changes per stage.
        D_num = ca.evalf(D).full()
        ocp.constraints.C = np.zeros((D_num.shape[0], 12))
        ocp.constraints.D = D_num
        ocp.constraints.lg = -BIG * np.ones(D_num.shape[0])
        ocp.constraints.ug = np.zeros(D_num.shape[0])
    else:
        raise ValueError(friction)

    # (c) initial state x_0 = x0. Setting x0 tells acados that node 0 has an equality
    #     constraint on the full state; the value is replaced at every solve.
    ocp.constraints.x0 = np.zeros(12)

    # ---- solver options ----
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = TF                          # dt = tf / N
    ocp.solver_options.integrator_type = 'ERK'          # explicit Runge-Kutta (RK4 by default)
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # cheap, always gives convex QPs
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.nlp_solver_type = nlp_solver_type   # 'SQP' = iterate to convergence,
                                                            # 'SQP_RTI' = one iteration per call (closed loop)
    ocp.solver_options.nlp_solver_max_iter = 50
    # globalization:
    #   'FIXED_STEP'         full SQP steps (default). Measured fastest in every test.
    #   'MERIT_BACKTRACKING' line search on a merit function (step shortened if it makes things
    #                        worse). Measured 2-20x MORE iterations here and did not rescue the
    #                        exact cone, so it is off by default.
    ocp.solver_options.globalization = globalization

    # Generate the C code into this folder, compile it to a .so, and load it into Python.
    # (acados >= 0.5.6: output folder and json file are set in ocp.code_gen_options;
    #  json_file is a file NAME only, it is always written into the export folder)
    ocp.code_gen_options.code_export_directory = f'c_generated_code_srbd_{friction}'
    ocp.code_gen_options.json_file = 'acados_ocp.json'
    return AcadosOcpSolver(ocp, verbose=False)


def solve(solver, friction, x0, cmd, contact, feet):
    """Fill in all runtime data for one MPC solve, solve, and return the plan.

    cmd     = (vx, vy, yaw_rate)               operator command, heading frame
    contact : N x 4 flags                       one row per interval k = 0 .. N-1,
              contact[k] = [c_FR, c_FL, c_RR, c_RL], 1 = stance (foot on the ground), 0 = swing
    feet    : (N+1) x 4 x 3                     world foot positions per node k = 0 .. N,
              feet[k] = [p_FR, p_FL, p_RR, p_RL], each [x, y, z]
    returns status (0 = success), X ((N+1) x 12 predicted states), U (N x 12 planned forces)
    """
    y_ref = np.array(make_y_ref(*cmd, Z_DES))

    # parameters: every node (including the terminal node N) gets foot positions, mass, inertia, mu
    for k in range(N + 1):
        solver.set(k, 'p', np.concatenate([feet[k].ravel(), [MASS], INERTIA, [MU]]))

    for k in range(N):
        c = np.array(contact[k], dtype=float)
        # contact schedule -> bounds on the 12 forces of this interval (swing foot: f = 0)
        lb, ub = u_bounds(contact[k])
        solver.constraints_set(k, 'lbu', np.array(lb))
        solver.constraints_set(k, 'ubu', np.array(ub))
        # cone on for stance feet, off for swing feet (at k = 0 this sets uh_0 of con_h_expr_0).
        # Not needed for the polygon: with f = 0 every face reads 0 <= 0, which always holds.
        if friction == 'cone':
            solver.constraints_set(k, 'uh', np.array(cone_upper(contact[k])))
        # same command over the whole horizon
        solver.set(k, 'yref', y_ref)
        # initial guess for SQP: the robot stays where it is, and the stance feet share the weight equally.
        u_guess = np.zeros(12)
        u_guess[IDX_FZ] = c * MASS * G0 / max(c.sum(), 1)
        solver.set(k, 'u', u_guess)
        solver.set(k, 'x', x0)
    solver.set(N, 'yref', y_ref[:NY_E])  # terminal reference has only the 7 non-force entries
    solver.set(N, 'x', x0)

    # initial-state constraint x_0 = x0 (lower bound = upper bound = measured state)
    solver.set(0, 'lbx', x0)
    solver.set(0, 'ubx', x0)
    status = solver.solve()

    # read the whole plan; in a real controller only U[0] would be applied
    X = np.array([solver.get(k, 'x') for k in range(N + 1)])
    U = np.array([solver.get(k, 'u') for k in range(N)])
    return status, X, U


def friction_check(U, contact):
    """For every stage: largest friction use max ||f_t|| / f_z over stance feet, and the largest
    violation of the TRUE cone and of the contact bounds (<= 0 means satisfied)."""
    used, viol = np.zeros(N), np.zeros(N)
    for k in range(N):
        lb, ub = u_bounds(contact[k])
        h = cone_fun(U[k], MU).full().ravel()          # true cone, whatever the solver used
        v = [np.max(np.array(lb) - U[k]), np.max(U[k] - np.array(ub))]
        for i, c in enumerate(contact[k]):
            if c:
                fx, fy, fz = U[k, 3*i:3*i+3]
                used[k] = max(used[k], np.hypot(fx, fy) / fz)
                v.append(h[i])
        viol[k] = max(v)
    return used, viol


def report(name, solver, status, X, U, contact, table=True):
    """Print solver statistics, u_0, the predicted trajectory, and a friction/contact check per stage."""
    used, viol = friction_check(U, contact)
    print(f"\n================ {name} ================")
    print(f"status {status} (0 = success), SQP iterations {solver.get_stats('sqp_iter')}, "
          f"time {1000 * solver.get_stats('time_tot'):.2f} ms")
    print("u_0 (forces to apply now) [N]:")
    for i, n in enumerate(FEET):
        print(f"  {n}: [{U[0, 3*i]:8.2f} {U[0, 3*i+1]:8.2f} {U[0, 3*i+2]:8.2f}]")

    if table:
        # one row per node: contact flags of that interval (one column per foot, 1 = stance, - = swing),
        # predicted state, total vertical force, friction use (must stay <= mu = 0.6),
        # and whether the true cone + contact bounds hold
        print(f"\n  {'k':>2} {'t[s]':>5}  {' '.join(FEET)}   {'v_x':>6} {'v_y':>6} {'z':>6} "
              f"{'roll°':>6} {'pitch°':>6} {'Σf_z':>7} {'|ft|/fz':>8}  ok")
        for k in range(N):
            flags = '  '.join('1' if c else '-' for c in contact[k])
            print(f"  {k:>2} {k*DT:5.3f}   {flags}    {X[k,9]:6.3f} {X[k,10]:6.3f} "
                  f"{X[k,5]:6.3f} {np.degrees(X[k,0]):6.2f} {np.degrees(X[k,1]):6.2f} "
                  f"{U[k, IDX_FZ].sum():7.1f} {used[k]:8.3f}  {'yes' if viol[k] <= 1e-4 else 'NO'}")
        # last node: state only, there is no force after x_N
        print(f"  {N:>2} {N*DT:5.3f}                 {X[N,9]:6.3f} {X[N,10]:6.3f} {X[N,5]:6.3f}")
    print(f"max friction use |f_t|/f_z = {used.max():.3f}  (mu = {MU}),  "
          f"max violation of true cone / contact bounds = {viol.max():.2e} N")
    print(f"predicted velocity at the end of the horizon: v_x = {X[N,9]:.3f}, v_y = {X[N,10]:.3f} m/s")


def trot(vx_cmd, vy_cmd=0.0):
    """Trot contact schedule (FR+RL / FL+RR, 4 stages = 0.1 s each) and a simple foothold plan:
    during each stance phase the foot stays fixed under its hip at the middle of that phase,
    assuming the body moves at the commanded velocity. (Swing-foot positions do not matter:
    their force is 0, so their lever arm has no effect.)"""
    #         FR FL RR RL
    pair_a = [1, 0, 0, 1]     # FR + RL on the ground, FL + RR in the air
    pair_b = [0, 1, 1, 0]     # FL + RR on the ground, FR + RL in the air
    contact = [pair_a if (k // 4) % 2 == 0 else pair_b for k in range(N)]
    feet = []
    for k in range(N + 1):
        t_mid = ((min(k, N - 1) // 4) * 4 + 2) * DT     # mid-time of the phase that node k belongs to
        feet.append(np.column_stack([HIPS + [vx_cmd * t_mid, vy_cmd * t_mid], np.zeros(4)]))
    return contact, np.array(feet)


if __name__ == '__main__':
    solver = build_solver('polygon')         # inscribed 16-gon of the friction cone (default)

    # start: standing at the desired height, level, at rest
    #                Theta      p            omega         v
    x0 = np.array([0, 0, 0,  0, 0, Z_DES,  0, 0, 0,  0, 0, 0], dtype=float)

    # ---- test 1: standing, command zero ----
    # all four feet in contact for the whole horizon, feet directly under the hips on the ground (z = 0)
    contact = [[1, 1, 1, 1]] * N
    feet = np.array([np.column_stack([HIPS, np.zeros(4)])] * (N + 1))
    report("test 1 (polygon): standing, command zero", solver,
           *solve(solver, 'polygon', x0, (0, 0, 0), contact, feet), contact)

    # ---- test 2: trot, command v_x = 0.5 m/s ----
    contact, feet = trot(0.5)
    report("test 2 (polygon): trot, command v_x = 0.5 m/s", solver,
           *solve(solver, 'polygon', x0, (0.5, 0, 0), contact, feet), contact)

    # ---- test 3: hard diagonal acceleration -> friction is the limit. All three friction models ----
    # 1.5 m/s diagonally (|v| = 2.1 m/s) from rest: the tracking cost wants a large acceleration,
    # so both f_x and f_y want to be as large as friction allows -- the pyramid's corner.
    # Expected: polygon converges and never exceeds mu; cone does not converge (status 2);
    # pyramid converges but exceeds mu (|f_t|/f_z ~ 0.85 -> would slip).
    v_diag = 1.5
    contact, feet = trot(v_diag, v_diag)
    report(f"test 3 (polygon): trot, command v_x = v_y = {v_diag} m/s", solver,
           *solve(solver, 'polygon', x0, (v_diag, v_diag, 0), contact, feet), contact, table=False)

    for fr in ['cone', 'pyramid']:
        other = build_solver(fr)
        report(f"test 3 ({fr}): trot, command v_x = v_y = {v_diag} m/s", other,
               *solve(other, fr, x0, (v_diag, v_diag, 0), contact, feet), contact, table=False)
