"""Step 11: minimal closed-loop check of the NMPC (sanity check, not a deliverable).

The MPC from step 10 runs in a loop, as it will in the real control framework:
    every DT = 25 ms:  measure state -> update contact schedule + footholds -> one SQP_RTI iteration
                       -> apply u_0 to the plant -> simulate DT

Plant ("real robot"): the SRBD model of step 7 (no legs), integrated with RK4 in 4 substeps per DT,
i.e. finer than the MPC's own prediction (1 RK4 step per DT). No swing-leg controller is needed:
a swing foot has zero force, so where it is during swing does not matter; at touchdown it is
simply placed at its new foothold.

Scenario (7 s), see SCENARIO below:
    stand | trot in place | trot forward 0.5 m/s | forward + turn 0.6 rad/s | forward on the new heading
"""
import numpy as np
from utils import plots                       # plotting helpers (derivation/utils/plots.py)

from step7_casadi_model import f_dyn, G0
from step8_constraints import u_bounds, cone_fun, MU, F_MIN, F_MAX, N_POLY
from step9_cost import make_y_ref
from step10_acados_ocp import build_solver, N, DT, MASS, INERTIA, HIPS, Z_DES, IDX_FZ, NY_E, FEET

# ---------------- scenario ----------------
# One row per phase: (start time [s], v_x, v_y, yaw rate, label). Commands are in the HEADING frame
# (v_x = forward, v_y = left), as in the task interface. The gait (stand -> trot) starts at T_TROT.
SCENARIO = [
    (0.0, 0.0, 0.0, 0.0, 'stand'),
    (0.5, 0.0, 0.0, 0.0, 'trot in place'),
    (1.5, 0.5, 0.0, 0.0, 'forward'),
    (3.5, 0.5, 0.0, 0.6, 'forward + turn'),
    (5.5, 0.5, 0.0, 0.0, 'forward, new heading'),
]
T_SIM = 7.0                                  # simulated time [s]
N_SIM = int(round(T_SIM / DT))               # 280 control steps
T_TROT = 0.5                                 # trot starts at 0.5 s (before that: all feet down)
PHASE = 4                                    # control steps per trot half-cycle (4 x 25 ms = 0.1 s)
T_STANCE = PHASE * DT                        # stance duration of one foot [s]
J_TROT = int(round(T_TROT / DT))             # control step at which the trot starts
#        FR FL RR RL
PAIR_A = [1, 0, 0, 1]                        # FR + RL on the ground
PAIR_B = [0, 1, 1, 0]                        # FL + RR on the ground


def contact_at(j):
    """Contact flags [c_FR, c_FL, c_RR, c_RL] at control step j (gait clock in absolute time)."""
    if j < J_TROT:
        return [1, 1, 1, 1]
    return PAIR_A if ((j - J_TROT) // PHASE) % 2 == 0 else PAIR_B


def phase_at(j):
    """Index of the SCENARIO row active at control step j."""
    t = j * DT
    return max(r for r, row in enumerate(SCENARIO) if row[0] <= t + 1e-9)


def command_at(j):
    """Operator command (vx, vy, yaw_rate) in the heading frame at control step j."""
    return SCENARIO[phase_at(j)][1:4]


def Rz2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def foothold(com_xy, yaw, vel_xy, leg):
    """Paper Eq. 33: put the foot under its hip, plus half the distance the body travels during
    stance (v * T_stance / 2), so the body passes over the foot symmetrically. Flat ground, z = 0."""
    xy = com_xy + Rz2(yaw) @ HIPS[leg] + vel_xy * T_STANCE / 2
    return np.array([xy[0], xy[1], 0.0])


def horizon_feet(i, x, stance_pos):
    """World foot positions for the MPC nodes k = 0..N, seen from control step i.
    - foot in its CURRENT stance: the position where it touched down (it does not move)
    - foot in a FUTURE stance:    predicted foothold, assuming the body moves with the command
    - foot in swing:              irrelevant (zero force); the last stance position is used"""
    feet = np.zeros((N + 1, 4, 3))
    for k in range(N + 1):
        j = i + min(k, N - 1)                # node N belongs to the last interval
        for leg in range(4):
            if contact_at(j)[leg] == 0:
                feet[k, leg] = stance_pos[leg]
                continue
            s = j                            # find the touchdown step s of this stance phase
            while s > i and contact_at(s - 1)[leg] == 1:
                s -= 1
            if s == i:                       # current stance -> fixed position
                feet[k, leg] = stance_pos[leg]
            else:                            # future stance -> predict where the body will be
                com_pred, yaw_pred = x[3:5].copy(), x[2]
                for m in range(i, s):        # follow the commanded (possibly turning) path
                    vx, vy, wz = command_at(m)
                    com_pred = com_pred + Rz2(yaw_pred) @ np.array([vx, vy]) * DT
                    yaw_pred += wz * DT
                v_ref = Rz2(yaw_pred) @ np.array(command_at(s)[:2])  # command in world frame
                feet[k, leg] = foothold(com_pred, yaw_pred, v_ref, leg)
    return feet


def plant_step(x, u, feet_now):
    """The 'real robot': integrate the SRBD model over DT with RK4 in 4 substeps."""
    p = np.concatenate([feet_now.ravel(), [MASS], INERTIA])
    h = DT / 4
    for _ in range(4):
        k1 = f_dyn(x, u, p).full().ravel()
        k2 = f_dyn(x + h / 2 * k1, u, p).full().ravel()
        k3 = f_dyn(x + h / 2 * k2, u, p).full().ravel()
        k4 = f_dyn(x + h * k3, u, p).full().ravel()
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def mpc_step(solver, i, x, stance_pos, first):
    """Fill in the runtime data for control step i, warm start, run ONE SQP_RTI iteration."""
    feet = horizon_feet(i, x, stance_pos)
    for k in range(N + 1):
        solver.set(k, 'p', np.concatenate([feet[k].ravel(), [MASS], INERTIA, [MU]]))
    for k in range(N):
        lb, ub = u_bounds(contact_at(i + k))                 # contact schedule over the horizon
        solver.constraints_set(k, 'lbu', np.array(lb))
        solver.constraints_set(k, 'ubu', np.array(ub))
        solver.set(k, 'yref', np.array(make_y_ref(*command_at(i + k), Z_DES)))   # command preview
    solver.set(N, 'yref', np.array(make_y_ref(*command_at(i + N), Z_DES))[:NY_E])

    if first:
        # cold start: body stays, stance feet share the weight
        for k in range(N):
            c = np.array(contact_at(i + k), dtype=float)
            u_guess = np.zeros(12)
            u_guess[IDX_FZ] = c * MASS * G0 / max(c.sum(), 1)
            solver.set(k, 'u', u_guess)
            solver.set(k, 'x', x)
        solver.set(N, 'x', x)
    else:
        # warm start: shift the previous plan one step forward in time
        X = [solver.get(k, 'x') for k in range(N + 1)]
        U = [solver.get(k, 'u') for k in range(N)]
        for k in range(N):
            solver.set(k, 'x', X[k + 1])
            solver.set(k, 'u', U[min(k + 1, N - 1)])
        solver.set(N, 'x', X[N])

    solver.set(0, 'lbx', x)                  # x_0 = measured state
    solver.set(0, 'ubx', x)
    status = solver.solve()                  # SQP_RTI: one SQP iteration
    return status, solver.get(0, 'u'), 1000 * solver.get_stats('time_tot')


if __name__ == '__main__':
    solver = build_solver('polygon', nlp_solver_type='SQP_RTI', tag='_rti')

    x = np.array([0, 0, 0,  0, 0, Z_DES,  0, 0, 0,  0, 0, 0], dtype=float)
    stance_pos = np.array([foothold(x[3:5], 0.0, np.zeros(2), leg) for leg in range(4)])

    log_x, log_u, log_t, log_status = [x.copy()], [], [], []
    log_feet, log_c = [], []
    for i in range(N_SIM):
        # touchdown: a foot that just landed gets its new foothold (from the measured state)
        for leg in range(4):
            if contact_at(i)[leg] == 1 and i > 0 and contact_at(i - 1)[leg] == 0:
                stance_pos[leg] = foothold(x[3:5], x[2], x[9:11], leg)

        status, u0, t_ms = mpc_step(solver, i, x, stance_pos, first=(i == 0))
        x = plant_step(x, u0, stance_pos)

        log_x.append(x.copy()); log_u.append(u0); log_t.append(t_ms); log_status.append(status)
        log_feet.append(stance_pos.copy()); log_c.append(contact_at(i))

    X, U, T = np.array(log_x), np.array(log_u), np.array(log_t)
    FEET_POS, C = np.array(log_feet), np.array(log_c)
    time = np.arange(N_SIM + 1) * DT

    # ---------------- summary ----------------
    CMD = np.array([command_at(i) for i in range(N_SIM)])          # (N_SIM x 3) commands
    VH = np.array([Rz2(X[i, 2]).T @ X[i, 9:11] for i in range(N_SIM + 1)])   # heading-frame v
    trot = time[:-1] >= T_TROT
    used, swing_f = 0.0, 0.0
    for i in range(N_SIM):
        for leg, c in enumerate(contact_at(i)):
            f = U[i, 3*leg:3*leg+3]
            if c:
                used = max(used, np.hypot(f[0], f[1]) / f[2])
            else:
                swing_f = max(swing_f, np.abs(f).max())
    print("\n================ closed loop summary ================")
    print(f"solver status != 0      : {sum(s != 0 for s in log_status)} of {N_SIM} steps")
    print(f"solve time per step     : median {np.median(T):.2f} ms, max {T.max():.2f} ms "
          f"(budget {1000*DT:.0f} ms)")
    print("tracking per phase (second half of each phase; heading frame):")
    print(f"  {'phase':<22} {'v_x':>13} {'v_y':>13} {'yaw rate':>15}")
    for r, (t0, vx, vy, wz, label) in enumerate(SCENARIO):
        t1 = SCENARIO[r + 1][0] if r + 1 < len(SCENARIO) else T_SIM
        sel = (time[:-1] >= (t0 + t1) / 2) & (time[:-1] < t1)
        print(f"  {label:<22} {VH[:-1][sel, 0].mean():6.3f} ({vx:4.2f}) {VH[:-1][sel, 1].mean():6.3f} ({vy:4.2f})"
              f" {X[:-1][sel, 8].mean():7.3f} ({wz:4.2f})")
    print(f"height during trot      : {X[1:][trot, 5].min():.3f} .. {X[1:][trot, 5].max():.3f} m "
          f"(target {Z_DES})")
    print(f"max |roll|, |pitch|     : {np.degrees(np.abs(X[:, 0]).max()):.2f} deg, "
          f"{np.degrees(np.abs(X[:, 1]).max()):.2f} deg")
    print(f"total heading change    : {np.degrees(X[-1, 2]):.1f} deg "
          f"(commanded {np.degrees(np.sum(CMD[:, 2]) * DT):.1f} deg)")
    print(f"applied forces          : max |f_t|/f_z = {used:.3f} (mu = {MU}), "
          f"max |f| of a swing foot = {swing_f:.2e} N")

    print(f"solve time distribution : min {T.min():.2f}, median {np.median(T):.2f}, "
          f"95% {np.percentile(T, 95):.2f}, 99% {np.percentile(T, 99):.2f}, max {T.max():.2f} ms")

    # ---------------- plots (utils/plots.py) ----------------
    events = [row[0] for row in SCENARIO[1:]]                     # phase changes
    # save the run for other tools (e.g. showcase/render_3d.py); *.npz is gitignored
    np.savez('step11_results.npz', time=time, X=X, U=U, C=C, FEET_POS=FEET_POS, T=T,
             cmd=CMD, vx_cmd=CMD[:, 0], hips=HIPS, dt=DT, mass=MASS, z_des=Z_DES)
    print("stored results in step11_results.npz")
    plots.plot_overview(time, X, U, T, CMD, Z_DES, events,
                        'Step 11: closed loop (SQP_RTI, 16-gon friction) | stand / trot / forward / turn',
                        'step11_closed_loop.png')
    plots.plot_foot_forces(time, U, C, MU, F_MIN, F_MAX, events, 'step11_foot_forces.png', n_poly=N_POLY)
    plots.plot_foot_forces(time, U, C, MU, F_MIN, F_MAX, events, 'step11_foot_forces_zoom.png',
                           n_poly=N_POLY, t_window=(0.4, 1.6))
    plots.animate_robot(X, U, C, FEET_POS, HIPS, DT, MASS, 'step11_animation.gif', cmd=CMD[:, 0])
    plots.plot_solve_time_histogram(T, N, 'step11_solve_time.png')
