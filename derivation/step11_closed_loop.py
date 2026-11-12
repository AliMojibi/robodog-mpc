"""Step 11: minimal closed-loop check of the NMPC (sanity check, not a deliverable).

The MPC from step 10 runs in a loop, as it will in the real control framework:
    every DT = 25 ms:  measure state -> update contact schedule + footholds -> one SQP_RTI iteration
                       -> apply u_0 to the plant -> simulate DT

Plant ("real robot"): the SRBD model of step 7 (no legs), integrated with RK4 in 4 substeps per DT,
i.e. finer than the MPC's own prediction (1 RK4 step per DT). No swing-leg controller is needed:
a swing foot has zero force, so where it is during swing does not matter; at touchdown it is
simply placed at its new foothold.

Scenario (5 s):  0.0-0.5 s stand  |  0.5-1.0 s trot in place  |  1.0-5.0 s trot forward, v_x = 0.5 m/s
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')                        # no display in WSL: save the figure to a PNG
import matplotlib.pyplot as plt

from step7_casadi_model import f_dyn, G0
from step8_constraints import u_bounds, cone_fun, MU
from step9_cost import make_y_ref
from step10_acados_ocp import build_solver, N, DT, MASS, INERTIA, HIPS, Z_DES, IDX_FZ, NY_E, FEET

# ---------------- scenario ----------------
T_SIM = 5.0                                  # simulated time [s] (200 control steps -> 200 solve times)
N_SIM = int(round(T_SIM / DT))               # 100 control steps
T_TROT, T_FWD, VX_CMD = 0.5, 1.0, 0.5        # trot starts at 0.5 s; forward command from 1.0 s
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


def command_at(j):
    """Operator command (vx, vy, yaw_rate) in the heading frame at control step j."""
    return (VX_CMD if j * DT >= T_FWD else 0.0, 0.0, 0.0)


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
                v_ref = Rz2(x[2]) @ np.array(command_at(s)[:2])     # command in world frame
                com_pred = x[3:5] + v_ref * (s - i) * DT
                feet[k, leg] = foothold(com_pred, x[2], v_ref, leg)
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
    for i in range(N_SIM):
        # touchdown: a foot that just landed gets its new foothold (from the measured state)
        for leg in range(4):
            if contact_at(i)[leg] == 1 and i > 0 and contact_at(i - 1)[leg] == 0:
                stance_pos[leg] = foothold(x[3:5], x[2], x[9:11], leg)

        status, u0, t_ms = mpc_step(solver, i, x, stance_pos, first=(i == 0))
        x = plant_step(x, u0, stance_pos)

        log_x.append(x.copy()); log_u.append(u0); log_t.append(t_ms); log_status.append(status)

    X, U, T = np.array(log_x), np.array(log_u), np.array(log_t)
    time = np.arange(N_SIM + 1) * DT

    # ---------------- summary ----------------
    fwd = time[:-1] >= T_SIM - 0.5            # last 0.5 s, steady forward trot
    trot = time[:-1] >= T_TROT
    used, swing_f = 0.0, 0.0
    for i in range(N_SIM):
        h = cone_fun(U[i], MU).full().ravel()
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
    print(f"v_x, last 0.5 s         : mean {X[1:][fwd, 9].mean():.3f} m/s (command {VX_CMD}), "
          f"std {X[1:][fwd, 9].std():.3f}")
    print(f"v_y, last 0.5 s         : mean {X[1:][fwd, 10].mean():.3f} m/s (command 0)")
    print(f"height during trot      : {X[1:][trot, 5].min():.3f} .. {X[1:][trot, 5].max():.3f} m "
          f"(target {Z_DES})")
    print(f"max |roll|, |pitch|     : {np.degrees(np.abs(X[:, 0]).max()):.2f} deg, "
          f"{np.degrees(np.abs(X[:, 1]).max()):.2f} deg")
    print(f"yaw drift at the end    : {np.degrees(X[-1, 2]):.2f} deg")
    print(f"applied forces          : max |f_t|/f_z = {used:.3f} (mu = {MU}), "
          f"max |f| of a swing foot = {swing_f:.2e} N")

    # ---------------- plot ----------------
    fig, ax = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    ax[0].plot(time, X[:, 9], label='v_x'); ax[0].plot(time, X[:, 10], label='v_y')
    ax[0].plot(time[:-1], [command_at(i)[0] for i in range(N_SIM)], 'k--', label='v_x command')
    ax[0].set_ylabel('velocity [m/s]'); ax[0].legend(loc='upper left')
    ax[1].plot(time, X[:, 5]); ax[1].axhline(Z_DES, color='k', ls='--')
    ax[1].set_ylabel('height z [m]')
    ax[2].plot(time, np.degrees(X[:, 0]), label='roll'); ax[2].plot(time, np.degrees(X[:, 1]), label='pitch')
    ax[2].plot(time, np.degrees(X[:, 2]), label='yaw'); ax[2].set_ylabel('angle [deg]'); ax[2].legend(loc='upper left')
    for leg in range(4):
        ax[3].step(time[:-1], U[:, IDX_FZ[leg]], where='post', label=FEET[leg])
    ax[3].set_ylabel('f_z [N]'); ax[3].legend(loc='upper left', ncol=4)
    ax[4].plot(time[:-1], T); ax[4].set_ylabel('solve time [ms]'); ax[4].set_xlabel('time [s]')
    for a in ax:
        a.axvline(T_TROT, color='gray', lw=0.8); a.axvline(T_FWD, color='gray', lw=0.8); a.grid(alpha=0.3)
    fig.suptitle('Step 11: closed loop (SQP_RTI, 16-gon friction)  |  stand / trot in place / trot forward')
    fig.tight_layout()
    fig.savefig('step11_closed_loop.png', dpi=90)
    print("stored figure in step11_closed_loop.png")

    # ---------------- solve-time distribution (like Fig. 4 of the paper) ----------------
    # time_tot from acados = time of the C solver only (linearization + QP), one SQP_RTI iteration
    p50, p95, p99 = np.percentile(T, [50, 95, 99])
    print(f"\nsolve time distribution over {N_SIM} MPC solves: "
          f"min {T.min():.2f}, median {p50:.2f}, 95% {p95:.2f}, 99% {p99:.2f}, max {T.max():.2f} ms")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(T, bins=np.arange(0, np.ceil(p99) + 0.25, 0.25), color='tab:blue', edgecolor='black')
    if T.max() > np.ceil(p99):
        ax2.text(0.98, 0.95, f'{(T > np.ceil(p99)).sum()} solve(s) above {np.ceil(p99):.0f} ms\n(max {T.max():.1f} ms)',
                 transform=ax2.transAxes, ha='right', va='top', fontsize=9)
    ax2.set_xlabel('Solve Time (ms)'); ax2.set_ylabel('Frequency')
    ax2.set_title(f'Solve Time (SQP_RTI, N = {N}, {N_SIM} solves)')
    fig2.tight_layout()
    fig2.savefig('step11_solve_time.png', dpi=90)
    print("stored figure in step11_solve_time.png")
