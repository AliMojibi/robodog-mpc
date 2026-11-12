"""Plotting helpers for the closed-loop results (step 11 and later).

All functions take plain numpy arrays and save a PNG (WSL has no display).

Conventions (same as the rest of the project):
  X       : (n+1) x 12 states  [phi, theta, psi, p_x, p_y, p_z, w_x, w_y, w_z, v_x, v_y, v_z]
  U       : n x 12 applied forces  [f_FR, f_FL, f_RR, f_RL], each [f_x, f_y, f_z], world frame
  C       : n x 4 contact flags (1 = stance), foot order FR, FL, RR, RL
  FEET_POS: n x 4 x 3 world foot positions used at each step
  time    : (n+1) sample times [s]; forces/contacts belong to the intervals time[:-1]
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEET = ['FR', 'FL', 'RR', 'RL']
FOOT_COLORS = {'FR': 'tab:blue', 'FL': 'tab:orange', 'RR': 'tab:green', 'RL': 'tab:red'}


def _shade_swing(ax, t, c):
    """Grey background wherever the foot is in swing (c == 0)."""
    t_end = t[-1] + (t[1] - t[0])
    start = None
    for i, ci in enumerate(c):
        if ci == 0 and start is None:
            start = t[i]
        if ci == 1 and start is not None:
            ax.axvspan(start, t[i], color='0.88', lw=0)
            start = None
    if start is not None:
        ax.axvspan(start, t_end, color='0.88', lw=0)


def _events(ax, events):
    for te in events:
        ax.axvline(te, color='gray', lw=0.8, ls=':')


def plot_overview(time, X, U, T, cmd, z_des, events, title, path):
    """Body states, vertical forces and solve time over time (one column, shared time axis).
    cmd: n x 3 commands (v_x, v_y, yaw rate) in the HEADING frame; velocities are shown in the
    heading frame too (forward / left of the robot), so they stay comparable while turning."""
    psi = X[:, 2]
    vh_x = np.cos(psi) * X[:, 9] + np.sin(psi) * X[:, 10]
    vh_y = -np.sin(psi) * X[:, 9] + np.cos(psi) * X[:, 10]
    fig, ax = plt.subplots(6, 1, figsize=(9, 13), sharex=True)
    ax[0].plot(time, vh_x, label='v_x (forward)'); ax[0].plot(time, vh_y, label='v_y (left)')
    ax[0].step(time[:-1], cmd[:, 0], 'k--', where='post', label='v_x command')
    ax[0].step(time[:-1], cmd[:, 1], ':', color='0.4', where='post', label='v_y command')
    ax[0].set_ylabel('velocity, heading\nframe [m/s]'); ax[0].legend(loc='upper left', ncol=2, fontsize=8)
    ax[1].plot(time, X[:, 8], label='yaw rate ω_z')
    ax[1].step(time[:-1], cmd[:, 2], 'k--', where='post', label='command')
    ax[1].set_ylabel('yaw rate [rad/s]'); ax[1].legend(loc='upper left', fontsize=8)
    ax[2].plot(time, X[:, 5]); ax[2].axhline(z_des, color='k', ls='--')
    ax[2].set_ylabel('height z [m]')
    ax[3].plot(time, np.degrees(X[:, 0]), label='roll'); ax[3].plot(time, np.degrees(X[:, 1]), label='pitch')
    ax3b = ax[3].twinx()
    ax3b.plot(time, np.degrees(psi), color='tab:green', label='yaw (right axis)')
    ax3b.set_ylabel('yaw [deg]', color='tab:green')
    ax[3].set_ylabel('roll, pitch [deg]'); ax[3].legend(loc='upper left', fontsize=8)
    ax3b.legend(loc='upper right', fontsize=8)
    for i, n in enumerate(FEET):
        ax[4].step(time[:-1], U[:, 3 * i + 2], where='post', label=n, color=FOOT_COLORS[n])
    ax[4].set_ylabel('f_z [N]'); ax[4].legend(loc='upper left', ncol=4, fontsize=8)
    ax[5].plot(time[:-1], T); ax[5].set_ylabel('solve time [ms]'); ax[5].set_xlabel('time [s]')
    for a in ax:
        _events(a, events); a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    print(f"stored figure in {path}")


def plot_foot_forces(time, U, C, mu, f_min, f_max, events, path, n_poly=None, t_window=None):
    """Detailed per-foot plot: one column per foot (FR, FL, RR, RL), rows:
         1) f_z with its limits f_min / f_max
         2) f_x and f_y
         3) friction use |f_t| / f_z (stance only) against mu (and the n-gon face limit)
       Swing phases are shaded grey. t_window = (t0, t1) zooms into a time range."""
    t = time[:-1]
    fig, ax = plt.subplots(3, 4, figsize=(15, 8), sharex=True)
    for i, n in enumerate(FEET):
        f = U[:, 3 * i:3 * i + 3]
        c = C[:, i]
        for r in range(3):
            _shade_swing(ax[r, i], t, c)
            _events(ax[r, i], events)
            ax[r, i].grid(alpha=0.3)
        # 1) normal force
        ax[0, i].step(t, f[:, 2], where='post', color=FOOT_COLORS[n])
        ax[0, i].axhline(f_min, color='k', ls=':', lw=0.8)
        ax[0, i].axhline(f_max, color='k', ls=':', lw=0.8)
        ax[0, i].set_ylim(-20, max(f_max * 0.5, f[:, 2].max() * 1.15))
        ax[0, i].set_title(n)
        # 2) tangential forces
        ax[1, i].step(t, f[:, 0], where='post', label='f_x')
        ax[1, i].step(t, f[:, 1], where='post', label='f_y')
        # 3) friction use, only meaningful in stance
        ratio = np.where(c == 1, np.hypot(f[:, 0], f[:, 1]) / np.maximum(f[:, 2], 1e-9), np.nan)
        ax[2, i].step(t, ratio, where='post', color='tab:purple')
        ax[2, i].axhline(mu, color='r', ls='--', lw=1, label='μ')
        if n_poly:
            ax[2, i].axhline(mu * np.cos(np.pi / n_poly), color='r', ls=':', lw=1,
                             label=f'μ cos(π/{n_poly})')
        ax[2, i].set_ylim(0, mu * 1.25)
        ax[2, i].set_xlabel('time [s]')
    ax[0, 0].set_ylabel('f_z [N]')
    ax[1, 0].set_ylabel('f_x, f_y [N]'); ax[1, 0].legend(loc='upper left', fontsize=8)
    ax[2, 0].set_ylabel('|f_t| / f_z'); ax[2, 0].legend(loc='upper left', fontsize=8)
    if t_window:
        ax[0, 0].set_xlim(*t_window)
    fig.suptitle('Foot forces (world frame) — grey: swing, dotted lines: f_min / f_max, '
                 'dashed: friction limit')
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    print(f"stored figure in {path}")


def _rot(phi, theta, psi):
    """R = Rz(psi) Ry(theta) Rx(phi), body -> world."""
    cx, sx, cy, sy, cz, sz = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _foot_trajectories(FEET_POS, C, swing_height=0.05):
    """Foot positions for DISPLAY: stance feet stay at their foothold; a swing foot moves from its
    previous to its next foothold along a small arc. (The SRBD model itself has no swing motion.)"""
    n = len(C)
    P = FEET_POS.copy()
    for leg in range(4):
        i = 0
        while i < n:
            if C[i, leg] == 1:
                i += 1
                continue
            j = i
            while j < n and C[j, leg] == 0:
                j += 1                                   # swing phase covers steps i .. j-1
            start = FEET_POS[i - 1, leg] if i > 0 else FEET_POS[i, leg]
            goal = FEET_POS[j, leg] if j < n else start
            for k in range(i, j):
                s = (k - i + 1) / (j - i + 1)            # 0 -> 1 across the swing
                P[k, leg, :2] = (1 - s) * start[:2] + s * goal[:2]
                P[k, leg, 2] = swing_height * np.sin(np.pi * s)
            i = j
    return P


def animate_robot(X, U, C, FEET_POS, hips, dt, mass, path, cmd=None, g=9.81):
    """GIF: side view (x-z) with body, legs and ground reaction forces, and top view (x-y) with the
    hip rectangle, CoM path and stance (filled) / swing (hollow) feet. One frame per control step,
    played in real time. The view follows the robot."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    n = len(U)
    P = _foot_trajectories(FEET_POS, C)
    hips3 = np.column_stack([hips, np.zeros(4)])
    scale = 0.15 / (mass * g / 2)                        # m*g/2 -> 0.15 m arrow
    order = [0, 1, 3, 2, 0]                              # FR, FL, RL, RR, FR (rectangle)

    fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1.3, 1]})
    for ax in (ax_s, ax_t):
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax_s.set_xlabel('x [m]'); ax_s.set_ylabel('z [m]'); ax_s.set_title('side view: legs and ground reaction forces')
    ax_t.set_xlabel('x [m]'); ax_t.set_ylabel('y [m]'); ax_t.set_title('top view: filled = stance, hollow = swing')
    ax_s.axhline(0, color='k', lw=1)

    body_s, = ax_s.plot([], [], 'k-', lw=4, solid_capstyle='round')
    com_s, = ax_s.plot([], [], 'ko', ms=5)
    legs_s = [ax_s.plot([], [], '-', color=FOOT_COLORS[f], lw=2)[0] for f in FEET]
    feet_s = [ax_s.plot([], [], 'o', color=FOOT_COLORS[f], ms=6)[0] for f in FEET]
    arrows = [ax_s.annotate('', xy=(0, 0), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color=FOOT_COLORS[f], lw=2)) for f in FEET]
    body_t, = ax_t.plot([], [], 'k-', lw=1.5)
    path_t, = ax_t.plot([], [], '-', color='0.5', lw=1)
    feet_t = [ax_t.plot([], [], 'o', color=FOOT_COLORS[f], ms=9, mec=FOOT_COLORS[f], label=f)[0] for f in FEET]
    ax_t.legend(loc='upper right', fontsize=8, ncol=4)
    text = ax_s.text(0.02, 0.95, '', transform=ax_s.transAxes, va='top', fontsize=9)

    def frame(k):
        x = X[k]
        R = _rot(*x[0:3])
        hip_w = x[3:6] + hips3 @ R.T                     # hip positions in the world
        cx = x[3]
        # side view
        front, rear = (hip_w[0] + hip_w[1]) / 2, (hip_w[2] + hip_w[3]) / 2
        body_s.set_data([rear[0], front[0]], [rear[2], front[2]])
        com_s.set_data([x[3]], [x[5]])
        for i in range(4):
            legs_s[i].set_data([hip_w[i, 0], P[k, i, 0]], [hip_w[i, 2], P[k, i, 2]])
            feet_s[i].set_data([P[k, i, 0]], [P[k, i, 2]])
            f = U[k, 3 * i:3 * i + 3]
            tip = P[k, i] + scale * f
            arrows[i].set_position((P[k, i, 0], P[k, i, 2]))
            arrows[i].xy = (tip[0], tip[2])
            arrows[i].set_visible(C[k, i] == 1)
        ax_s.set_xlim(cx - 0.65, cx + 0.65); ax_s.set_ylim(-0.05, 0.7)
        # top view
        body_t.set_data(hip_w[order, 0], hip_w[order, 1])
        path_t.set_data(X[:k + 1, 3], X[:k + 1, 4])
        for i in range(4):
            feet_t[i].set_data([P[k, i, 0]], [P[k, i, 1]])
            feet_t[i].set_markerfacecolor(FOOT_COLORS[FEET[i]] if C[k, i] else 'none')
        ax_t.set_xlim(cx - 0.55, cx + 0.55); ax_t.set_ylim(-0.3, 0.3)
        vx_txt = f',  v_x cmd = {cmd[k]:.2f} m/s' if cmd is not None else ''
        text.set_text(f't = {k * dt:.3f} s{vx_txt}\nv_x = {x[9]:.2f} m/s,  pitch = {np.degrees(x[1]):.1f}°')
        return []

    anim = FuncAnimation(fig, frame, frames=n, interval=1000 * dt)
    fig.tight_layout()
    anim.save(path, writer=PillowWriter(fps=round(1 / dt)))
    plt.close(fig)
    print(f"stored animation in {path}")


def plot_solve_time_histogram(T, N, path):
    """Distribution of the solve times, like Fig. 4 of the paper."""
    p99 = np.percentile(T, 99)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(T, bins=np.arange(0, np.ceil(p99) + 0.25, 0.25), color='tab:blue', edgecolor='black')
    if T.max() > np.ceil(p99):
        ax.text(0.98, 0.95, f'{(T > np.ceil(p99)).sum()} solve(s) above {np.ceil(p99):.0f} ms\n'
                f'(max {T.max():.1f} ms)', transform=ax.transAxes, ha='right', va='top', fontsize=9)
    ax.set_xlabel('Solve Time (ms)'); ax.set_ylabel('Frequency')
    ax.set_title(f'Solve Time (SQP_RTI, N = {N}, {len(T)} solves)')
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    print(f"stored figure in {path}")
