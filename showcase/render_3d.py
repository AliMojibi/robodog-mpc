"""3D showcase animation of the closed-loop NMPC run (portfolio material, not part of the task).

Reads the results saved by robodog_nmpc/scripts/derivation/step11_closed_loop.py and renders:
  - the robot body as a shaded box, with two-link legs (thigh + shank, knees bent backward)
  - a checkerboard floor, the body's shadow, footprints of past touchdowns, the CoM trail
  - the ground reaction forces of the stance feet as arrows
  - a slowly orbiting camera that follows the robot

Note: the legs are for DISPLAY only. The MPC uses the single-rigid-body model (no legs); the knee
positions come from simple two-link geometry, and swing feet follow a small arc.

Usage (from the repo root):
    ~/.venvs/robodog/bin/python showcase/render_3d.py            # -> showcase/robodog_nmpc.mp4 + .gif
"""
import os
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVATION = os.path.join(HERE, '..', 'robodog_nmpc', 'scripts', 'derivation')
sys.path.insert(0, DERIVATION)
from utils.plots import _foot_trajectories, _rot, FEET          # noqa: E402

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# ---------------- look ----------------
BG = '#0d1117'
TILE_A, TILE_B = '#1c2128', '#262c36'
BODY_TOP, BODY_SIDE, BODY_EDGE = '#5aa9e6', '#3a7cb8', '#c9e4ff'
LEG = '#d0d7de'
FOOT_COLORS = {'FR': '#4cc9f0', 'FL': '#f72585', 'RR': '#7bd389', 'RL': '#ffb703'}
FORCE = '#ff7b00'
TEXT = '#e6edf3'

# ---------------- geometry (display only) ----------------
BODY_L, BODY_W, BODY_H = 0.72, 0.26, 0.11       # body box [m]
L_THIGH, L_SHANK = 0.30, 0.30                    # leg links [m]
FORCE_SCALE = 0.18 / 211.0                       # m*g/2 (~211 N) -> 0.18 m arrow


def box_faces(center, R, l, w, h):
    """6 faces of a box (center, rotation R, size l x w x h) as lists of 3D corners."""
    sx, sy, sz = l / 2, w / 2, h / 2
    c = np.array([[sx, sy, sz], [sx, -sy, sz], [-sx, -sy, sz], [-sx, sy, sz],
                  [sx, sy, -sz], [sx, -sy, -sz], [-sx, -sy, -sz], [-sx, sy, -sz]])
    v = center + c @ R.T
    idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]
    return [v[i] for i in idx], v


def knee(hip, foot, forward):
    """Two-link leg: knee position, bent backward (as on the MIT Cheetah), in the plane of the leg
    and the body's forward axis."""
    d = foot - hip
    L = min(np.linalg.norm(d), L_THIGH + L_SHANK - 1e-6)
    u = d / max(np.linalg.norm(d), 1e-9)
    n = forward - (forward @ u) * u                   # forward axis, orthogonal to the leg
    n /= max(np.linalg.norm(n), 1e-9)
    a = (L_THIGH ** 2 - L_SHANK ** 2 + L ** 2) / (2 * L)
    h = np.sqrt(max(L_THIGH ** 2 - a ** 2, 0.0))
    return hip + a * u - h * n                       # minus: knee points backward


def floor_tiles(x0, x1, y0, y1, size=0.25):
    """Checkerboard tiles inside [x0, x1] x [y0, y1], aligned to a fixed world grid
    (so the floor scrolls under the robot) and clipped to the window (mplot3d does not clip)."""
    tiles, colors = [], []
    for i in range(int(np.floor(x0 / size)), int(np.ceil(x1 / size))):
        for j in range(int(np.floor(y0 / size)), int(np.ceil(y1 / size))):
            xa, xb = max(i * size, x0), min((i + 1) * size, x1)
            ya, yb = max(j * size, y0), min((j + 1) * size, y1)
            tiles.append([[xa, ya, 0], [xb, ya, 0], [xb, yb, 0], [xa, yb, 0]])
            colors.append(TILE_A if (i + j) % 2 == 0 else TILE_B)
    return tiles, colors


def render(npz_path, out_mp4, out_gif):
    d = np.load(npz_path)
    X, U, C, FEET_POS, dt = d['X'], d['U'], d['C'], d['FEET_POS'], float(d['dt'])
    hips = d['hips']
    cmd = d['cmd'] if 'cmd' in d else np.column_stack([d['vx_cmd'], 0 * d['vx_cmd'], 0 * d['vx_cmd']])
    n = len(U)
    P = _foot_trajectories(FEET_POS, C)               # display foot positions incl. swing arcs
    hips3 = np.column_stack([hips, np.zeros(4)])

    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d', facecolor=BG)

    def frame(k):
        ax.cla()
        ax.set_axis_off()
        # mplot3d sorts whole objects by average depth, which puts the big floor in front of the
        # robot. Use a fixed drawing order instead (zorder): floor < shadows < far legs < body < near legs
        ax.computed_zorder = False
        x = X[k]
        R = _rot(*x[0:3])
        com = x[3:6]
        forward = R[:, 0]

        # camera: follow the robot, orbit slowly
        span = 0.95
        ax.set_xlim(com[0] - span, com[0] + span)
        ax.set_ylim(com[1] - span, com[1] + span)
        ax.set_zlim(0, 0.95)
        ax.set_box_aspect((1, 1, 0.56), zoom=1.55)  # zoom in: robot fills more of the frame
        azim = -62 + 40 * k / n
        ax.view_init(elev=18, azim=azim)
        cam = np.array([np.cos(np.radians(azim)), np.sin(np.radians(azim))])   # camera side (x-y)

        # floor (a window around the robot; mplot3d does not clip, so it is cut to this window)
        fl = 2.4 * span                                   # floor larger than the view: fills the frame
        tiles, tile_colors = floor_tiles(com[0] - fl, com[0] + fl, com[1] - fl, com[1] + fl)
        ax.add_collection3d(Poly3DCollection(tiles, facecolors=tile_colors, edgecolors='none', zorder=0,
                                             clip_on=False))   # extend past the square 3D axes area

        # CoM trail on the floor and footprints of past touchdowns
        ax.plot(X[:k + 1, 3], X[:k + 1, 4], 0.002, color='#58a6ff', lw=1.2, alpha=0.6, zorder=1)
        for i, f in enumerate(FEET):
            past = np.where(C[:k + 1, i] == 1)[0]
            if len(past):
                pts = np.unique(np.round(FEET_POS[past, i, :2], 4), axis=0)
                inside = (np.abs(pts[:, 0] - com[0]) < span) & (np.abs(pts[:, 1] - com[1]) < span)
                ax.scatter(pts[inside, 0], pts[inside, 1], 0.002, s=18, color=FOOT_COLORS[f],
                           alpha=0.3, depthshade=False, zorder=1)

        # body shadow (box projected straight down)
        _, corners = box_faces(com, R, BODY_L, BODY_W, BODY_H)
        sh = corners[[0, 1, 2, 3], :2]
        ax.add_collection3d(Poly3DCollection([np.column_stack([sh, np.full(4, 0.003)])],
                                             facecolors='black', alpha=0.35, edgecolors='none', zorder=1))

        # legs
        hip_w = com + hips3 @ R.T
        for i, f in enumerate(FEET):
            foot = P[k, i]
            kn = knee(hip_w[i], foot, forward)
            near = (hip_w[i, :2] - com[:2]) @ cam > 0      # leg on the camera side of the body
            z = 5 if near else 3
            ax.plot(*zip(hip_w[i], kn), color=LEG, lw=5, solid_capstyle='round', zorder=z)
            ax.plot(*zip(kn, foot), color=LEG, lw=4, solid_capstyle='round', zorder=z)
            ax.scatter(*kn, s=30, color='#8b949e', depthshade=False, zorder=z)
            stance = C[k, i] == 1
            ax.scatter(*foot, s=90 if stance else 45, color=FOOT_COLORS[f],
                       edgecolors='white' if stance else 'none', linewidths=1.2, depthshade=False,
                       zorder=z)
            # foot shadow while in the air
            if not stance:
                ax.scatter(foot[0], foot[1], 0.002, s=25, color='black', alpha=0.4, depthshade=False,
                           zorder=1)
            # ground reaction force
            if stance:
                fv = U[k, 3 * i:3 * i + 3] * FORCE_SCALE
                ax.quiver(*foot, *fv, color=FORCE, lw=2.5, arrow_length_ratio=0.18, zorder=z + 1)

        # body (drawn last so it sits on top of the hips)
        faces, _ = box_faces(com, R, BODY_L, BODY_W, BODY_H)
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors=[BODY_TOP, BODY_SIDE, BODY_SIDE, BODY_SIDE, BODY_SIDE, BODY_SIDE],
            edgecolors=BODY_EDGE, linewidths=0.6, alpha=0.95, zorder=4))
        ax.scatter(*hip_w.T, s=40, color='#c9d1d9', depthshade=False, zorder=4)

        # overlay text
        phase = 'STAND' if C[k].sum() == 4 else 'TROT'
        fig.texts.clear()
        fig.text(0.03, 0.93, 'Nonlinear MPC for quadruped locomotion', color=TEXT, fontsize=18,
                 weight='bold')
        fig.text(0.03, 0.885, 'single-rigid-body model  ·  acados SQP-RTI  ·  16-gon friction cone',
                 color='#8b949e', fontsize=11)
        v_fwd = np.cos(x[2]) * x[9] + np.sin(x[2]) * x[10]          # forward speed (heading frame)
        fig.text(0.03, 0.08, f't = {k * dt:4.2f} s     gait: {phase}\n'
                             f'forward speed {v_fwd:4.2f} m/s  (cmd {cmd[k, 0]:.2f})\n'
                             f'yaw rate      {x[8]:4.2f} rad/s (cmd {cmd[k, 2]:.2f})',
                 color=TEXT, fontsize=12, family='monospace')
        fig.text(0.97, 0.08, 'orange arrows: ground reaction forces', color=FORCE, fontsize=10,
                 ha='right')
        return []

    anim = FuncAnimation(fig, frame, frames=n, interval=1000 * dt)
    fps = round(1 / dt)
    anim.save(out_mp4, writer=FFMpegWriter(fps=fps, bitrate=4000,
                                           extra_args=['-pix_fmt', 'yuv420p']), dpi=100)
    print(f"stored {out_mp4}")
    # small GIF for the README, converted from the MP4 by ffmpeg with an optimized color palette:
    # 10 fps, 560 px wide, 64 colors, no dithering: ~1.7 MB (dithering adds visible grain on the dark
    # background and makes the file larger)
    vf = ('fps=10,scale=560:-1:flags=lanczos,split[a][b];'
          '[a]palettegen=max_colors=64:stats_mode=diff[p];'
          '[b][p]paletteuse=dither=none:diff_mode=rectangle')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-y', '-i', out_mp4,
                    '-vf', vf, out_gif], check=True)
    print(f"stored {out_gif}")
    plt.close(fig)


if __name__ == '__main__':
    render(os.path.join(DERIVATION, 'step11_results.npz'),
           os.path.join(HERE, 'robodog_nmpc.mp4'),
           os.path.join(HERE, 'robodog_nmpc.gif'))
