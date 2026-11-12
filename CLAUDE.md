# CLAUDE.md

## Project
Nonlinear MPC (NMPC) for quadruped locomotion on the Single Rigid Body Dynamics (SRBD) model.
Based on Di Carlo et al. 2018, *Dynamic Locomotion in the MIT Cheetah 3 Through Convex MPC*
(https://ieeexplore.ieee.org/document/8594448), but solved as **nonlinear** MPC: yaw ψ and the
lever arms `r_i = p_foot,i − p` stay functions of the state instead of being frozen to the reference.

**Final deliverable (task spec, `references/taskMpc.pdf`):**
- ROS 2 C++ package with a standalone solver class: `initialize()`, `run()`, `getSolution()`; no pubs/subs.
- Inputs: desired `v_x`, `v_y`, yaw rate, contact state per foot. Output: 3D GRF per foot.
- Constraints: friction cone, zero force for swing feet.
- acados recommended: Python (CasADi) for model and code generation, C++ for the runtime solver.
- Everything configurable via `config.yaml` (mass, inertia, horizon, dt, μ, weights, solver settings).
- Also: math formulation, codegen scripts, README.

## Working style (important)
- Work in **small, explained steps**: one short script per step, run it, explain the output, then wait for the user.
  Never drop a large file at once.
- The user reviews and edits the scripts themselves. Treat their edits as intentional; point out problems, don't silently revert.
- Commit and push only when asked. Commit as `Ali Mojibi <alimojibi99@gmail.com>` (already set in repo config).

## Layout
- `derivation/stepN_*.py` – step-by-step derivation, each step imports earlier ones. **Run from inside `derivation/`.**
  - 1–6: SymPy (rotations, Euler kinematics, inertia, equations of motion, Eq. 16/17 matrix form)
  - 7: CasADi model `f_dyn(x, u, P)`; 8: force constraints; 9: least-squares cost
  - 10: full NMPC as an acados OCP (`build_solver`, `solve`), standing / trot / hard-acceleration tests
- `examples/pendulum/` – acados closed-loop pendulum example (commented), for learning acados.
- Generated acados code (`c_generated_code*/`) and plots are gitignored.
- `references/` – paper and task PDFs, **local only** (gitignored, licensed).
- `notebooks/srbd_derivation.ipynb` – old all-in-one draft, gitignored, not used.

## Conventions
- State `x = [Θ(φ,θ,ψ), p, ω, v]` (12), all world frame; Z-Y-X Euler, `R = Rz(ψ) Ry(θ) Rx(φ)` maps body → world.
- Input `u = [f_FR, f_FL, f_RR, f_RL]` (12), world-frame GRFs. Foot order is always **FR, FL, RR, RL**.
- Parameters `P = [p_foot (12), m, Ixx, Iyy, Izz]` (16); the acados solver uses `p = [P, μ]` (17).
  Gravity fixed: `g = [0, 0, −9.81]`, **added**.
- Friction: **default is the inscribed 16-gon** of the cone (linear in u, μ as parameter, ≤ 1.9 % conservative).
  The exact cone is available (`friction='cone'`) but converges poorly; the paper's pyramid is kept only for
  comparison (lies outside the cone). Decision: keep it this way; revisit the exact cone only if asked.
- Contact: per stage, swing feet get bounds `f = 0` on all three components; stance `f_min ≤ f_z ≤ f_max`.
- acados: nonlinear constraints need `con_h_expr_0` too, otherwise node 0 (u_0, the applied force) is unconstrained.
- Solver defaults in step 10: N = 16, tf = 0.4 s, ERK, GAUSS_NEWTON, HPIPM, `FIXED_STEP`
  (MERIT_BACKTRACKING measured slower in every case).
- Model: `Θ̇ = Rz(ψ)ᵀ ω`, `ṗ = v`, `ω̇ = Rz ᴮI⁻¹ Rzᵀ Σ (p_foot,i − p) × f_i`, `v̇ = Σ f_i / m + g`.
- Paper errata already found: Eq. 9 entry (3,1) is `−sin θ`; Eqs. 12/16 should use `Rz(ψ)ᵀ`; gravity sign double-counted in Eq. 5 vs Table I.
- Default numbers: Cheetah 3 Table I (m = 43, I = diag(0.41, 2.1, 2.1), μ = 0.6, f_min = 10, f_max = 666; weights v 1, yaw rate 1, z 50, Θ 1, force 1e-6).

## Environment (WSL quirks)
- Python env: `~/.venvs/robodog` (sympy, numpy, **casadi pinned to 3.7.x** — tested range of acados v0.6.0, scipy, matplotlib, acados_template).
  Run scripts with `~/.venvs/robodog/bin/python`.
- acados v0.6.0 in `~/acados`. Needs `ACADOS_SOURCE_DIR=$HOME/acados` and `LD_LIBRARY_PATH=$HOME/acados/lib`;
  Claude may not edit `~/.bashrc`, so pass them on the command line.
  The venv cannot live in the project: `/mnt/d` is mounted with `root` as owner, so `chmod` fails there.
- Git data lives in `~/repos/robodog.git` (project `.git` is a pointer file); the project dir is in git's `safe.directory`.
- Remote: https://github.com/AliMojibi/robodog-mpc (private, HTTPS, credentials stored).
- DNS was fixed by setting `/etc/resolv.conf` to 1.1.1.1 / 8.8.8.8; if pip fails with "name resolution", check that first.
  `/etc/wsl.conf` must contain `[network] generateResolvConf = false`, otherwise WSL rewrites resolv.conf on restart.
- `sudo` is not available to Claude; ask the user to run such commands.

## Interface decisions
- Foot positions: provided by the caller, per horizon node (in `p`).
- Contact: a schedule over the horizon (N × 4 flags), not just the current state.

## Next steps
1. Step 11: closed loop in Python (acados simulator as plant, `SQP_RTI`, warm start, moving contact schedule/footholds).
2. config.yaml for all parameters (mass, inertia, μ, weights, N, dt, solver options).
3. Generate C code, wrap in the ROS 2 C++ class (`initialize/run/getSolution`).
4. Formulation document and full README.
