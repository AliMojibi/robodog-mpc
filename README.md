# robodog-mpc

Nonlinear MPC for quadruped locomotion based on the Single Rigid Body Dynamics (SRBD) model.

**Reference:** J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, S. Kim,
*Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*, IROS 2018.
[IEEE Xplore](https://ieeexplore.ieee.org/document/8594448)

The paper freezes yaw and the foot lever arms to get a linear (convex) MPC. We keep the same
model but leave those as functions of the state, which makes it nonlinear and suitable for NMPC.

![Closed-loop NMPC: stand, trot, forward, turn](showcase/robodog_nmpc.gif)

*Closed-loop simulation (step 11): stand → trot in place → trot forward at 0.5 m/s → turn at 0.6 rad/s.
Orange arrows are the ground reaction forces computed by the NMPC. The legs are for display only;
the controller uses the single-rigid-body model.*

## Derivation (`derivation/`)

Step-by-step SymPy scripts following Section III of the paper:

| Step | Script | Paper | Content |
|---|---|---|---|
| 1 | `step1_rotations.py` | Eq. 8 | `R = Rz(ψ) Ry(θ) Rx(φ)` |
| 2 | `step2_angular_velocity.py` | Eq. 9 | `ω = E(Θ) Θ̇` from `Ṙ Rᵀ = [ω]×` |
| 3 | `step3_euler_rates.py` | Eqs. 10–12 | `E⁻¹`, small-angle `Θ̇ ≈ Rz(ψ)ᵀ ω` |
| 4 | `step4_inertia.py`, `step4b_check_inertia.py` | Eqs. 14–15 | World-frame inertia `Î = Rz ᴮI Rzᵀ` |
| 5 | `step5_equations_of_motion.py` | Eqs. 5–7, 13 | Full 12-state model `ẋ = f(x, u)` |
| 6 | `step6_matrix_form.py` | Eqs. 16–17 | Same model in matrix form, standing check |
| 7 | `step7_casadi_model.py` | — | Model in CasADi (`f_dyn(x, u, P)`), verified against SymPy |
| 8 | `step8_constraints.py` | Eqs. 20–24 | Contact bounds, friction models: 16-gon (default), exact cone, pyramid |
| 9 | `step9_cost.py` | Eq. 18 | Least-squares cost: heading-frame velocity, yaw rate, height, roll/pitch, forces |
| 10 | `step10_acados_ocp.py` | Sec. IV | Full NMPC as an acados OCP; standing, trot and hard-acceleration tests |
| 11 | `step11_closed_loop.py` | Sec. V | Closed loop with SQP-RTI: stand, trot, forward, turn; plots, solve-time histogram (cf. Fig. 4) |

**Friction model.** The friction cone `√(f_x² + f_y²) ≤ μ f_z` is enforced through one of three options
(`build_solver(friction=...)` in step 10):

| Option | Constraint | Notes |
|---|---|---|
| `polygon` (**default**) | inscribed 16-gon of the cone, `cos θⱼ f_x + sin θⱼ f_y − μ cos(π/16) f_z ≤ 0` | linear in `u` → fast, robust SQP; never exceeds μ; at most 1.9 % conservative; μ is a runtime parameter |
| `cone` | exact cone, `√(f_x² + f_y² + ε²) − μ f_z ≤ 0` | available; converges slowly (26 vs. 6 SQP iterations on a 0.5 m/s trot) and not at all when friction is the limiting factor |
| `pyramid` | paper's `\|f_x\|, \|f_y\| ≤ μ f_z` | comparison only: lies outside the cone and allows up to 1.41 μ (feet would slip) |

**Model** (state `x = [Θ, p, ω, v]`, input `u = [f_FR, f_FL, f_RR, f_RL]`, world frame):

```
Θ̇ = Rz(ψ)ᵀ ω
ṗ = v
ω̇ = Î⁻¹(ψ) Σᵢ (p_foot,i − p) × fᵢ
v̇ = Σᵢ fᵢ / m + g,   g = [0, 0, −9.81]ᵀ
```

**Corrections to the paper found along the way**
- Eq. 9: entry (3,1) is `−sin θ`, not 0 (Eq. 10 is correct).
- Eqs. 12/16: the small-angle map is `Rz(ψ)ᵀ`, not `Rz(ψ)`.
- Eq. 5 vs. Table I: the gravity sign is counted twice; we use `g = [0, 0, −9.81]ᵀ`, added.

## Dependencies

Tested versions:

| Component | Version | Notes |
|---|---|---|
| Ubuntu | 22.04 (WSL2) | |
| Python | 3.10.12 | |
| SymPy | 1.14.0 | derivation, steps 1–6 |
| NumPy | 2.2.6 | |
| CasADi | **3.7.2** | pinned to 3.7.x: the range acados v0.6.0 is tested with (3.8 gives a warning) |
| acados | **v0.6.0** | built from source with qpOASES; HPIPM is the QP solver used |
| SciPy / Matplotlib | 1.15.3 / 3.10.9 | plots, pendulum example |
| imageio-ffmpeg | 0.8.1 | bundled ffmpeg for the showcase video |
| ROS 2 | Humble | C++ package (see Roadmap) |

## Running

```bash
python3 -m venv ~/.venvs/robodog
~/.venvs/robodog/bin/pip install sympy numpy "casadi==3.7.*"
cd derivation
~/.venvs/robodog/bin/python step6_matrix_form.py
```

Run the scripts from inside `derivation/`, since later steps import earlier ones.

Step 10 also needs [acados](https://github.com/acados/acados) v0.6.0 and its Python interface:

```bash
# build acados (once)
git clone --branch v0.6.0 --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/acados/acados.git ~/acados
mkdir -p ~/acados/build && cd ~/acados/build
cmake -DACADOS_WITH_QPOASES=ON .. && make install -j$(nproc)
~/.venvs/robodog/bin/pip install -e ~/acados/interfaces/acados_template

# every terminal (or add to ~/.bashrc)
export ACADOS_SOURCE_DIR=$HOME/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/acados/lib
~/.venvs/robodog/bin/python step10_acados_ocp.py
```

## Showcase animation

`showcase/render_3d.py` renders the closed-loop run of step 11 in 3D (MP4 1280×720 + a small GIF):

```bash
cd derivation && ~/.venvs/robodog/bin/python step11_closed_loop.py   # writes step11_results.npz
cd .. && ~/.venvs/robodog/bin/python showcase/render_3d.py            # writes showcase/robodog_nmpc.{mp4,gif}
```

## Roadmap

### Done
- SRBD model derived step by step (SymPy), verified against the paper, three errata documented
- Nonlinear MPC in CasADi + acados: SQP-RTI, contact schedule over the horizon, 16-gon friction cone
- Closed-loop validation in simulation: velocity and yaw-rate tracking, median solve time ≈ 1.5 ms

### Next: C++ and ROS 2 package
- **Code generation script** that builds the acados solver from a single `config.yaml`
  (mass, inertia, μ, cost weights, horizon N and dt, solver options), and documents which settings
  are fixed at code generation (N, friction model) and which can change at runtime
- **ROS 2 (Humble) C++ package** with a standalone solver class wrapping the generated acados C code:
  - `initialize()`: load `config.yaml`, create the solver
  - `run()`: inputs are the measured state, the commanded `v_x`, `v_y`, yaw rate, the contact schedule
    over the horizon (N × 4) and the foot positions; one SQP-RTI iteration with warm start
  - `getSolution()`: the optimal 3D ground reaction force of each foot (plus the predicted trajectory)
- CMake integration of acados and the generated code; build with `colcon`
- Unit tests: the C++ solver reproduces the Python results (standing, trot, turn)
- Written formulation document (model, OCP, friction approximation)

### Then: towards real hardware
- **Whole-body integration**: map the forces to joint torques `τ = Jᵀ Rᵀ f` (paper Eq. 4), with a
  swing-leg controller (Eqs. 1–2), a gait scheduler and a foothold planner (Raibert heuristic, Eq. 33)
- **State estimation** (IMU + leg kinematics) as the source of the measured state
- **Real-time execution**: dedicated control thread, timing budget per MPC call, handling of solver
  failures (fall back to the previous plan), latency compensation
- **Robustness tests** before hardware: model mismatch (leg mass, inertia errors), external pushes,
  uneven terrain, early and late touchdowns, sensor noise, in a multi-body simulator (e.g. MuJoCo or Gazebo)
- **Hardware experiments**: standing, trotting, turning; tune μ with a safety margin and the cost weights

### Further improvements
- Exact friction cone with fast convergence (exact Hessian or a better initial guess for SQP)
- QP solver settings (iteration limit, warm start) to remove the few slow solves at gait/command changes
- Orientation without Euler angles (rotation matrix / quaternion) for aggressive maneuvers
- Foothold optimization inside the MPC and other gaits (pace, bound, gallop)
