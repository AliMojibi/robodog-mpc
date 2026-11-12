# robodog-mpc

Nonlinear MPC for quadruped locomotion based on the Single Rigid Body Dynamics (SRBD) model.

**Reference:** J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, S. Kim,
*Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*, IROS 2018.
[IEEE Xplore](https://ieeexplore.ieee.org/document/8594448)

The paper freezes yaw and the foot lever arms to get a linear (convex) MPC. We keep the same
model but leave those as functions of the state, which makes it nonlinear and suitable for NMPC.

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
| SciPy / Matplotlib | 1.15.3 / 3.10.9 | pendulum example only |
| ROS 2 | Humble | C++ package (planned) |

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
