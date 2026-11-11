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

## Running

```bash
python3 -m venv ~/.venvs/robodog
~/.venvs/robodog/bin/pip install sympy casadi
cd derivation
~/.venvs/robodog/bin/python step6_matrix_form.py
```

Run the scripts from inside `derivation/`, since later steps import earlier ones.
