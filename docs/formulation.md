# Mathematical formulation

Nonlinear model predictive control (NMPC) of a quadruped on the **single rigid body dynamics** (SRBD) model,
following Di Carlo et al., *Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*
(IROS 2018), but solved as a **nonlinear** problem. Where this document refers to an equation number, it is the
paper's. The step-by-step derivation is in `robodog_nmpc/scripts/derivation/` (steps 1–11), the optimal control
problem in `robodog_nmpc/scripts/srbd_ocp.py`, and the runtime part in `robodog_nmpc/src/nmpc_solver.cpp`.

**Notation.** World frame $\mathcal{W}$ with $z$ up, gravity $g = [0, 0, -9.81]^\top\,\mathrm{m/s^2}$. Feet
$i \in \{\mathrm{FR}, \mathrm{FL}, \mathrm{RR}, \mathrm{RL}\}$, always in this order. $R_z(\psi)$ is the rotation
about $z$ by the yaw angle $\psi$.

---

## 1. Model

### State, input, parameters

| | Symbol | Size | Content |
|---|---|---|---|
| state | $x = [\Theta, p, \omega, v]$ | 12 | Euler angles $\Theta = [\phi, \theta, \psi]$ (Z-Y-X, $R = R_z(\psi) R_y(\theta) R_x(\phi)$ maps body to world), CoM position $p$, angular velocity $\omega$, linear velocity $v$; all in $\mathcal{W}$ |
| input | $u = [f_\mathrm{FR}, f_\mathrm{FL}, f_\mathrm{RR}, f_\mathrm{RL}]$ | 12 | ground reaction forces acting on the robot, in $\mathcal{W}$ |
| parameters | $P = [p_{\mathrm{foot},i}, m, I_{xx}, I_{yy}, I_{zz}, \mu]$ | 17 | foot positions (12), mass, principal inertia ${}^B I = \mathrm{diag}(I_{xx}, I_{yy}, I_{zz})$, friction coefficient |

The parameters are set per shooting node at runtime, so the foot positions can change along the horizon and the
robot properties can be changed in `config.yaml` without regenerating the solver.

### Dynamics

The robot is one rigid body; the legs are massless and only transmit the foot forces:

$$
\begin{aligned}
\dot\Theta &= R_z(\psi)^\top\, \omega \\
\dot p &= v \\
\dot\omega &= \hat I(\psi)^{-1} \textstyle\sum_i (p_{\mathrm{foot},i} - p) \times f_i,
\qquad \hat I(\psi) = R_z(\psi)\, {}^B I\, R_z(\psi)^\top \\
\dot v &= \tfrac{1}{m} \textstyle\sum_i f_i + g
\end{aligned}
$$

Two approximations are taken from the paper:

- **Euler rates** (Eqs. 10–12): the exact map $\dot\Theta = E(\Theta)^{-1}\omega$ is replaced by its small roll/pitch
  form $R_z(\psi)^\top \omega$. Valid while $|\phi|, |\theta|$ stay small (locomotion on flat ground).
- **Inertia** (Eqs. 14–15): the world-frame inertia uses only the yaw rotation. The term $\omega \times I\omega$
  is neglected (small angular velocities).

**What makes it nonlinear.** The paper linearises around the reference: it fixes $\psi$ to the reference yaw in
$R_z(\psi)$ and the lever arms $p_{\mathrm{foot},i} - p$ to their reference values, which gives a linear
time-varying model and a convex QP. Here both stay **functions of the state**: the yaw in the attitude kinematics
and the inertia, and the bilinear lever-arm terms $(p_{\mathrm{foot},i} - p) \times f_i$. The NMPC therefore
remains exact for large heading changes within the horizon and for CoM motion relative to the feet.

### Corrections to the paper

Found while deriving the model symbolically (steps 2–5):

- Eq. 9: entry (3,1) of $E(\Theta)$ is $-\sin\theta$ (Eq. 10 is correct).
- Eqs. 12/16: the small-angle map is $R_z(\psi)^\top$, not $R_z(\psi)$.
- Eq. 5 vs. Table I: the sign of gravity is counted twice; here $g = [0, 0, -9.81]^\top$ is **added**.

---

## 2. Optimal control problem

### Discretisation

Multiple shooting with $N$ intervals of length $\Delta t$ (default $N = 16$, $\Delta t = 25\,\mathrm{ms}$, horizon
$0.4\,\mathrm{s}$). On each interval, the input $u_k$ and the parameters $P_k$ are constant, and the state is
propagated with an explicit Runge–Kutta integrator (default RK4, one step per interval):
$x_{k+1} = \Phi(x_k, u_k, P_k)$.

### Problem

$$
\begin{aligned}
\min_{x_0..x_N,\ u_0..u_{N-1}} \quad & \sum_{k=0}^{N-1} \big\| y(x_k, u_k) - y_\mathrm{ref} \big\|^2_{W}
\;+\; \big\| y_N(x_N) - y_{\mathrm{ref},N} \big\|^2_{W_N} \\
\text{s.t.} \quad & x_0 = \hat x \quad \text{(measured state)} \\
& x_{k+1} = \Phi(x_k, u_k, P_k), \qquad k = 0, \dots, N-1 \\
& \text{contact and friction constraints on } u_k, \qquad k = 0, \dots, N-1
\end{aligned}
$$

### Cost

Nonlinear least squares on

$$
y(x, u) = \big[\, R_z(\psi)^\top v \ (3),\ \ \omega_z,\ \ p_z,\ \ \phi,\ \ \theta,\ \ u \ (12) \,\big] \in \mathbb{R}^{19},
\qquad y_N(x) = \text{the first 7 entries}.
$$

| Entry | Reference | Weight (config key, default) |
|---|---|---|
| velocity in the **heading frame** $R_z(\psi)^\top v$ | $[v_x^\mathrm{cmd}, v_y^\mathrm{cmd}, 0]$ | `w_velocity` (1, 1, 1) |
| yaw rate $\omega_z$ | $\dot\psi^\mathrm{cmd}$ | `w_yaw_rate` (1) |
| height $p_z$ | `nominal_height` | `w_height` (50) |
| roll, pitch | 0, 0 | `w_roll_pitch` (1, 1) |
| forces $u$ | 0 | `w_force` ($10^{-6}$, regularisation) |

The weights are diagonal, $W = \mathrm{diag}(w)$, $W_N = \mathrm{diag}(w_{1..7}) \cdot$ `terminal_weight_scale`.
(acados minimises $\tfrac12\|\cdot\|^2_{W}$, so the solver is given $2W$: the cost is exactly
$\sum_j w_j (y_j - y_{\mathrm{ref},j})^2$.) The velocity is tracked in the heading frame, so a command
"forward 0.5 m/s" stays forward while the robot turns; the yaw angle itself is not tracked, only its rate.
The command is constant over the horizon.

### Contact

Per interval $k$ and foot $i$ (contact schedule from the caller, $c_{k,i} = 1$: stance):

$$
c_{k,i} = 0:\ f_{k,i} = 0 \qquad\qquad
c_{k,i} = 1:\ f_\mathrm{min} \le f_{k,i,z} \le f_\mathrm{max},\ \ f_{k,i,x}, f_{k,i,y} \text{ limited by friction}
$$

These are simple bounds on $u_k$ (set at runtime). "No bound" is passed as $10^{10}$ (= `ACADOS_INFTY`); a smaller
value would be treated as a real bound and slows the QP solver down considerably.

### Friction

The friction cone $\sqrt{f_x^2 + f_y^2} \le \mu f_z$ is a second-order cone, not a linear constraint; as a
nonlinear constraint in SQP it converges poorly (step 8, step 10). Three options (`friction.model`):

| Model | Constraint per stance foot | Properties |
|---|---|---|
| `polygon` (**default**) | $\cos\theta_j f_x + \sin\theta_j f_y \le \mu \cos(\pi/n)\, f_z$, $\ \theta_j = 2\pi j / n$, $j = 0..n-1$ | linear in $u$; $\mu$ stays a runtime parameter |
| `cone` | $\sqrt{f_x^2 + f_y^2 + \varepsilon^2} \le \mu f_z$, $\varepsilon = 10^{-3}$ N | exact; slow / poor SQP convergence |
| `pyramid` | $\lvert f_x\rvert \le \mu f_z,\ \lvert f_y\rvert \le \mu f_z$ (as in the paper) | comparison only: $\mu$ compiled in |

**Polygon.** Each row is a half-plane whose boundary lies at distance $\mu\cos(\pi/n) f_z$ from the $f_z$ axis.
The $n$ half-planes form a regular $n$-gon with apothem $\mu\cos(\pi/n) f_z$, so its circumradius is exactly
$\mu f_z$: the corners lie **on** the cone and the polygon is **inscribed**. Hence

- every feasible force satisfies the true friction cone (no slipping by construction), and
- the usable tangential force is reduced by at most $1 - \cos(\pi/n)$ (in the middle of an edge):
  1.9 % for $n = 16$, 7.6 % for $n = 8$.

The paper's pyramid lies **outside** the cone: in its corners the tangential force reaches $\sqrt{2}\,\mu f_z$,
so feet could slip. The constraints are applied at every interval including $k = 0$ (in acados, the first stage
has its own constraint function; without it, $u_0$, the force actually applied, would be unconstrained).

---

## 3. Solution method

- **SQP with real-time iteration (SQP-RTI).** One SQP iteration per control step: linearise around the current
  guess, solve one QP, take the full step. The solution improves from step to step as the problem moves.
- **Gauss–Newton Hessian** $J^\top W J$ of the least-squares cost (constraint curvature neglected; the polygon is
  linear anyway).
- **QP solver:** HPIPM (interior point) with partial condensing; iteration limit `solver.qp_solver_iter_max` (50).
  A QP that reaches the limit is accepted (acados reports success), a QP failure is reported as an error.
- **Warm start:** the previous solution shifted by one interval:
  $x_k \leftarrow x^\mathrm{prev}_{k+1}$, $u_k \leftarrow u^\mathrm{prev}_{\min(k+1, N-1)}$. First call (cold start):
  $x_k = \hat x$, and the stance feet of each interval share the weight, $f_{z} = m\,9.81 / n_\mathrm{stance}$.
- **Failure handling** (C++ class): if the solve fails, the iterate (including the multipliers) is reset, the next
  call warm-starts from the last successful plan, and the forces returned are that plan's next step with the feet
  now in swing set to zero.

The QP of one step has $N \cdot 12$ inputs, $(N+1) \cdot 12$ states, $N \cdot 64$ friction rows ($n = 16$) and
the input bounds; one RTI step takes about 1.3 ms (median) on a desktop CPU.

---

## 4. Inputs per control step

At every call the C++ class fills, for $k = 0..N$:

- $x_0 = \hat x$ (measured state);
- parameters $P_k$: the foot positions for node $k$, plus mass, inertia and $\mu$ from `config.yaml`;
- bounds of $u_k$ from the contact flags of interval $k$;
- $y_\mathrm{ref}$ from the command and `nominal_height`.

The contact schedule and the foot positions can be given for the whole horizon, or reduced forms are used:

| `run()` form | Contact over the horizon | Foot positions over the horizon |
|---|---|---|
| (a) | the 4 current flags, held | current positions, held |
| (b) | schedule $c_{k,i}$ | current positions; predicted footholds for later touchdowns |
| (c) | schedule $c_{k,i}$ | given by the caller for every node |

**Foothold prediction (form b).** A foot keeps its position while it stays on the ground (and in swing, where
its force is zero anyway). If foot $i$ touches down at interval $k_\mathrm{td} > 0$, its position for that stance
phase is the Raibert heuristic (paper Eq. 33) evaluated on the commanded path:

$$
p_{\mathrm{foot},i} = \bar p_{xy}(k_\mathrm{td}) + R_z(\bar\psi(k_\mathrm{td}))\, h_i
+ R_z(\bar\psi(k_\mathrm{td}))\, v^\mathrm{cmd}_{xy} \,\frac{T_\mathrm{stance}}{2},
\qquad z = \text{height of the lowest current foot}
$$

where $\bar p_{xy}, \bar\psi$ follow the commanded velocity and yaw rate from the measured state, $h_i$ is the hip
offset (`hip_offsets_xy`) and $T_\mathrm{stance}$ is counted from the schedule (cut at the end of the horizon).
The foot is placed so that the body passes over it symmetrically during stance.

---

## 5. Verification

- Model: SymPy derivation vs. the paper's equations; CasADi model vs. SymPy at 1000 random points (step 7).
- Closed loop in Python (step 11): stand, trot, forward 0.5 m/s, turn 0.6 rad/s; tracking, friction usage
  $\le \mu$, swing forces 0.
- C++ class: 45 GoogleTests, among them a step-by-step comparison with the acados Python interface on a closed
  loop (bit-identical forces) and a 10 s closed-loop walk (tracking, force limits on every step, timing).
