"""Step 7: the step-5 SRBD model written in CasADi (the form acados needs).

x = [Theta(3), p(3), omega(3), v(3)]                 12 states
u = [f_FR(3), f_FL(3), f_RR(3), f_RL(3)]             12 inputs (world-frame GRFs)
P = [p_foot_FR(3), .._FL, .._RR, .._RL, m, Ixx, Iyy, Izz]   16 parameters
"""
import casadi as ca

G0 = 9.81


def Rz(a):
    c, s = ca.cos(a), ca.sin(a)
    return ca.vertcat(ca.horzcat(c, -s, 0),
                      ca.horzcat(s,  c, 0),
                      ca.horzcat(0,  0, 1))


# ---------------- symbols (vectors) ----------------
x = ca.SX.sym('x', 12)
u = ca.SX.sym('u', 12)
P = ca.SX.sym('P', 16)

Theta, p, omega, v = x[0:3], x[3:6], x[6:9], x[9:12]
psi = Theta[2]
f = [u[3 * i:3 * i + 3] for i in range(4)]
p_foot = [P[3 * i:3 * i + 3] for i in range(4)]
m = P[12]
I_body_inv = ca.diag(1 / P[13:16])                       # diag(1/Ixx, 1/Iyy, 1/Izz)

# ---------------- dynamics (same as step 5) ----------------
F_total = ca.SX.zeros(3, 1)
tau_total = ca.SX.zeros(3, 1)
for i in range(4):
    F_total += f[i]
    tau_total += ca.cross(p_foot[i] - p, f[i])           # (p_foot - p) x f

I_hat_inv = Rz(psi) @ I_body_inv @ Rz(psi).T

Theta_dot = Rz(psi).T @ omega
p_dot = v
omega_dot = I_hat_inv @ tau_total
v_dot = F_total / m + ca.vertcat(0, 0, -G0)

x_dot = ca.vertcat(Theta_dot, p_dot, omega_dot, v_dot)

# A CasADi Function: callable with numbers, and later handed to acados
f_dyn = ca.Function('f_dyn', [x, u, P], [x_dot], ['x', 'u', 'P'], ['x_dot'])

if __name__ == '__main__':
    print(f_dyn)

    # --- check 1: standing still -> x_dot = 0 ---
    hx, hy, h, mass = 0.30, 0.15, 0.45, 43.0
    feet = [hx, -hy, 0,  hx, hy, 0,  -hx, -hy, 0,  -hx, hy, 0]   # FR, FL, RR, RL
    P_num = feet + [mass, 0.41, 2.1, 2.1]
    x_stand = [0, 0, 0,  0, 0, h,  0, 0, 0,  0, 0, 0]
    u_stand = [0, 0, mass * G0 / 4] * 4
    print("\nx_dot at standing:", f_dyn(x_stand, u_stand, P_num).full().ravel().round(12))

    # --- check 2: compare with the SymPy model of step 5 at 100 random points ---
    import numpy as np
    import sympy as sp
    import step5_equations_of_motion as s5
    import step4_inertia as s4

    # SymPy model as a fast numeric function, with arguments in the same order as f_dyn
    P_syms = [c for pf in s5.p_foot for c in pf] + [s5.m, s4.Ixx, s4.Iyy, s4.Izz]
    f_sympy = sp.lambdify((list(s5.x), list(s5.u), P_syms),
                          s5.x_dot.subs(s5.g0, G0), 'numpy')

    rng = np.random.default_rng(1)
    n_points = 1000
    errors = []
    for _ in range(n_points):
        xr = rng.uniform(-0.5, 0.5, 12)                                   # states
        ur = rng.uniform(-50, 150, 12)                                    # forces [N]
        Pr = np.concatenate([rng.uniform(-0.4, 0.4, 12),                  # foot positions [m]
                             rng.uniform([20, 0.1, 0.5, 0.5], [60, 1.0, 3.0, 3.0])])  # m, I
        diff = f_dyn(xr, ur, Pr).full().ravel() - np.ravel(f_sympy(xr, ur, Pr))
        errors.append(np.abs(diff).max())                                 # worst entry at this point

    errors = np.array(errors)
    print(f"\n|CasADi - SymPy| over {n_points} random points (largest entry of x_dot per point):")
    print(f"  mean: {errors.mean():.3e}")
    print(f"  max : {errors.max():.3e}")
