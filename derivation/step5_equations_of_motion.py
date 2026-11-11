"""Step 5: the complete SRBD equations of motion  x_dot = f(x, u).

State  x = [Theta, p, omega, v]        (12)
Input  u = [f_FR, f_FL, f_RR, f_RL]    (12)  world-frame ground reaction forces

Theta_dot = Rz(psi)^T * omega                               (step 3)
p_dot     = v
omega_dot = I_hat^-1(psi) * sum_i (p_foot_i - p) x f_i      (Eq. 6 + 13, step 4)
v_dot     = sum_i f_i / m + g,   g = [0, 0, -g0]            (Eq. 5, sign fixed)
"""
import sympy as sp
from step1_rotations import phi, theta, psi, Rz
from step4_inertia import Ixx, Iyy, Izz, I_hat_inv

# ---------------- state ----------------
Theta = sp.Matrix([phi, theta, psi])
p = sp.Matrix(sp.symbols('p_x p_y p_z', real=True))
omega = sp.Matrix(sp.symbols('omega_x omega_y omega_z', real=True))
v = sp.Matrix(sp.symbols('v_x v_y v_z', real=True))
x = sp.Matrix.vstack(Theta, p, omega, v)

# ---------------- input: one 3D force per foot ----------------
feet = ['FR', 'FL', 'RR', 'RL']
f = [sp.Matrix(sp.symbols(f'f_{n}_x f_{n}_y f_{n}_z', real=True)) for n in feet]
u = sp.Matrix.vstack(*f)

# ---------------- parameters ----------------
p_foot = [sp.Matrix(sp.symbols(f'pf_{n}_x pf_{n}_y pf_{n}_z', real=True)) for n in feet]
m, g0 = sp.symbols('m g_0', positive=True)
g = sp.Matrix([0, 0, -g0])

# ---------------- dynamics ----------------
r = [p_foot[i] - p for i in range(4)]                                    # lever arms
F_total = sum(f, sp.zeros(3, 1))                                         # net force
tau_total = sum((r[i].cross(f[i]) for i in range(4)), sp.zeros(3, 1))    # net torque about CoM

Theta_dot = Rz(psi).T * omega
p_dot = v
omega_dot = I_hat_inv * tau_total
v_dot = F_total / m + g

x_dot = sp.Matrix.vstack(Theta_dot, p_dot, omega_dot, v_dot)

if __name__ == '__main__':
    print("state x  :", list(x))
    print("input u  :", list(u))
    print("size: nx =", x.shape[0], ", nu =", u.shape[0], ", x_dot has", x_dot.shape[0], "rows")

    print("\n--- Theta_dot ---")
    sp.pprint(Theta_dot)

    print("\n--- p_dot ---")
    sp.pprint(p_dot)

    print("\n--- v_dot ---")
    sp.pprint(v_dot)

    print("\n--- omega_dot = I_hat^-1 * tau,  torque of ONE foot (FR) = (p_foot - p) x f ---")
    sp.pprint(r[0].cross(f[0]))

    n_ops = sp.count_ops(omega_dot)
    print(f"\n(omega_dot fully expanded has {n_ops} operations, too long to print usefully)")
