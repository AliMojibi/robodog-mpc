"""Step 6: write the step-5 model in the paper's matrix form (Eqs. 16-17).

6a  Standing still: symmetric feet, each pushing m*g/4  ->  x_dot must be 0.
6b  Eq. 16:  x_dot = A_c(psi) x + B_c(r_1..r_4, psi) u + g      (12 states, gravity as constant)
6c  Eq. 17:  x~_dot = A~ x~ + B~ u                              (13 states, gravity as extra state)

Same equations as step 5, only regrouped. Corrections vs. the paper: Rz(psi)^T instead of Rz(psi),
and g = [0, 0, -g0] added (see step 3 and step 5).
"""
import sympy as sp
from step1_rotations import phi, theta, psi, Rz
from step4_inertia import Ixx, Iyy, Izz, I_hat_inv
from step5_equations_of_motion import x, u, x_dot, p, omega, v, f, p_foot, m, g0, r


def skew(a):
    return sp.Matrix([[ 0   , -a[2],  a[1]],
                      [ a[2],  0   , -a[0]],
                      [-a[1],  a[0],  0   ]])


Z3, I3 = sp.zeros(3, 3), sp.eye(3)

# ---------------- 6b: Eq. 16 ----------------
#            Theta  p    omega       v
A_c = sp.BlockMatrix([[Z3, Z3, Rz(psi).T, Z3],      # Theta_dot = Rz^T omega
                      [Z3, Z3, Z3,        I3],      # p_dot     = v
                      [Z3, Z3, Z3,        Z3],      # omega_dot = (only from forces)
                      [Z3, Z3, Z3,        Z3]]).as_explicit()  # v_dot = (forces + gravity)

# one 12x3 column block per foot: [0; 0; I_hat^-1 [r_i]x; 1/m]
B_c = sp.Matrix.hstack(*[sp.Matrix.vstack(Z3, Z3, I_hat_inv * skew(r[i]), I3 / m)
                         for i in range(4)])

g_term = sp.Matrix([0] * 11 + [-g0])

# ---------------- 6c: Eq. 17, gravity as a 13th state ----------------
x_aug = sp.Matrix.vstack(x, sp.Matrix([g0]))                 # x~ = [x; g0], with g0_dot = 0
A_aug = sp.Matrix.vstack(sp.Matrix.hstack(A_c, g_term / g0),  # extra column feeds g0 into v_z_dot
                         sp.zeros(1, 13))                     # g0_dot = 0
B_aug = sp.Matrix.vstack(B_c, sp.zeros(1, 12))
x_aug_dot = sp.Matrix.vstack(x_dot, sp.Matrix([0]))

# ---------------- numbers: Cheetah 3 (Table I), standing pose ----------------
hx, hy, h = 0.30, 0.15, 0.45
foot_xy = [(hx, -hy), (hx, hy), (-hx, -hy), (-hx, hy)]          # FR, FL, RR, RL
stand = {m: 43, g0: 9.81, Ixx: 0.41, Iyy: 2.1, Izz: 2.1,
         phi: 0, theta: 0, psi: 0, p[0]: 0, p[1]: 0, p[2]: h}
stand.update({s: 0 for s in omega})
stand.update({s: 0 for s in v})
for i, (fx, fy) in enumerate(foot_xy):
    stand.update({p_foot[i][0]: fx, p_foot[i][1]: fy, p_foot[i][2]: 0})
    stand.update({f[i][0]: 0, f[i][1]: 0, f[i][2]: 43 * 9.81 / 4})

if __name__ == '__main__':
    print("================ 6a: standing check ================")
    print("x_dot at standing =", [round(float(e), 12) for e in x_dot.subs(stand)])

    print("\n================ 6b: Eq. 16 ================")
    print("x_dot == A_c x + B_c u + g :",
          sp.simplify(x_dot - (A_c * x + B_c * u + g_term)) == sp.zeros(12, 1))
    print("\nA_c (12x12) =")
    sp.pprint(A_c)
    print("\nB_c, column block of foot FR (12x3) = [0; 0; I_hat^-1 [r_FR]x; 1/m] with r_FR = p_foot_FR - p")
    # sp.pprint(B_c)
    
    states = set(x)
    print("\nAll 12 states, in order (x marks where a state appears inside the matrix):")
    print(f"  {'#':>2}  {'state':<8} {'in A_c':^7} {'in B_c':^7}")
    for k, s in enumerate(x):
        in_A = 'x' if s in A_c.free_symbols else ''
        in_B = 'x' if s in B_c.free_symbols else ''
        print(f"  {k:>2}  {str(s):<8} {in_A:^7} {in_B:^7}")
    print("\nstates appearing inside A_c:", sorted(map(str, A_c.free_symbols & states)))
    print("states appearing inside B_c:", sorted(map(str, B_c.free_symbols & states)))

    print("\n================ 6c: Eq. 17 (13 states) ================")
    print("x~_dot == A~ x~ + B~ u :",
          sp.simplify(x_aug_dot - (A_aug * x_aug + B_aug * u)) == sp.zeros(13, 1))
    print("last column of A~ (feeds the gravity state into v_z_dot):", list(A_aug[:, 12]))
