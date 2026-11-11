"""Step 2: angular velocity from Euler-angle rates (paper Eq. 9).

Idea:  R_dot * R^T = [omega]x  (a skew-symmetric matrix holding the world-frame angular velocity).
Then   omega = E(Theta) * [phi_dot, theta_dot, psi_dot]^T   and we find E.
"""
import sympy as sp
from step1_rotations import phi, theta, psi, R

# Euler-angle rates as plain symbols
dphi, dtheta, dpsi = sp.symbols('phi_dot theta_dot psi_dot', real=True)

# 1) Time derivative of R by the chain rule
R_dot = R.diff(phi) * dphi + R.diff(theta) * dtheta + R.diff(psi) * dpsi

# 2) R_dot * R^T must be skew-symmetric  (S + S^T = 0)
S = sp.simplify(R_dot * R.T)

# 3) Read omega off the skew matrix:  S = [[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]]
omega = sp.Matrix([S[2, 1], S[0, 2], S[1, 0]])

# 4) omega is linear in the rates -> E is its Jacobian w.r.t. the rates
E = sp.simplify(omega.jacobian([dphi, dtheta, dpsi]))

# The paper's Eq. (9), for comparison
E_paper = sp.Matrix([[sp.cos(theta) * sp.cos(psi), -sp.sin(psi), 0],
                     [sp.cos(theta) * sp.sin(psi),  sp.cos(psi), 0],
                     [0,                            0,           1]])

if __name__ == '__main__':
    print("R_dot * R^T is skew-symmetric:", sp.simplify(S + S.T) == sp.zeros(3, 3))
    print("\nomega (world frame) =")
    sp.pprint(omega)
    print("\nE(Theta), where omega = E * [phi_dot, theta_dot, psi_dot] :")
    sp.pprint(E)
    print("\nE - E_paper (Eq. 9) =")
    sp.pprint(sp.simplify(E - E_paper))
