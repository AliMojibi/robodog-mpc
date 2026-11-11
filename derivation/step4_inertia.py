"""Step 4: inertia tensor in the world frame (paper Eqs. 14-15).

I_world = R * I_body * R^T                      (Eq. 14, exact)
I_hat   = Rz(psi) * I_body * Rz(psi)^T          (Eq. 15, small roll/pitch)
I_hat^-1 = Rz(psi) * I_body^-1 * Rz(psi)^T      (cheap inverse, since Rz^-1 = Rz^T)
"""
import sympy as sp
from step1_rotations import phi, theta, psi, R, Rz

Ixx, Iyy, Izz = sp.symbols('I_xx I_yy I_zz', positive=True)
I_body = sp.diag(Ixx, Iyy, Izz)

# Exact world-frame inertia (Eq. 14)
I_world = R * I_body * R.T

# Small-angle version (Eq. 15)
I_hat = sp.simplify(Rz(psi) * I_body * Rz(psi).T)

# Its inverse, built without a general matrix inversion
I_hat_inv = sp.simplify(Rz(psi) * I_body.inv() * Rz(psi).T)

if __name__ == '__main__':
    print("I_body =")
    sp.pprint(I_body)

    print("\nR with phi = theta = 0 equals Rz(psi):",
          sp.simplify(R.subs({phi: 0, theta: 0}) - Rz(psi)) == sp.zeros(3, 3))
    print("I_world with phi = theta = 0 equals I_hat:",
          sp.simplify(I_world.subs({phi: 0, theta: 0}) - I_hat) == sp.zeros(3, 3))

    print("\nI_hat = Rz * I_body * Rz^T  (Eq. 15) =")
    sp.pprint(I_hat)

    print("\nI_hat^-1 = Rz * I_body^-1 * Rz^T =")
    sp.pprint(I_hat_inv)
    print("\nI_hat * I_hat^-1 is identity:", sp.simplify(I_hat * I_hat_inv) == sp.eye(3))
