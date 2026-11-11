"""Step 3: Euler-angle rates from angular velocity (paper Eqs. 10-12).

Part A: invert E  ->  Theta_dot = E^-1 * omega          (Eq. 10)
Part B: small roll/pitch  ->  Theta_dot ~ ? * omega      (Eqs. 11-12)
"""
import sympy as sp
from step1_rotations import theta, psi, Rz
from step2_angular_velocity import E

# ---------------- Part A: exact inverse ----------------
print("\n================ Part A: exact inverse ================")
print("det(E) =", sp.simplify(E.det()))

E_inv = sp.simplify(E.inv())
print("\nE^-1 (derived) =")
sp.pprint(E_inv)

E_inv_paper = sp.Matrix([[sp.cos(psi) / sp.cos(theta), sp.sin(psi) / sp.cos(theta), 0],
                         [-sp.sin(psi),                sp.cos(psi),                0],
                         [sp.cos(psi) * sp.tan(theta), sp.sin(psi) * sp.tan(theta), 1]])
print("\nE^-1 - E^-1_paper (Eq. 10) =")
sp.pprint(sp.simplify(E_inv - E_inv_paper))

# ---------------- Part B: small roll and pitch ----------------
print("\n================ Part B: small roll/pitch ================")
E_inv_small = E_inv.subs(theta, 0)       # E^-1 has no phi in it, so only theta matters
print("E^-1 with theta = 0  (Eq. 11) =")
sp.pprint(E_inv_small)

print("\nRz(psi) =")
sp.pprint(Rz(psi))
print("\nequal to Rz(psi)   ? (what Eq. 12 says):", sp.simplify(E_inv_small - Rz(psi)) == sp.zeros(3, 3))
print("equal to Rz(psi)^T ?                    :", sp.simplify(E_inv_small - Rz(psi).T) == sp.zeros(3, 3))
