"""Step 4b: independent check of I_world = R * I_body * R^T.

Model the body as a few point masses. Compute the inertia directly from its
definition  I = sum m_k (|r_k|^2 * 1 - r_k r_k^T)  in the world frame (after
rotating the points), and compare with R * I_body * R^T.
"""
import sympy as sp
from step1_rotations import phi, theta, psi, R


def inertia_from_points(masses, points):
    I = sp.zeros(3, 3)
    for m, r in zip(masses, points):
        I += m * ((r.T * r)[0] * sp.eye(3) - r * r.T)
    return I


# A few point masses forming a long, narrow, asymmetric body (positions in the body frame)
masses = [2, 1, 3, 1.5, 0.5]
points_body = [sp.Matrix(p) for p in [[0.3, 0.1, 0.05], [-0.3, -0.1, 0.0],
                                      [0.1, -0.05, -0.05], [-0.2, 0.08, 0.02], [0.25, -0.1, 0.1]]]

# some arbitrary orientation, converted to a numeric rotation matrix
R_num = R.subs({phi: 0.3, theta: -0.5, psi: 1.2}).evalf()

I_body = inertia_from_points(masses, points_body)
I_world_direct = inertia_from_points(masses, [R_num * r for r in points_body])  # rotate points first
I_world_formula = R_num * I_body * R_num.T
I_world_wrong = R_num.T * I_body * R_num                                         # opposite order


def max_abs(M):
    return max(abs(v) for v in M)


print("I_world from rotated points:")
sp.pprint(I_world_direct.evalf(4))
print("\nI_world from R * I_body * R^T:")
sp.pprint(I_world_formula.evalf(4))
print("\nmax difference, R * I_body * R^T :", max_abs(I_world_direct - I_world_formula))
print("max difference, R^T * I_body * R :", max_abs(I_world_direct - I_world_wrong))
