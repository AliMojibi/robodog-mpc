"""Step 8: force constraints (paper Eqs. 20-24), contact flag c_i in {0, 1} per foot.

1) normal force:     c_i * f_min <= f_z <= c_i * f_max        (box bounds on u)
2) friction pyramid: +-f_x - mu f_z <= 0,  +-f_y - mu f_z <= 0  (linear: D u <= 0)

For a swing foot (c = 0): 1) gives f_z = 0, then 2) gives f_x = f_y = 0.
"""
import casadi as ca
from step7_casadi_model import u, f

MU, F_MIN, F_MAX = 0.6, 10.0, 666.0          # paper, Table I

# ---------------- 2) friction pyramid: 4 rows per foot, 16 in total ----------------
pyramid = ca.vertcat(*[ca.vertcat( fi[0] - MU * fi[2],
                                  -fi[0] - MU * fi[2],
                                   fi[1] - MU * fi[2],
                                  -fi[1] - MU * fi[2]) for fi in f])
D = ca.jacobian(pyramid, u)                   # pyramid = D u  (linear, no constant term)


# ---------------- 1) normal-force bounds from the contact flags ----------------
def fz_bounds(contact):
    """Lower/upper bounds on [f_FR_z, f_FL_z, f_RR_z, f_RL_z] for contact flags c_i."""
    return [c * F_MIN for c in contact], [c * F_MAX for c in contact]


pyramid_fun = ca.Function('pyramid', [u], [pyramid])


def check(name, u_num, contact, tol=1e-9):
    """Print whether a force vector satisfies all constraints for a given contact state."""
    lb, ub = fz_bounds(contact)
    fz = [u_num[3 * i + 2] for i in range(4)]
    ok_fz = all(l - tol <= z <= h + tol for l, z, h in zip(lb, fz, ub))
    ok_pyr = bool((pyramid_fun(u_num).full() <= tol).all())
    print(f"{name:<44} f_z bounds: {'ok' if ok_fz else 'VIOLATED':<9} pyramid: {'ok' if ok_pyr else 'VIOLATED'}")


if __name__ == '__main__':
    print("D is constant (does not depend on u):", not ca.depends_on(D, u))
    print("D shape:", D.shape)
    print("\nD rows for foot FR (columns: f_FR_x, f_FR_y, f_FR_z):")
    print(ca.DM(D[0:4, 0:3]))

    W = 43 * 9.81                              # robot weight [N]
    all_stance, trot = [1, 1, 1, 1], [1, 0, 0, 1]   # trot: FR + RL in stance

    print("\nExamples:")
    check("standing, W/4 per foot",
          [0, 0, W / 4] * 4, all_stance)
    check("standing, one foot pushes sideways 0.5*f_z",
          [0, 0.5 * W / 4, W / 4] + [0, 0, W / 4] * 3, all_stance)
    check("standing, one foot pushes sideways 0.8*f_z",
          [0, 0.8 * W / 4, W / 4] + [0, 0, W / 4] * 3, all_stance)
    check("trot, stance feet W/2, swing feet zero",
          [0, 0, W / 2, 0, 0, 0, 0, 0, 0, 0, 0, W / 2], trot)
    check("trot, swing foot FL still pushing W/4",
          [0, 0, W / 2, 0, 0, W / 4, 0, 0, 0, 0, 0, W / 2], trot)
    check("standing, one foot pulls (f_z = -20 N)",
          [0, 0, -20] + [0, 0, W / 3] * 3, all_stance)
