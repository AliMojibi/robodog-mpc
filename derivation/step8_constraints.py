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

# ---------------- 3) exact friction cone (nonlinear), one row per foot ----------------
#   h_i = sqrt(f_x^2 + f_y^2 + eps^2) - mu * f_z <= 0
# eps makes the square root smooth at f_x = f_y = 0 (SQP needs derivatives there).
# mu is a symbol, so it can be a runtime parameter of the solver (config.yaml).
CONE_EPS = 1e-3                               # [N], shrinks the cone only negligibly
mu = ca.SX.sym('mu')
cone = ca.vertcat(*[ca.sqrt(fi[0] ** 2 + fi[1] ** 2 + CONE_EPS ** 2) - mu * fi[2] for fi in f])
cone_fun = ca.Function('cone', [u, mu], [cone])

# ---------------- 4) inscribed polygon (DEFAULT friction model) ----------------
# The cone's circle (radius mu f_z) is replaced by a regular n-gon whose corners lie ON the circle.
# Face j has outward direction theta_j = 2 pi j / n and distance mu cos(pi/n) f_z from the center:
#   cos(theta_j) f_x + sin(theta_j) f_y - mu cos(pi/n) f_z <= 0        j = 0 .. n-1
# - inside the cone -> a foot can never exceed mu (no slipping); loses at most 1 - cos(pi/n)
#   of the friction (n = 16: 1.9 %)
# - linear in u for a given mu -> SQP linearizes it exactly -> converges as fast as the pyramid
# - mu is a symbol -> runtime parameter (config.yaml), unlike the pyramid where mu sits in D
N_POLY = 16


def polygon_expr(n=N_POLY):
    rows = []
    for fi in f:
        for j in range(n):
            th = 2 * ca.pi * j / n
            rows.append(ca.cos(th) * fi[0] + ca.sin(th) * fi[1] - mu * ca.cos(ca.pi / n) * fi[2])
    return ca.vertcat(*rows)


polygon = polygon_expr()                      # 4 * N_POLY = 64 rows
polygon_fun = ca.Function('polygon', [u, mu], [polygon])

BIG = 1e9                                     # "no bound" for acados


def u_bounds(contact):
    """Bounds on all 12 force components from the contact flags.
    stance: f_x, f_y free (the cone limits them), f_min <= f_z <= f_max
    swing : f = 0 (all three components)"""
    lb, ub = [], []
    for c in contact:
        if c:
            lb += [-BIG, -BIG, F_MIN]
            ub += [+BIG, +BIG, F_MAX]
        else:
            lb += [0.0, 0.0, 0.0]
            ub += [0.0, 0.0, 0.0]
    return lb, ub


def cone_upper(contact):
    """Upper bound of h per foot: 0 for stance (cone active), BIG for swing (cone switched off,
    because with f = 0 the smoothed cone would read eps <= 0, which is infeasible)."""
    return [0.0 if c else BIG for c in contact]


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

    # ---------------- pyramid vs. exact cone ----------------
    print(f"\nPyramid vs. exact cone vs. {N_POLY}-gon (mu = 0.6), one foot with f_z = 100 N:")
    print(f"  {'f_x':>6} {'f_y':>6}  |f_t|/f_z   pyramid   cone   {N_POLY}-gon")
    for fx, fy in [(50, 0), (58, 0), (60, 0), (40, 40), (42.4, 42.4), (60, 60)]:
        u1 = [fx, fy, 100.0] + [0, 0, 100.0] * 3
        pyr_ok = bool((pyramid_fun(u1).full()[0:4] <= 1e-9).all())
        cone_ok = bool(cone_fun(u1, MU).full()[0] <= 1e-9)
        poly_ok = bool((polygon_fun(u1, MU).full()[0:N_POLY] <= 1e-9).all())
        print(f"  {fx:6.1f} {fy:6.1f}  {((fx**2 + fy**2) ** 0.5) / 100:8.3f}   "
              f"{'ok' if pyr_ok else 'VIOL':<8}  {'ok' if cone_ok else 'VIOL':<5}  {'ok' if poly_ok else 'VIOL'}")
    print(f"  {N_POLY}-gon face distance: mu * cos(pi/{N_POLY}) = {MU * ca.cos(ca.pi / N_POLY):.4f}")

    print("\nu_bounds for trot contact [1, 0, 0, 1]:")
    lb, ub = u_bounds([1, 0, 0, 1])
    for i, n in enumerate(['FR', 'FL', 'RR', 'RL']):
        print(f"  {n}: lb {lb[3*i:3*i+3]}  ub {ub[3*i:3*i+3]}")
    print("cone upper bounds:", cone_upper([1, 0, 0, 1]))
