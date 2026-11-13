"""Step 9: cost function (paper Eq. 18, adapted to velocity tracking).

stage cost  l(x, u) = || y(x, u) - y_ref ||^2_W

y = [v_heading(3), omega_z, p_z, phi, theta, f(12)]                     19 entries
v_heading = Rz(psi)^T v   (velocity in the robot's heading frame)
"""
import casadi as ca
from step7_casadi_model import x, u, Rz, G0

phi, theta, psi = x[0], x[1], x[2]
p_z = x[5]
omega_z = x[8]
v = x[9:12]

v_heading = Rz(psi).T @ v

y = ca.vertcat(v_heading,          # 0-2  : v_x, v_y, v_z in heading frame
               omega_z,            # 3    : yaw rate
               p_z,                # 4    : height
               phi, theta,         # 5-6  : roll, pitch
               u)                  # 7-18 : forces

# weights (paper, Table I)
W_V, W_YAW_RATE, W_Z, W_THETA, W_FORCE = 1.0, 1.0, 50.0, 1.0, 1e-6
W = ca.diag(ca.vertcat(W_V, W_V, W_V, W_YAW_RATE, W_Z, W_THETA, W_THETA, *[W_FORCE] * 12))


def make_y_ref(vx_cmd, vy_cmd, yaw_rate_cmd, z_des):
    """Reference for y from the operator command (forces referenced to 0, as in the paper)."""
    return [vx_cmd, vy_cmd, 0.0, yaw_rate_cmd, z_des, 0.0, 0.0] + [0.0] * 12


y_ref = ca.SX.sym('y_ref', 19)
err = y - y_ref
cost = ca.Function('cost', [x, u, y_ref], [err.T @ W @ err], ['x', 'u', 'y_ref'], ['l'])
y_fun = ca.Function('y', [x, u], [y])

if __name__ == '__main__':
    import math
    labels = ['v_x (heading)', 'v_y (heading)', 'v_z (heading)', 'omega_z', 'p_z', 'phi', 'theta'] + \
             [f'f_{n}_{c}' for n in ['FR', 'FL', 'RR', 'RL'] for c in 'xyz']

    print("================ y(x, u): the cost vector (19 entries) ================")
    for k in range(y.shape[0]):
        print(f"  y[{k:>2}] {labels[k]:<14} = {y[k]}")

    print("\n================ W: diagonal weight matrix (19 x 19) ================")
    W_diag = ca.DM(ca.diag(W)).full().ravel()
    for k in range(y.shape[0]):
        print(f"  W[{k:>2},{k:>2}] {labels[k]:<14} = {W_diag[k]:g}")

    m, z_des = 43.0, 0.45
    u_stand = [0, 0, m * G0 / 4] * 4

    print("\n================ y_ref for command v_x = 0.5, v_y = 0.2, yaw rate = 0.3, z = 0.45 ================")
    ref_example = make_y_ref(0.5, 0.2, 0.3, z_des)
    for k in range(len(ref_example)):
        print(f"  y_ref[{k:>2}] {labels[k]:<14} = {ref_example[k]:g}")

    print("\n================ worked example: cost broken down entry by entry ================")
    print("state: psi = 30 deg, z = 0.43, roll = 2 deg, world v = [0.4, 0.3, 0], yaw rate 0.2; forces W/4 per foot")
    x_ex = [math.radians(2), 0, math.radians(30),  0, 0, 0.43,  0, 0, 0.2,  0.4, 0.3, 0]
    y_ex = y_fun(x_ex, u_stand).full().ravel()
    print(f"  {'entry':<14} {'y':>10} {'y_ref':>10} {'error':>10} {'W':>8} {'W*err^2':>10}")
    total = 0.0
    for k in range(y.shape[0]):
        e = y_ex[k] - ref_example[k]
        c = W_diag[k] * e ** 2
        total += c
        print(f"  {labels[k]:<14} {y_ex[k]:>10.4f} {ref_example[k]:>10.4f} {e:>10.4f} {W_diag[k]:>8g} {c:>10.5f}")
    print(f"  {'sum':<14} {'':>10} {'':>10} {'':>10} {'':>8} {total:>10.5f}")
    print(f"  cost() function gives: {float(cost(x_ex, u_stand, ref_example)):.5f}")

    print("\n================ examples ================")

    def state(psi_=0.0, z=z_des, vx=0.0, vy=0.0):
        return [0, 0, psi_,  0, 0, z,  0, 0, 0,  vx, vy, 0]

    def show(name, x_num, ref):
        l = float(cost(x_num, u_stand, ref))
        print(f"{name:<58} cost = {l:.4f}")

    print("\nExamples (forces = W/4 per foot in all cases):")
    show("standing at z_des, command zero",
         state(), make_y_ref(0, 0, 0, z_des))
    show("standing at z_des, command forward 1 m/s",
         state(), make_y_ref(1, 0, 0, z_des))
    show("moving world +x at 1 m/s, facing +x, command forward 1",
         state(vx=1), make_y_ref(1, 0, 0, z_des))
    show("moving world +y at 1 m/s, facing +y (psi=90), command fwd 1",
         state(psi_=math.pi / 2, vy=1), make_y_ref(1, 0, 0, z_des))
    show("moving world +y at 1 m/s, facing +x, command forward 1",
         state(vy=1), make_y_ref(1, 0, 0, z_des))
    show("standing 5 cm too low, command zero",
         state(z=z_des - 0.05), make_y_ref(0, 0, 0, z_des))
