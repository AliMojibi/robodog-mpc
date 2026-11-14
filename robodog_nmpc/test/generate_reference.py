"""Reference data for the C++ test PythonReference.ClosedLoop (test/test_nmpc_solver.cpp).

A closed loop in Python with the acados Python interface on the same OCP (srbd_ocp.build_ocp(config.yaml)):
    measured state -> contact schedule + current feet -> one SQP-RTI step -> forces -> plant (step 11) -> ...
The per-step logic is an independent Python version of NmpcSolver::run(), form (b) (contact schedule + current
feet, Raibert footholds): parameters, contact bounds, reference, shifted warm start / cold start, and after a
solver failure: reset + the previous plan's next step with swing feet zeroed. The C++ test feeds the recorded
inputs to the C++ class and must reproduce the recorded outputs.

Rerun after changing the OCP, config.yaml or NmpcSolver::run() (needs ACADOS_SOURCE_DIR, LD_LIBRARY_PATH):
    python test/generate_reference.py [--out path]
"""
import argparse
import math
import os
import sys
import tempfile

import numpy as np

PKG = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(PKG, 'scripts'))
sys.path.insert(0, os.path.join(PKG, 'scripts', 'derivation'))
from srbd_ocp import load_config, build_ocp, force_bounds, NX, NU   # noqa: E402
import step11_closed_loop as step11                                # noqa: E402  plant_step (the "robot")

OUT = os.path.join(PKG, 'test', 'reference', 'closed_loop.txt')
G = 9.81                     # as in nmpc_solver.cpp (cold start)
OK, SOLVER_FAILED = 0, 5     # robodog_nmpc::Status values

# scenario: (start time [s], v_x, v_y, yaw rate) in the heading frame; trot from T_TROT, pairs switch every PHASE steps
SCENARIO = [(0.0, 0.0, 0.0, 0.0), (0.75, 0.5, 0.0, 0.0), (1.75, 0.5, 0.0, 0.6), (2.75, 0.5, 0.0, 0.0)]
T_END, T_TROT, PHASE = 3.25, 0.25, 4


def rot(yaw, v):
    """Rz(yaw) v for a 2D vector, written out as in Eigen::Rotation2D (bit-identical to the C++ code)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def raibert(cfg, com, yaw, cmd, leg, stance, ground):
    """Foothold: hip on the path + v_cmd * T_stance / 2 (same operation order as detail::predictFeet)."""
    hip = cfg['robot']['hip_offsets_xy'][2 * leg:2 * leg + 2]
    v_ref = rot(yaw, cmd[:2])
    xy = (com + rot(yaw, hip)) + v_ref * (stance * cfg['horizon']['dt']) / 2.0
    return np.array([xy[0], xy[1], ground])


def predict_feet(cfg, x, cmd, schedule, current):
    """Python version of detail::predictFeet (form b)."""
    n, dt = len(schedule), cfg['horizon']['dt']
    ground = min(p[2] for p in current)
    out = np.zeros((n + 1, 4, 3))
    for leg in range(4):
        com, yaw, touchdown, foothold = x[3:5].copy(), x[2], -1, current[leg]
        for k in range(n + 1):
            j = min(k, n - 1)
            if 0 < k < n and schedule[k][leg] and not schedule[k - 1][leg]:
                stance = 0
                while k + stance < n and schedule[k + stance][leg]:
                    stance += 1
                foothold, touchdown = raibert(cfg, com, yaw, cmd, leg, stance, ground), k
            out[k, leg] = foothold if (schedule[j][leg] and touchdown >= 0) else current[leg]
            if k < n:
                com = com + rot(yaw, cmd[:2]) * dt
                yaw += cmd[2] * dt
    return out


class PyNmpc:
    """Python version of NmpcSolver::run() on the acados Python interface."""

    def __init__(self, cfg):
        from acados_template import AcadosOcpSolver
        self.cfg, self.N = cfg, cfg['horizon']['N']
        ocp = build_ocp(cfg)
        self.tmp = tempfile.mkdtemp(prefix='robodog_reference_')
        ocp.code_gen_options.code_export_directory = os.path.join(self.tmp, 'c_generated_code')
        ocp.code_gen_options.json_file = os.path.join(self.tmp, 'ocp.json')
        self.solver = AcadosOcpSolver(ocp, verbose=False)
        self.have_plan, self.reset_needed, self.since = False, False, 0
        self.X, self.U, self.contact_plan = None, None, None

    def run(self, x0, cmd, schedule, feet):
        c, N, s = self.cfg, self.N, self.solver
        r, fr = c['robot'], c['friction']
        self.since += 1
        for k in range(N + 1):
            s.set(k, 'p', np.concatenate([feet[k].ravel(), [r['mass']], r['inertia'], [fr['mu']]]))
        yref = np.zeros(19)
        yref[0], yref[1], yref[3], yref[4] = cmd[0], cmd[1], cmd[2], r['nominal_height']
        for k in range(N):
            lb, ub = force_bounds(schedule[k], fr['f_min'], fr['f_max'])
            s.constraints_set(k, 'lbu', lb)
            s.constraints_set(k, 'ubu', ub)
            s.cost_set(k, 'yref', yref)
        s.cost_set(N, 'yref', yref[:7])
        if self.reset_needed:
            s.reset()
            self.reset_needed = False
        for k in range(N + 1):
            if self.have_plan:
                s.set(k, 'x', self.X[min(k + self.since, N)])
                if k < N:
                    s.set(k, 'u', self.U[min(k + self.since, N - 1)])
            else:
                s.set(k, 'x', x0)
                if k < N:
                    u = np.zeros(NU)
                    n_stance = sum(schedule[k])
                    for leg in range(4):
                        if schedule[k][leg]:
                            u[3 * leg + 2] = r['mass'] * G / n_stance
                    s.set(k, 'u', u)
        s.constraints_set(0, 'lbx', x0)
        s.constraints_set(0, 'ubx', x0)
        status = s.solve()
        qp_iter = int(s.get_stats('qp_iter')[-1])
        u0 = s.get(0, 'u')

        def forces(u, contact):           # swing feet as exact zeros
            f = u.copy()
            for leg in range(4):
                if not contact[leg]:
                    f[3 * leg:3 * leg + 3] = 0.0
            return f

        if status != 0 or not np.all(np.isfinite(u0)):
            self.reset_needed = True
            if not self.have_plan:
                return SOLVER_FAILED, status, qp_iter, np.zeros(NU)
            k = min(self.since, N - 1)
            return SOLVER_FAILED, status, qp_iter, forces(self.U[k], [a and b for a, b in zip(self.contact_plan[k], schedule[0])])
        self.X = np.array([s.get(k, 'x') for k in range(N + 1)])
        self.U = np.array([s.get(k, 'u') for k in range(N)])
        self.contact_plan = [list(c_) for c_ in schedule]
        self.have_plan, self.since = True, 0
        return OK, status, qp_iter, forces(u0, schedule[0])


def contact_at(j):
    if j * step11.DT < T_TROT - 1e-9:
        return [True, True, True, True]
    a = ((j - round(T_TROT / step11.DT)) // PHASE) % 2 == 0
    return [a, not a, not a, a]


def command_at(j):
    return np.array(max(row for row in SCENARIO if row[0] <= j * step11.DT + 1e-9)[1:])


def schedule_hex(schedule):
    return ''.join('%x' % sum(1 << leg for leg in range(4) if c[leg]) for c in schedule)


def main(out):
    cfg = load_config(os.path.join(PKG, 'config', 'config.yaml'))
    r = cfg['robot']
    # the plant of step 11 uses the derivation's constants: they must be the config values
    assert (step11.MASS, list(step11.INERTIA), step11.DT, step11.N) == \
        (r['mass'], r['inertia'], cfg['horizon']['dt'], cfg['horizon']['N']), 'step 11 constants differ from config.yaml'
    N = cfg['horizon']['N']
    mpc = PyNmpc(cfg)
    x = np.zeros(NX)
    x[5] = r['nominal_height']
    feet = np.array([[r['hip_offsets_xy'][2 * l], r['hip_offsets_xy'][2 * l + 1], 0.0] for l in range(4)])
    lines = []
    for i in range(int(round(T_END / step11.DT))):
        cmd = command_at(i)
        schedule = [contact_at(i + k) for k in range(N)]
        for leg in range(4):              # touchdown now: the foot lands on its Raibert foothold
            if schedule[0][leg] and not contact_at(i - 1)[leg]:
                stance = 0
                while stance < N and schedule[stance][leg]:
                    stance += 1
                feet[leg] = raibert(cfg, x[3:5], x[2], cmd, leg, stance, min(p[2] for p in feet))
        x_in, feet_in = x.copy(), feet.copy()
        st, status, qp_iter, u = mpc.run(x_in, cmd, schedule, predict_feet(cfg, x_in, cmd, schedule, feet_in))
        lines.append(' '.join([str(i)] + ['%.17g' % v for v in x_in] + ['%.17g' % v for v in cmd] +
                              [schedule_hex(schedule)] + ['%.17g' % v for v in feet_in.ravel()] +
                              [str(st), str(status), str(qp_iter)] + ['%.17g' % v for v in u]))
        x = step11.plant_step(x, u, feet)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        f.write('# Generated by test/generate_reference.py (do not edit). One control step per line:\n'
                '# step, state x (12: rpy, position, angular velocity, linear velocity), command (vx, vy, yaw rate),\n'
                '# contact schedule (one hex digit per interval: FR=1 FL=2 RR=4 RL=8), current feet (12),\n'
                '# expected: Status, acados status, QP iterations, forces (12: FR, FL, RR, RL; x, y, z)\n')
        f.write('\n'.join(lines) + '\n')
    sts = [int(l.split()[1 + 12 + 3 + 1 + 12]) for l in lines]
    print(f"{len(lines)} steps written to {out}: {sts.count(OK)} Ok, {sts.count(SOLVER_FAILED)} SolverFailed; "
          f"final position {x[3]:.3f} {x[4]:.3f} {x[5]:.3f} m, yaw {x[2]:.3f} rad, v_x {x[9]:.3f} m/s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=OUT, help='output file (default: the committed reference)')
    main(ap.parse_args().out)
