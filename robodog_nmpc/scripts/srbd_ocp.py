"""Optimal control problem of the robodog NMPC (single-rigid-body model), built from config.yaml.

The model itself is NOT defined here: it is imported from the step-by-step derivation in
scripts/derivation/ (single source of truth). This file only adds what the derivation keeps as fixed
demo values: reading config.yaml, the weights and force limits, and assembling the AcadosOcp.

    from derivation step 7: x, u, P, x_dot       SRBD dynamics
    from derivation step 8: mu, polygon_expr(n), cone, BIG    friction (symbolic, mu as parameter)
    from derivation step 9: y, make_y_ref         least-squares cost vector and reference

    min   sum_{k=0}^{N-1} 1/2 ||y(x_k, u_k) - y_ref||^2_W  +  1/2 ||y_e(x_N) - y_ref_e||^2_W_e
    s.t.  x_0 = x_measured,  x_{k+1} = RK4(f(x_k, u_k, p_k), dt),  friction,  contact bounds

    x (12) = [Theta, p, omega, v]     u (12) = [f_FR, f_FL, f_RR, f_RL]     p (17) = [P (16), mu]

Usage:
    from srbd_ocp import load_config, build_ocp
    ocp = build_ocp(load_config('config/config.yaml'))
"""
import os
import re
import sys
import numpy as np
import casadi as ca
import yaml
from acados_template import AcadosModel, AcadosOcp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'derivation'))
from step7_casadi_model import x, u, P, x_dot                        # noqa: E402  dynamics
from step8_constraints import mu, polygon_expr, cone, BIG           # noqa: E402  friction
from step9_cost import y, make_y_ref                                 # noqa: E402  cost vector

p = ca.vertcat(P, mu)                     # per-stage parameters: [p_foot (12), m, Ixx, Iyy, Izz, mu]
NX, NU, NP = x.shape[0], u.shape[0], p.shape[0]
NY, NY_E = y.shape[0], 7                  # terminal cost: the 7 state terms of y (no forces)


class ConfigError(ValueError):
    """Raised when config.yaml is missing keys or has invalid values (all problems listed at once)."""


# ---- schema: one checker per key; every group and key must be present, unknown keys are errors ----
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _scalar(cond, text):
    def check(v):
        if isinstance(v, str):
            try:
                float(v)
                return (f"is the text '{v}', not a number: YAML reads e.g. 1e-6 without a decimal point "
                        f"as text, write 1.0e-6")
            except ValueError:
                return f"must be a number ({text}), got the text '{v}'"
        if not _num(v) or not cond(v):
            return f"must be a number {text}, got {v!r}"
    return check


def _int(cond, text):
    def check(v):
        if isinstance(v, bool) or not isinstance(v, int) or not cond(v):
            return f"must be an integer {text}, got {v!r}"
    return check


def _list(n, item_check):
    def check(v):
        if not isinstance(v, list) or len(v) != n:
            return f"must be a list of {n} entries, got {v!r}"
        for i, item in enumerate(v):
            msg = item_check(item)
            if msg:
                return f"entry {i}: {msg}"
    return check


def _choice(options):
    def check(v):
        if v not in options:
            return f"must be one of {options}, got {v!r}"
    return check


POSITIVE = _scalar(lambda v: v > 0, '> 0')
NONNEG = _scalar(lambda v: v >= 0, '>= 0')
ANY = _scalar(lambda v: True, '')

SCHEMA = {
    'robot': {'mass': POSITIVE, 'inertia': _list(3, POSITIVE), 'nominal_height': POSITIVE,
              'hip_offsets_xy': _list(8, ANY)},
    'horizon': {'N': _int(lambda v: v >= 1, '>= 1'), 'dt': POSITIVE},
    'friction': {'model': _choice(['polygon', 'cone', 'pyramid']),
                 'polygon_sides': _int(lambda v: v >= 3, '>= 3'),
                 'mu': POSITIVE, 'f_min': NONNEG, 'f_max': POSITIVE},
    'cost': {'w_velocity': _list(3, NONNEG), 'w_yaw_rate': NONNEG, 'w_height': NONNEG,
             'w_roll_pitch': _list(2, NONNEG),
             'w_force': POSITIVE,              # > 0: keeps the QP Hessian regular in the forces
             'terminal_weight_scale': NONNEG},
    'solver': {'nlp_solver_type': _choice(['SQP_RTI', 'SQP']),
               'qp_solver': _choice(['PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_HPIPM',
                                     'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_DAQP']),
               'hessian_approx': _choice(['GAUSS_NEWTON', 'EXACT']),
               'integrator_type': _choice(['ERK', 'IRK']),
               'sim_method_num_stages': _int(lambda v: 1 <= v <= 4, 'in 1..4'),
               'sim_method_num_steps': _int(lambda v: v >= 1, '>= 1'),
               'globalization': _choice(['FIXED_STEP', 'MERIT_BACKTRACKING']),
               'nlp_solver_max_iter': _int(lambda v: v >= 1, '>= 1'),
               'qp_solver_iter_max': _int(lambda v: v >= 1, '>= 1')},
    'codegen': {'solver_name': lambda v: None if isinstance(v, str) and re.fullmatch(r'[A-Za-z_]\w*', v)
                else f"must be a valid C identifier (letters, digits, _), got {v!r}",
                'export_dir': lambda v: None if isinstance(v, str) and v.strip()
                else f"must be a non-empty folder name, got {v!r}"},
}


def validate_config(cfg):
    """Check the parameter dictionary against SCHEMA; raise ConfigError listing every problem."""
    errors = []
    if not isinstance(cfg, dict):
        raise ConfigError("config: 'robodog_nmpc: ros__parameters:' must contain the parameter groups")
    for group in sorted(set(cfg) - set(SCHEMA)):
        errors.append(f"unknown group '{group}' (typo?)")
    for group, keys in SCHEMA.items():
        values = cfg.get(group)
        if not isinstance(values, dict):
            errors.append(f"missing group '{group}'")
            continue
        for key in sorted(set(values) - set(keys)):
            errors.append(f"{group}.{key}: unknown key (typo?)")
        for key, check in keys.items():
            if key not in values:
                errors.append(f"{group}.{key}: missing")
                continue
            msg = check(values[key])
            if msg:
                errors.append(f"{group}.{key}: {msg}")
    fr = cfg.get('friction', {})
    if not errors and fr['f_min'] >= fr['f_max']:
        errors.append(f"friction: f_min ({fr['f_min']}) must be smaller than f_max ({fr['f_max']})")
    if errors:
        raise ConfigError("invalid config.yaml:\n  - " + "\n  - ".join(errors))
    return cfg


def load_config(path):
    """Read the ROS 2 parameter file, validate it, and return the parameter dictionary."""
    with open(path) as f:
        try:
            doc = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"{path}: not valid YAML:\n{e}") from None
    try:
        cfg = doc['robodog_nmpc']['ros__parameters']
    except (TypeError, KeyError):
        raise ConfigError(f"{path}: expected the layout 'robodog_nmpc: ros__parameters: ...'") from None
    return validate_config(cfg)


def cost_weights(cfg):
    """Diagonal stage and terminal weights from the config (without acados' factor 1/2).
    Order of y (step 9): v_x, v_y, v_z (heading frame), yaw rate, height, roll, pitch, 12 forces."""
    c = cfg['cost']
    w_state = list(c['w_velocity']) + [c['w_yaw_rate'], c['w_height']] + list(c['w_roll_pitch'])
    w = np.array(w_state + [c['w_force']] * NU)
    w_e = np.array(w_state) * c['terminal_weight_scale']
    return w, w_e


def force_bounds(contact, f_min, f_max):
    """Bounds on all 12 forces from the contact flags [c_FR, c_FL, c_RR, c_RL] (1 = stance).
    Same rule as step 8's u_bounds, but with f_min / f_max from the config."""
    lb, ub = [], []
    for c in contact:
        lb += [-BIG, -BIG, f_min] if c else [0.0, 0.0, 0.0]
        ub += [+BIG, +BIG, f_max] if c else [0.0, 0.0, 0.0]
    return np.array(lb), np.array(ub)


def friction_pyramid_D(mu_value):
    """Paper's pyramid as a constant matrix (D u <= 0) with mu from the config. (Step 8 builds the
    same matrix with its fixed demo mu, so it is redefined here; comparison model only.)"""
    rows = np.array([[1, 0, -mu_value], [-1, 0, -mu_value], [0, 1, -mu_value], [0, -1, -mu_value]])
    D = np.zeros((16, NU))
    for i in range(4):
        D[4 * i:4 * i + 4, 3 * i:3 * i + 3] = rows
    return D


def default_parameters(cfg):
    """Parameter vector with feet under the hips: used as the code-generation default only."""
    r = cfg['robot']
    hips = np.array(r['hip_offsets_xy']).reshape(4, 2)
    feet = np.column_stack([hips, np.zeros(4)]).ravel()
    return np.concatenate([feet, [r['mass']], r['inertia'], [cfg['friction']['mu']]])


def build_ocp(cfg):
    """Assemble the AcadosOcp from the config. Does NOT generate code (see generate_solver.py)."""
    name = cfg['codegen']['solver_name']
    fr = cfg['friction']

    model = AcadosModel()
    model.name = name
    model.x, model.u, model.p = x, u, p
    model.f_expl_expr = x_dot

    ocp = AcadosOcp()
    ocp.model = model
    ocp.name = name                                    # fixed name -> no hash in file/function names
    ocp.parameter_values = default_parameters(cfg)

    # cost (acados uses 1/2 ||.||^2_W, so W = 2 * weights)
    w, w_e = cost_weights(cfg)
    y_ref = np.array(make_y_ref(0.0, 0.0, 0.0, cfg['robot']['nominal_height']))
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = y
    ocp.cost.W = 2 * np.diag(w)
    ocp.cost.yref = y_ref
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = y[:NY_E]
    ocp.cost.W_e = 2 * np.diag(w_e)
    ocp.cost.yref_e = y_ref[:NY_E]

    # contact: bounds on all 12 forces (default: all feet in stance; set per stage at runtime)
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu, ocp.constraints.ubu = force_bounds([1, 1, 1, 1], fr['f_min'], fr['f_max'])

    # friction
    if fr['model'] in ('polygon', 'cone'):
        h = polygon_expr(fr['polygon_sides']) if fr['model'] == 'polygon' else cone
        nh = h.shape[0]
        # node 0 has its own nonlinear constraint (con_h_expr_0); without it u_0 would be unconstrained
        ocp.model.con_h_expr_0 = h
        ocp.constraints.lh_0, ocp.constraints.uh_0 = -BIG * np.ones(nh), np.zeros(nh)
        ocp.model.con_h_expr = h
        ocp.constraints.lh, ocp.constraints.uh = -BIG * np.ones(nh), np.zeros(nh)
    elif fr['model'] == 'pyramid':
        D = friction_pyramid_D(fr['mu'])
        ocp.constraints.C = np.zeros((D.shape[0], NX))
        ocp.constraints.D = D
        ocp.constraints.lg, ocp.constraints.ug = -BIG * np.ones(D.shape[0]), np.zeros(D.shape[0])
    else:
        raise ValueError(f"unknown friction model '{fr['model']}' (polygon | cone | pyramid)")

    ocp.constraints.x0 = np.zeros(NX)                 # placeholder; the measured state is set at runtime

    # solver options
    s = cfg['solver']
    ocp.solver_options.N_horizon = cfg['horizon']['N']
    ocp.solver_options.tf = cfg['horizon']['N'] * cfg['horizon']['dt']
    for key in ['nlp_solver_type', 'qp_solver', 'hessian_approx', 'integrator_type',
                'sim_method_num_stages', 'sim_method_num_steps', 'globalization',
                'nlp_solver_max_iter', 'qp_solver_iter_max']:
        setattr(ocp.solver_options, key, s[key])
    return ocp


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(here, '..', 'config', 'config.yaml'))
    ocp = build_ocp(cfg)
    fr = cfg['friction']['model']
    print(f"solver name      : {ocp.name}")
    print(f"dimensions       : nx = {NX}, nu = {NU}, np = {NP}, ny = {NY}, ny_e = {NY_E}")
    print(f"horizon          : N = {ocp.solver_options.N_horizon}, tf = {ocp.solver_options.tf:.3f} s")
    print(f"friction         : {fr}, {16 if fr == 'pyramid' else ocp.model.con_h_expr.shape[0]} rows")
    print(f"solver           : {ocp.solver_options.nlp_solver_type}, {ocp.solver_options.qp_solver}")
    print(f"default p        : {np.round(ocp.parameter_values, 3)}")
    print(f"W diagonal       : {np.diag(ocp.cost.W)[:7]} ... forces {np.diag(ocp.cost.W)[7]:g}")
