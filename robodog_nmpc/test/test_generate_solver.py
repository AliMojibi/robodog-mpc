"""Tests of the code-generation workflow: config.yaml -> scripts/generate_solver.py -> generated C solver.

Each test writes a modified copy of config/config.yaml to a temporary folder, runs generate_solver.py
as a separate process (like a user would), and checks either the generated code or the error message.
Generation tests write to <package>/solver_test/ (removed afterwards); the real solver/ is never touched.

Run (from anywhere; generation tests are skipped if ACADOS_SOURCE_DIR is not set):
    ACADOS_SOURCE_DIR=$HOME/acados LD_LIBRARY_PATH=$HOME/acados/lib \\
        python -m unittest discover -s robodog_nmpc/test -v
"""
import copy
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

PKG = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
GENERATE = os.path.join(PKG, 'scripts', 'generate_solver.py')
TEST_DIR = 'solver_test'
HAVE_ACADOS = bool(os.environ.get('ACADOS_SOURCE_DIR'))

with open(os.path.join(PKG, 'config', 'config.yaml')) as f:
    BASE = yaml.safe_load(f)


def params(doc):
    return doc['robodog_nmpc']['ros__parameters']


class GenerateSolverTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='robodog_nmpc_test_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        shutil.rmtree(os.path.join(PKG, TEST_DIR), ignore_errors=True)

    # ---------------- helpers ----------------
    def generate(self, edit=None, text_edit=None, env=None):
        """Write a modified config, run generate_solver.py from the temp folder, return the result."""
        doc = copy.deepcopy(BASE)
        params(doc)['codegen']['export_dir'] = TEST_DIR
        if edit:
            edit(params(doc))
        text = yaml.safe_dump(doc, sort_keys=False)
        if text_edit:
            text = text_edit(text)
        path = os.path.join(self.tmp, 'config.yaml')
        with open(path, 'w') as f:
            f.write(text)
        before = set(os.listdir(self.tmp))
        r = subprocess.run([sys.executable, GENERATE, '--config', path], cwd=self.tmp,
                           env=env or os.environ, capture_output=True, text=True)
        self.assertEqual(set(os.listdir(self.tmp)), before, 'generation left files in the working directory')
        return r

    def assert_rejected(self, r, *fragments):
        """The run must fail with a clear message containing every fragment, and generate nothing."""
        self.assertNotEqual(r.returncode, 0, 'invalid config was accepted')
        msg = r.stderr + r.stdout
        for frag in fragments:
            self.assertIn(frag, msg)
        self.assertNotIn('Traceback', msg, 'error was not reported cleanly')

    def generated(self):
        """Dimension macros from the generated header, the problem json, and a checksum of the C code."""
        d = os.path.join(PKG, TEST_DIR)
        with open(os.path.join(d, 'acados_solver_srbd_nmpc.h')) as f:
            header = f.read()
        macros = {k: int(v) for k, v in re.findall(r'#define SRBD_NMPC_(\w+)\s+(\d+)\s*$', header, re.M)}
        with open(os.path.join(d, 'srbd_nmpc.json')) as f:
            js = json.load(f)
        md5 = hashlib.md5()
        for p in sorted(glob.glob(os.path.join(d, '**', '*.c'), recursive=True) + glob.glob(os.path.join(d, '*.h'))):
            with open(p, 'rb') as f:
                md5.update(f.read())
        md5 = md5.hexdigest()
        self.assertTrue(os.path.isfile(os.path.join(d, 'libacados_ocp_solver_srbd_nmpc.so')))
        return macros, js, md5

    def codegen_config(self):
        """Entries of srbd_nmpc_codegen_config.h as {'group.key': value} (int, float or str)."""
        with open(os.path.join(PKG, TEST_DIR, 'srbd_nmpc_codegen_config.h')) as f:
            text = f.read()
        conv = {'INT': int, 'DOUBLE': float, 'STRING': lambda s: s.strip('"')}
        entries = re.findall(r'^\s+(INT|DOUBLE|STRING)\("(\w+)", "(\w+)", ("?[\w.+-]+"?)\)', text, re.M)
        out = {f'{g}.{k}': conv[t](v) for t, g, k, v in entries}
        self.assertEqual(len(out), len(entries), 'duplicate entry in the codegen-config header')
        self.assertEqual(len(entries), text.count('("'), 'header entry the test could not parse')
        return out

    @staticmethod
    def set(**groups):
        def edit(cfg):
            for group, values in groups.items():
                cfg[group].update(values)
        return edit

    # ---------------- valid changes show up in the generated code ----------------
    @unittest.skipUnless(HAVE_ACADOS, 'ACADOS_SOURCE_DIR not set')
    def test_baseline_and_determinism(self):
        self.assertEqual(self.generate().returncode, 0)
        m, js, md5_first = self.generated()
        self.assertEqual((m['N'], m['NX'], m['NU'], m['NP'], m['NH'], m['NH0'], m['NY'], m['NYN']),
                         (16, 12, 12, 17, 64, 64, 19, 7))
        self.assertAlmostEqual(js['solver_options']['tf'], 0.4)
        # codegen-config header: every [codegen] setting except export_dir, values as in config.yaml
        cc, p = self.codegen_config(), params(BASE)
        expected = (['horizon.N', 'horizon.dt', 'friction.model', 'friction.polygon_sides', 'codegen.solver_name']
                    + [f'solver.{k}' for k in p['solver']])
        self.assertEqual(sorted(cc), sorted(expected))
        for name, value in cc.items():
            group, key = name.split('.')
            self.assertEqual(value, p[group][key], name)
        self.assertIsInstance(cc['horizon.dt'], float)
        # nothing machine-specific in the files that get committed (generated .c/.h and the marker)
        d = os.path.join(PKG, TEST_DIR)
        for path in glob.glob(os.path.join(d, '**', '*.[ch]'), recursive=True) + [os.path.join(d, '.generated_by_generate_solver')]:
            with open(path) as f:
                self.assertNotIn(os.path.expanduser('~'), f.read(), os.path.relpath(path, d))
        self.assertEqual(self.generate().returncode, 0)          # regenerate over the previous output
        self.assertEqual(self.generated()[2], md5_first, 'same config must give identical code')

    @unittest.skipUnless(HAVE_ACADOS, 'ACADOS_SOURCE_DIR not set')
    def test_horizon(self):
        self.assertEqual(self.generate(self.set(horizon={'N': 20, 'dt': 0.02})).returncode, 0)
        m, js, _ = self.generated()
        self.assertEqual(m['N'], 20)
        self.assertAlmostEqual(js['solver_options']['tf'], 0.4)
        cc = self.codegen_config()
        self.assertEqual((cc['horizon.N'], cc['horizon.dt']), (20, 0.02))
        # an integer dt in the YAML must still be written as a double
        self.assertEqual(self.generate(self.set(horizon={'N': 4, 'dt': 1})).returncode, 0)
        self.assertEqual(self.codegen_config()['horizon.dt'], 1.0)
        with open(os.path.join(PKG, TEST_DIR, 'srbd_nmpc_codegen_config.h')) as f:
            self.assertIn('DOUBLE("horizon", "dt", 1.0)', f.read())

    @unittest.skipUnless(HAVE_ACADOS, 'ACADOS_SOURCE_DIR not set')
    def test_friction_models(self):
        for edit, nh, ng, model, sides, mu in [
                (self.set(friction={'polygon_sides': 8}), 32, 0, 'polygon', 8, None),
                (self.set(friction={'model': 'cone'}), 4, 0, 'cone', 16, None),
                (self.set(friction={'model': 'pyramid', 'mu': 0.5}), 0, 16, 'pyramid', 16, 0.5)]:
            self.assertEqual(self.generate(edit).returncode, 0)
            m, js, _ = self.generated()
            self.assertEqual((m['NH'], m['NH0'], m['NG']), (nh, nh, ng))
            cc = self.codegen_config()                               # mu is listed only when compiled in
            self.assertEqual((cc['friction.model'], cc['friction.polygon_sides'], cc.get('friction.mu')),
                             (model, sides, mu))
        self.assertEqual(js['constraints']['D'][0][:3], [1.0, 0.0, -0.5])   # pyramid: mu compiled in

    @unittest.skipUnless(HAVE_ACADOS, 'ACADOS_SOURCE_DIR not set')
    def test_runtime_defaults_follow_config(self):
        edit = self.set(robot={'mass': 30.0}, friction={'mu': 0.5, 'f_min': 5.0}, cost={'w_height': 10.0})
        self.assertEqual(self.generate(edit).returncode, 0)
        _, js, _ = self.generated()
        self.assertEqual(js['parameter_values'][12], 30.0)       # mass
        self.assertEqual(js['parameter_values'][16], 0.5)        # mu
        self.assertEqual(js['cost']['W'][4][4], 20.0)            # 2 * w_height (acados' factor 1/2)
        self.assertEqual(js['constraints']['lbu'][2], 5.0)       # f_min of FR

    # ---------------- invalid configs are rejected with a clear message ----------------
    def test_invalid_values(self):
        cases = [
            (self.set(friction={'model': 'polyogn'}), ["friction.model: must be one of"]),
            (self.set(cost={'w_velocity': [1.0, 1.0]}), ["cost.w_velocity: must be a list of 3 entries"]),
            (self.set(robot={'inertia': [-0.41, 2.1, 2.1]}), ["robot.inertia: entry 0: must be a number > 0"]),
            (self.set(horizon={'N': 0}), ["horizon.N: must be an integer >= 1"]),
            (self.set(horizon={'N': 16.5}), ["horizon.N: must be an integer >= 1"]),
            (self.set(friction={'polygon_sides': 2}), ["friction.polygon_sides: must be an integer >= 3"]),
            (self.set(friction={'f_min': 700.0}), ["f_min (700.0) must be smaller than f_max (666.0)"]),
            (self.set(cost={'w_force': 0.0}), ["cost.w_force: must be a number > 0"]),
            (self.set(solver={'nlp_solver_type': 'SQP_FAST'}), ["solver.nlp_solver_type: must be one of"]),
            (self.set(codegen={'solver_name': 'srbd-nmpc'}), ["codegen.solver_name: must be a valid C identifier"]),
            (self.set(robot={'mass': True}), ["robot.mass: must be a number > 0"]),
        ]
        for edit, fragments in cases:
            with self.subTest(fragments[0]):
                self.assert_rejected(self.generate(edit), 'invalid config.yaml', *fragments)

    def test_yaml_text_number_trap(self):
        r = self.generate(text_edit=lambda t: t.replace('w_force: 1.0e-06', 'w_force: 1e-6'))
        self.assert_rejected(r, "cost.w_force: is the text '1e-6'", 'write 1.0e-6')

    def test_typo_and_missing_keys(self):
        def typo(cfg):
            cfg['cost']['w_heigth'] = cfg['cost'].pop('w_height')
        self.assert_rejected(self.generate(typo), 'cost.w_heigth: unknown key', 'cost.w_height: missing')
        self.assert_rejected(self.generate(lambda c: c.pop('horizon')), "missing group 'horizon'")
        self.assert_rejected(self.generate(lambda c: c.update({'extras': {}})), "unknown group 'extras'")

    def test_all_errors_reported_at_once(self):
        r = self.generate(self.set(horizon={'N': -3, 'dt': 0.0}, robot={'inertia': [1.0, 2.0]}))
        self.assert_rejected(r, 'horizon.N:', 'horizon.dt:', 'robot.inertia:')

    def test_broken_yaml_and_layout(self):
        self.assert_rejected(self.generate(text_edit=lambda t: t.replace('    horizon:', '   horizon:  [')),
                             'not valid YAML')
        self.assert_rejected(self.generate(text_edit=lambda t: t.replace('ros__parameters', 'parameters')),
                             "expected the layout 'robodog_nmpc: ros__parameters: ...'")

    def test_missing_acados_environment(self):
        env = {k: v for k, v in os.environ.items() if k != 'ACADOS_SOURCE_DIR'}
        self.assert_rejected(self.generate(env=env), 'ACADOS_SOURCE_DIR is not set')

    # ---------------- the output folder can never delete anything else ----------------
    def test_export_dir_safety(self):
        protected = ['config/config.yaml', 'scripts/srbd_ocp.py', 'scripts/generate_solver.py',
                     'scripts/derivation/step7_casadi_model.py', 'test/test_generate_solver.py']
        for export_dir, fragment in [('.', 'must be a sub-folder inside the package'),
                                     ('..', 'must be a sub-folder inside the package'),
                                     ('scripts/../../outside', 'must be a sub-folder inside the package'),
                                     ('/tmp/robodog_x', 'must be relative to the package'),
                                     ('', 'must be a non-empty folder name'),
                                     ('config', 'was not created by this script'),
                                     ('scripts', 'was not created by this script')]:
            with self.subTest(export_dir=export_dir):
                self.assert_rejected(self.generate(self.set(codegen={'export_dir': export_dir})), fragment)
                for p in protected:
                    self.assertTrue(os.path.isfile(os.path.join(PKG, p)), f'{p} was deleted')
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(PKG), 'outside')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
