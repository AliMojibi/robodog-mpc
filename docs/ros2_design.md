# ROS 2 package design: `robodog_nmpc`

Status: **implemented.** This document was the design proposal; it now records the decisions (section 10) and
the design as built. Usage is described in the README; the mathematics in [formulation.md](formulation.md).

The package turns the NMPC developed in `robodog_nmpc/scripts/` into a reusable C++ library that other ROS 2
packages (e.g. a locomotion controller) can link against. It contains **no ROS node, no topics, no services**:
ROS 2 is used only as the **build and packaging system**. Each section first explains the ROS 2 / CMake
background needed ("Background"), then the design decision for this package.

---

## 1. ROS 2 as a build system

### Background: workspace, colcon, ament

A ROS 2 **workspace** is a folder with a `src/` subfolder that contains packages:

```
~/ros2_ws/
├── src/robodog-mpc/robodog_nmpc/   ← our package (any depth below src/ is fine)
├── build/     ← created by colcon: CMake build folders, one per package
├── install/   ← created by colcon: what other packages and programs use (headers, libraries, config)
└── log/
```

- **colcon** is the build tool: `colcon build` finds every folder below `src/` that contains a `package.xml`,
  sorts the packages by their dependencies, and builds each one with its build system.
- **ament_cmake** is that build system for C++ packages: ordinary CMake plus a few helper macros
  (`ament_package()`, `ament_export_targets()`, …) that make a package findable by other packages.
- `source install/setup.bash` makes the installed packages visible to the shell (paths, `find_package`).
  It must be sourced in every new terminal, after `source /opt/ros/humble/setup.bash`.

A package needs exactly two files to be a ROS 2 package: **`package.xml`** (what it is, what it needs) and
**`CMakeLists.txt`** (how to build and install it). The rest of the repo (`showcase/`, `examples/`) has no
`package.xml`, so colcon ignores it.

### Background: `package.xml`

The package's ID card, read by colcon and by `rosdep` (which can install missing dependencies):

```xml
<package format="3">
  <name>robodog_nmpc</name>                      <!-- must match the folder and project() name -->
  <version>0.1.0</version>
  <description>Nonlinear MPC (single-rigid-body model) for quadruped ground reaction forces</description>
  <maintainer email="...">...</maintainer>
  <license>BSD-3-Clause</license>

  <buildtool_depend>ament_cmake</buildtool_depend>   <!-- needed to build at all -->
  <depend>eigen</depend>                             <!-- needed to build AND by users of our headers -->
  <depend>yaml-cpp</depend>
  <test_depend>ament_cmake_gtest</test_depend>       <!-- only for `colcon test` -->

  <export><build_type>ament_cmake</build_type></export>
</package>
```

acados is not a ROS package, so it cannot be listed here; it is found through `ACADOS_SOURCE_DIR` (section 3).

---

## 2. Package layout

```
robodog_nmpc/
├── package.xml
├── CMakeLists.txt
├── config/config.yaml              all parameters, [codegen] / [runtime]       (exists)
├── scripts/                        Python: model, code generation, derivation  (exists)
│   ├── srbd_ocp.py, generate_solver.py
│   └── derivation/
├── solver/                         generated acados C code                     (generated)
├── include/robodog_nmpc/
│   ├── nmpc_solver.hpp             the class (public API)
│   ├── types.hpp                   input/output structs
│   └── config.hpp                  config struct + YAML loading/validation
├── src/
│   ├── nmpc_solver.cpp
│   └── config.cpp
└── test/
    ├── test_generate_solver.py     codegen workflow tests                     (exists)
    └── test_nmpc_solver.cpp        GoogleTest: the C++ class
```

Headers go into `include/<package_name>/` so users write `#include <robodog_nmpc/nmpc_solver.hpp>` — the
package name in the path avoids clashes with other packages' headers.

---

## 3. Build: CMake, acados and the generated code

### Background: what has to be linked

```
their controller ──links──► librobodog_nmpc.so ──contains──► generated solver C code (solver/*.c)
                                        └──links──► libacados.so, libhpipm.so, libblasfeo.so   (~/acados/lib)
```

The generated code in `solver/` is plain C. It calls the acados libraries, which are installed separately.

### Decision A: compile the generated C sources into our library

Two options:

| | (1) link the pre-built `solver/libacados_ocp_solver_srbd_nmpc.so` | **(2) compile `solver/*.c` into `librobodog_nmpc.so`** |
|---|---|---|
| build | needs the Makefile output of `generate_solver.py` | CMake compiles the C files itself |
| install | two of our libraries + acados | one library + acados |
| runtime paths | must find both `.so` files | simpler |

**Proposal: (2).** CMake collects the generated sources with a glob (`solver/*.c`, `solver/srbd_nmpc_*/*.c`,
excluding the example programs `main_*.c` and the stand-alone integrator `acados_sim_solver_*.c`, i.e. the same
set as the generator's own `ocp_shared_lib`) and compiles them together with our C++ files. The glob uses
`CONFIGURE_DEPENDS`, because the file set depends on the config (the pyramid model has no constraint functions).

### Decision B: finding acados

```cmake
if(NOT DEFINED ENV{ACADOS_SOURCE_DIR})
  message(FATAL_ERROR "ACADOS_SOURCE_DIR is not set (e.g. export ACADOS_SOURCE_DIR=$HOME/acados)")
endif()
set(ACADOS_DIR $ENV{ACADOS_SOURCE_DIR})
target_include_directories(robodog_nmpc PRIVATE ${ACADOS_DIR}/include ${ACADOS_DIR}/include/blasfeo/include
                                                 ${ACADOS_DIR}/include/hpipm/include ${CMAKE_CURRENT_SOURCE_DIR}/solver)
target_link_libraries(robodog_nmpc PUBLIC ${ACADOS_DIR}/lib/libacados.so ${ACADOS_DIR}/lib/libhpipm.so
                                          ${ACADOS_DIR}/lib/libblasfeo.so)
```

**RPATH:** the acados library folder is written into our library (`INSTALL_RPATH`), so at runtime it finds
`libacados.so` **without** `LD_LIBRARY_PATH`. This removes the most common "cannot open shared object file" error.
It must be the older **DT_RPATH** (`-Wl,--disable-new-dtags`): the acados libraries carry no search path of their
own, and the default DT_RUNPATH only covers our library's direct dependencies, so `libhpipm`, `libblasfeo` and
`libqpOASES_e` (needed by `libacados`) would not be found. DT_RPATH also covers them.

acados and the generated C headers are **PRIVATE**: users of our class only see our own headers (section 4),
never acados types. They therefore do not need acados include paths themselves.

### Decision C: the generated code must exist and match the config

CMake fails with a clear message if `solver/acados_solver_srbd_nmpc.h` is missing ("run
scripts/generate_solver.py first"). To catch a **stale** solver (config changed, solver not regenerated),
`generate_solver.py` additionally writes `solver/srbd_nmpc_codegen_config.h` with the `[codegen]` values it used,
as one list macro:

```c
#define SRBD_NMPC_CODEGEN_CONFIG(INT, DOUBLE, STRING) \
    INT("horizon", "N", 16) \
    DOUBLE("horizon", "dt", 0.025) \
    STRING("friction", "model", "polygon") \
    ...   /* all [codegen] keys except codegen.export_dir; friction.mu only for the pyramid model */
```

At `initialize()` the C++ class expands this macro, compares **every** entry literally with the YAML, and
refuses to start (`SolverMismatch`) listing all differing keys. The key list lives only in `generate_solver.py`
(`CODEGEN_KEYS`), so the C++ side needs no copy of it. Doubles are written with Python's `repr()` (exact round
trip), so `==` is safe. Dimensions (NX, NU, NP) are additionally checked at compile time with `static_assert`.
Not covered: changes to the model/cost code in `scripts/derivation/` without regenerating; the
regenerate-and-compare test of Decision D catches those.

### Decision D (open): commit `solver/` to git?

| | commit the generated C sources | keep `solver/` generated only (current) |
|---|---|---|
| user needs Python + acados_template to build | **no** — only the acados C library | yes |
| risk | generated code out of date | none |
| repo size | ~1 MB of generated C | — |

**Proposal: commit the generated `.c`/`.h` files** (not `.o`, `.so`, json). Building the C++ package then needs
only the acados C library, which is what a user integrating the solver wants. The stale-code risk is covered by
Decision C plus a test that regenerates the code and checks it is identical (the generation is deterministic:
verified, same config → same checksum).

### Background: install and export (how other packages find us)

```cmake
install(DIRECTORY include/ DESTINATION include)                 # headers  -> install/robodog_nmpc/include
install(TARGETS robodog_nmpc EXPORT robodog_nmpcTargets         # library  -> install/robodog_nmpc/lib
        LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin INCLUDES DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})     # config.yaml -> share/robodog_nmpc/config
ament_export_targets(robodog_nmpcTargets HAS_LIBRARY_TARGET)    # makes the target importable
ament_export_dependencies(Eigen3 yaml-cpp)
ament_package()
```

Another package then only needs:

```cmake
find_package(robodog_nmpc REQUIRED)
target_link_libraries(their_controller robodog_nmpc::robodog_nmpc)
```

The installed config can be located at runtime with
`ament_index_cpp::get_package_share_directory("robodog_nmpc") + "/config/config.yaml"`.

---

## 4. Class API

Namespace `robodog_nmpc`. Fixed-size Eigen types (no heap allocation, sizes checked by the compiler).
Foot order everywhere: **FR, FL, RR, RL**.

### Input / output types (`types.hpp`)

```cpp
struct State {                       // measured body state, world frame
  Eigen::Vector3d rpy;               // roll, pitch, yaw [rad] (Z-Y-X Euler)
  Eigen::Vector3d position;          // CoM position [m]
  Eigen::Vector3d angular_velocity;  // world frame [rad/s]
  Eigen::Vector3d linear_velocity;   // world frame [m/s]
};

struct Command {                     // operator command, HEADING frame (forward / left of the robot)
  double vx{0.0}, vy{0.0}, yaw_rate{0.0};
};

using Contact = std::array<bool, 4>;                  // true = stance
using FootPositions = std::array<Eigen::Vector3d, 4>; // world frame [m]

struct Solution {
  std::array<Eigen::Vector3d, 4> forces;   // ground reaction forces to apply now (u_0), WORLD frame [N]
  Status status;                           // see section 6
  int acados_status;                       // 0 = success
  int qp_iterations;                       // == qp_solver_iter_max: the QP stopped at its limit (still "success")
  double solve_time_ms;
  std::vector<State> predicted_states;     // last successful plan, x_0..x_N (sized in initialize())
  std::vector<Forces> predicted_forces;    // u_0..u_{N-1}, swing feet exactly 0
  int plan_age;                            // steps since that plan was computed (0 after Ok, -1 = none)
};
```

### The class (`nmpc_solver.hpp`)

```cpp
class NmpcSolver {
public:
  // Load + validate config.yaml, check it against the generated solver, create the acados solver.
  Status initialize(const std::string& config_path);

  // (a) task-literal form: contact flags and foot positions held over the whole horizon
  //     (no touchdown is ever predicted, so no foothold heuristic is needed).
  Status run(const State& state, const Command& cmd, const Contact& contact, const FootPositions& feet);

  // (b) contact schedule (N entries) + current foot positions only: footholds of feet that touch down
  //     later in the horizon are predicted with the Raibert heuristic (Eq. 33): hip on the commanded path
  //     at touchdown + v_cmd T_stance / 2, T_stance counted from the schedule, height of the lowest foot now.
  Status run(const State& state, const Command& cmd,
             const std::vector<Contact>& contact_schedule,          // N entries
             const FootPositions& feet);                            // current positions

  // (c) full form: a contact schedule and planned foot positions for every horizon node.
  Status run(const State& state, const Command& cmd,
             const std::vector<Contact>& contact_schedule,          // N entries
             const std::vector<FootPositions>& foot_plan);          // N + 1 entries

  const Solution& getSolution() const;     // result of the last run()
  int horizonLength() const;               // N of the generated solver
};
```

**What `run()` does** (one control step):
1. check inputs (section 6), convert the state to the solver's `x_0`;
2. per stage: parameters `p = [foot positions, m, Ixx, Iyy, Izz, μ]`, force bounds from the contact
   (swing: f = 0), cost reference from the command;
3. warm start: shift the previous solution one step forward;
4. one SQP-RTI iteration (`srbd_nmpc_acados_solve`);
5. store `u_0` as the four forces and the statistics in `Solution`.

**Real-time rules:** no memory allocation, no file access and no exceptions inside `run()`; all buffers are
allocated in `initialize()`. One instance is **not thread-safe**; use one instance per control thread.

---

## 5. Configuration in C++

`config.hpp` defines a plain struct mirroring `config.yaml`. `loadConfig(path)` reads it with **yaml-cpp** and
applies **the same validation rules** as `validate_config()` in `scripts/srbd_ocp.py` (every key present, no
unknown keys, types, ranges, list lengths, f_min < f_max), reporting all problems at once.

How each setting reaches the solver:

| Setting | How it is applied |
|---|---|
| `[runtime]` mass, inertia, μ | written into the parameter vector `p` of every stage |
| `[runtime]` f_min, f_max | stance-foot bounds (`lbu`/`ubu`) in every `run()` |
| `[runtime]` cost weights | `W` of every stage and `W_e`, set once in `initialize()` |
| `[runtime]` nominal height, hip offsets | cost reference; foothold fallback |
| `[codegen]` N, dt, friction model, solver type | **compared** with `srbd_nmpc_codegen_config.h`; mismatch → `initialize()` fails with "regenerate the solver" |

---

## 6. Errors and runtime checks

`Status` is an enum plus a message: `Ok`, `NotInitialized`, `InvalidConfig`, `SolverMismatch`,
`InvalidInput`, `SolverFailed`.

- `initialize()`: any config or mismatch problem → error status with the full message; the object stays unusable.
- `run()` input checks (cheap, a few µs against ~1.5 ms per solve): all numbers finite, contact schedule /
  foot plan of the right length. Configurable (`check_inputs: true` in the YAML) so they can be switched off.
- **Solver failure** (acados status ≠ 0): `run()` returns `SolverFailed`; `getSolution()` then holds the
  **next step of the previous plan** (shifted warm start), so the caller always gets physically consistent forces
  (inside the friction and contact limits of the previous solution) and can decide what to do.

---

## 7. Tests (`colcon test`)

GoogleTest program `test_nmpc_solver.cpp` (registered with `ament_add_gtest`), never part of the runtime path:

| Test | Checks |
|---|---|
| standing | 4 stance feet, zero command → each f_z ≈ m g / 4, f_x = f_y ≈ 0 |
| trot / turn | swing feet exactly 0; friction ≤ μ; f_min ≤ f_z ≤ f_max; same forces as the Python solver for identical inputs (reference values exported by a script) |
| config errors | invalid YAML values, N mismatch, missing generated code → clear error status |
| input errors | NaN in the state, wrong schedule length → `InvalidInput`, no solve |
| timing | median / max solve time over many calls (reported, generous limit) |

The existing Python workflow tests stay in `test/test_generate_solver.py`.

---

## 8. Dependencies (all available on this machine)

| Dependency | Version | Source |
|---|---|---|
| ROS 2 + ament_cmake | Humble | `/opt/ros/humble` |
| Eigen | 3.4.0 | `libeigen3-dev` |
| yaml-cpp | 0.7.0 | `libyaml-cpp-dev` |
| GoogleTest | via `ament_cmake_gtest` | `ros-humble-ament-cmake-gtest` |
| acados | v0.6.0 | built from source, `ACADOS_SOURCE_DIR` |

## 9. Build and use (target workflow)

```bash
# once: generate the solver (only after changing a [codegen] setting)
cd ~/ros2_ws/src/robodog-mpc/robodog_nmpc && python scripts/generate_solver.py

# build and test
source /opt/ros/humble/setup.bash
export ACADOS_SOURCE_DIR=$HOME/acados
cd ~/ros2_ws && colcon build --packages-select robodog_nmpc
colcon test --packages-select robodog_nmpc && colcon test-result --verbose
```

```cpp
#include <robodog_nmpc/nmpc_solver.hpp>

robodog_nmpc::NmpcSolver mpc;
if (mpc.initialize(config_path) != robodog_nmpc::Status::Ok) { /* report and stop */ }

// every control cycle (e.g. 40 Hz):
mpc.run(state, command, contact, feet);
const auto& sol = mpc.getSolution();   // sol.forces[0..3]: FR, FL, RR, RL, world frame
```

---

## 10. Decisions for review

| # | Decision | Proposal |
|---|---|---|
| A | generated code: separate `.so` or compiled into our library | **compile into our library** |
| B | finding acados | `ACADOS_SOURCE_DIR` + RPATH (no `LD_LIBRARY_PATH` needed at runtime) |
| C | stale solver protection | **decided:** `srbd_nmpc_codegen_config.h` (list macro), every `[codegen]` key compared literally at `initialize()` |
| D | commit `solver/` | **yes, `.c`/`.h` only**, plus a regenerate-and-compare test |
| E | vector types | **decided:** fixed-size Eigen (`Eigen::Vector3d`) |
| F | error handling | **decided:** `Status` return values + `errorMessage()`, no exceptions out of the class |
| G | solver failure | **decided:** return `SolverFailed` + previous plan's next step (zero forces if there is none) |
| H | where the workspace lives | e.g. `~/ros2_ws/src/robodog-mpc` (a clone or a symlink to this repo) |
