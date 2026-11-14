cmake_minimum_required(VERSION 3.8)
project(robodog_nmpc LANGUAGES C CXX)   # C: the generated acados solver code

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# ---- acados (not a ROS package): found through the environment variable ACADOS_SOURCE_DIR ----
# Checked when CMake configures. After changing the environment, reconfigure:
#   colcon build --packages-select robodog_nmpc --cmake-clean-cache
set(ACADOS_DIR "$ENV{ACADOS_SOURCE_DIR}")
if(ACADOS_DIR STREQUAL "")
  message(FATAL_ERROR "ACADOS_SOURCE_DIR is not set (e.g. export ACADOS_SOURCE_DIR=$HOME/acados)")
endif()
if(NOT IS_ABSOLUTE "${ACADOS_DIR}")
  message(FATAL_ERROR "ACADOS_SOURCE_DIR must be an absolute path, got '${ACADOS_DIR}'")
endif()
foreach(lib acados hpipm blasfeo)
  if(NOT EXISTS "${ACADOS_DIR}/lib/lib${lib}.so")
    message(FATAL_ERROR "acados library not found: ${ACADOS_DIR}/lib/lib${lib}.so "
                        "(is acados built and installed in ACADOS_SOURCE_DIR?)")
  endif()
endforeach()

# ---- generated solver code (scripts/generate_solver.py) ----
# Must match codegen.solver_name in config/config.yaml.
set(SOLVER_NAME srbd_nmpc)
set(SOLVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/solver")
foreach(header acados_solver_${SOLVER_NAME}.h ${SOLVER_NAME}_codegen_config.h)
  if(NOT EXISTS "${SOLVER_DIR}/${header}")
    message(FATAL_ERROR "generated solver code not found: ${SOLVER_DIR}/${header}\n"
                        "Run scripts/generate_solver.py first (see README).")
  endif()
endforeach()

# ---- install ----
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})   # -> install/robodog_nmpc/share/robodog_nmpc/config

ament_package()
