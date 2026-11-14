cmake_minimum_required(VERSION 3.8)
project(robodog_nmpc LANGUAGES C CXX)   # C: the generated acados solver code

# colcon passes no build type by default, which means no optimization (-O0): far too slow for the solver.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default Release)" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

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

# Sources: the same set as the generator's own solver library (Makefile target ocp_shared_lib), i.e. without
# the example programs main_*.c and the stand-alone integrator acados_sim_solver_*.c. Globbed, because the
# file set depends on the config (e.g. no constraint functions for the pyramid model); CONFIGURE_DEPENDS
# re-collects the list on the next build after scripts/generate_solver.py has run.
file(GLOB SOLVER_SOURCES CONFIGURE_DEPENDS "${SOLVER_DIR}/*.c" "${SOLVER_DIR}/${SOLVER_NAME}_*/*.c")
list(FILTER SOLVER_SOURCES EXCLUDE REGEX "/(main_|acados_sim_solver_)[^/]*\\.c$")

# ---- the library ----
add_library(${PROJECT_NAME} SHARED
  src/config.cpp
  src/codegen_check.cpp
  src/nmpc_solver.cpp
  ${SOLVER_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  C_STANDARD 99                          # as in the generated Makefile
  INSTALL_RPATH "${ACADOS_DIR}/lib")     # find the acados libraries at runtime without LD_LIBRARY_PATH
# The acados libraries have no search path of their own, and libacados.so needs libhpipm, libblasfeo and
# libqpOASES_e. The default (new) DT_RUNPATH only covers our library's direct dependencies, so those would
# not be found. The older DT_RPATH is also searched for the dependencies of the libraries we load.
# (DT_RPATH takes priority over LD_LIBRARY_PATH: the acados the code was generated for is always used.)
target_link_options(${PROJECT_NAME} PRIVATE "LINKER:--disable-new-dtags")
# Link time of programs using this library (our tests, other packages): the linker also resolves the
# dependencies of libacados.so, which it cannot find without LD_LIBRARY_PATH. INTERFACE: exported with the target.
target_link_options(${PROJECT_NAME} INTERFACE "LINKER:-rpath-link,${ACADOS_DIR}/lib")
target_compile_options(${PROJECT_NAME} PRIVATE    # warnings for our C++ code only, not the generated C code
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
# Our headers are public and use Eigen (PUBLIC); acados and yaml-cpp are PRIVATE: users never see their types.
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME} PRIVATE
  "${SOLVER_DIR}"
  "${ACADOS_DIR}/include"
  "${ACADOS_DIR}/include/acados"
  "${ACADOS_DIR}/include/blasfeo/include"
  "${ACADOS_DIR}/include/hpipm/include")
target_link_libraries(${PROJECT_NAME} PRIVATE
  "${ACADOS_DIR}/lib/libacados.so" "${ACADOS_DIR}/lib/libhpipm.so" "${ACADOS_DIR}/lib/libblasfeo.so" m
  yaml-cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

# ---- install ----
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)                # -> #include <robodog_nmpc/config.hpp>
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})   # -> install/robodog_nmpc/share/robodog_nmpc/config

# ---- tests (colcon test) ----
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_config test/test_config.cpp)
  ament_add_gtest(test_nmpc_solver test/test_nmpc_solver.cpp)
  foreach(test test_config test_nmpc_solver)
    target_link_libraries(${test} ${PROJECT_NAME})
    target_compile_definitions(${test} PRIVATE CONFIG_YAML="${CMAKE_CURRENT_SOURCE_DIR}/config/config.yaml")
  endforeach()
endif()

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)   # other packages: robodog_nmpc::robodog_nmpc
ament_export_dependencies(Eigen3)                                 # our public headers include Eigen
ament_package()
