cmake_minimum_required(VERSION 3.10)

project(burgers_sgs VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(CheckCXXCompilerFlag)

option(BURGERS_ENABLE_SANITIZERS
  "Enable supported sanitizers for Debug builds"
  OFF)

set(BURGERS_GIT_REVISION "unknown")
set(BURGERS_SOURCE_DIRTY "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE BURGERS_GIT_REVISION_RESULT
    OUTPUT_VARIABLE BURGERS_GIT_REVISION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(BURGERS_GIT_REVISION_RESULT EQUAL 0)
    set(BURGERS_GIT_REVISION "${BURGERS_GIT_REVISION_OUTPUT}")

    execute_process(
      COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE BURGERS_GIT_STATUS_RESULT
      OUTPUT_VARIABLE BURGERS_GIT_STATUS_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(BURGERS_GIT_STATUS_RESULT EQUAL 0)
      if(BURGERS_GIT_STATUS_OUTPUT STREQUAL "")
        set(BURGERS_SOURCE_DIRTY "false")
      else()
        set(BURGERS_SOURCE_DIRTY "true")
      endif()
    endif()
  endif()
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildInfo.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/burgers/BuildInfo.h"
  @ONLY)

set(BURGERS_WARNING_FLAGS)
if(MSVC)
  set(BURGERS_WARNING_FLAGS /W4)
else()
  set(BURGERS_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
endif()

set(BURGERS_USE_SANITIZERS OFF)
if(BURGERS_ENABLE_SANITIZERS AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(BURGERS_SANITIZER_FLAG "-fsanitize=address,undefined")

  set(BURGERS_SAVED_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")
  list(APPEND CMAKE_REQUIRED_LIBRARIES "${BURGERS_SANITIZER_FLAG}")
  
  check_cxx_compiler_flag(
    "${BURGERS_SANITIZER_FLAG}"
    BURGERS_SANITIZER_COMPILE_AND_LINK_AVAILABLE)
  
  set(CMAKE_REQUIRED_LIBRARIES "${BURGERS_SAVED_REQUIRED_LIBRARIES}")

  if(BURGERS_SANITIZER_COMPILE_AND_LINK_AVAILABLE)
    set(BURGERS_USE_SANITIZERS ON)
  endif()
endif()

function(burgers_configure_target target_name)
  target_compile_features(${target_name} PUBLIC cxx_std_14)
  target_compile_options(${target_name} PRIVATE ${BURGERS_WARNING_FLAGS})

  if(MSVC)
    target_compile_options(${target_name} PRIVATE $<$<CONFIG:Debug>:/UNDEBUG>)
  else()
    target_compile_options(${target_name} PRIVATE $<$<CONFIG:Debug>:-UNDEBUG>)
  endif()

  if(BURGERS_USE_SANITIZERS)
    target_compile_options(${target_name} PRIVATE
      $<$<CONFIG:Debug>:-fsanitize=address,undefined>
      $<$<CONFIG:Debug>:-fno-omit-frame-pointer>)
    target_link_libraries(${target_name} PRIVATE
      $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_compile_definitions(${target_name} PRIVATE
      $<$<CONFIG:Debug>:BURGERS_SANITIZERS_ENABLED=1>)
  endif()
endfunction()

add_library(burgers_core STATIC
  src/BurgersSolver.cpp
  src/ClosureModel.cpp
  src/Config.cpp
  src/ConfigIO.cpp
  src/ConvectiveFlux.cpp
  src/Diagnostics.cpp
  src/Forcing.cpp
  src/Grid.cpp
  src/InitialCondition.cpp
  src/ManufacturedSolution.cpp
  src/Reconstruction.cpp
  src/State.cpp
  src/RunMetadata.cpp
  src/ViscousFlux.cpp)
add_library(burgers::core ALIAS burgers_core)

target_include_directories(burgers_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../source/extern")
target_compile_definitions(burgers_core PRIVATE
  BURGERS_BUILD_CONFIGURATION=\"$<CONFIG>\")
burgers_configure_target(burgers_core)

add_executable(run_solver tools/run_solver.cpp)
target_link_libraries(run_solver PRIVATE burgers::core)
burgers_configure_target(run_solver)

if(BUILD_TESTING)
  add_executable(test_config_io tests/test_config_io.cpp)
  target_link_libraries(test_config_io PRIVATE burgers::core)
  burgers_configure_target(test_config_io)

  add_test(NAME burgers.config_io COMMAND test_config_io)
  set_tests_properties(burgers.config_io PROPERTIES
    LABELS "configuration;verification")

  add_executable(test_phase0_scaffolding tests/test_phase0_scaffolding.cpp)
  target_link_libraries(test_phase0_scaffolding PRIVATE burgers::core)
  burgers_configure_target(test_phase0_scaffolding)

  add_test(NAME burgers.phase0.scaffolding COMMAND test_phase0_scaffolding)

  add_executable(test_phase1_grid_state tests/test_phase1_grid_state.cpp)
  target_link_libraries(test_phase1_grid_state PRIVATE burgers::core)
  burgers_configure_target(test_phase1_grid_state)

  add_test(NAME burgers.phase1.grid_state COMMAND test_phase1_grid_state)

  add_executable(test_phase2_fluxes tests/test_phase2_fluxes.cpp)
  target_link_libraries(test_phase2_fluxes PRIVATE burgers::core)
  burgers_configure_target(test_phase2_fluxes)

  add_test(NAME burgers.phase2.fluxes COMMAND test_phase2_fluxes)

  add_executable(test_phase2_solver tests/test_phase2_solver.cpp)
  target_link_libraries(test_phase2_solver PRIVATE burgers::core)
  burgers_configure_target(test_phase2_solver)

  add_test(NAME burgers.phase2.solver COMMAND test_phase2_solver)

  add_executable(test_phase3_manufactured
    tests/test_phase3_manufactured.cpp)
  target_link_libraries(test_phase3_manufactured PRIVATE burgers::core)
  burgers_configure_target(test_phase3_manufactured)

  add_test(NAME burgers.phase3.manufactured
    COMMAND test_phase3_manufactured)
  set_tests_properties(burgers.phase3.manufactured PROPERTIES
    LABELS "phase3;verification")

  add_executable(test_phase3_invariants tests/test_phase3_invariants.cpp)
  target_link_libraries(test_phase3_invariants PRIVATE burgers::core)
  burgers_configure_target(test_phase3_invariants)

  add_test(NAME burgers.phase3.invariants COMMAND test_phase3_invariants)
  set_tests_properties(burgers.phase3.invariants PROPERTIES
    LABELS "phase3;verification")

  add_executable(test_phase3_waves tests/test_phase3_waves.cpp)
  target_link_libraries(test_phase3_waves PRIVATE burgers::core)
  burgers_configure_target(test_phase3_waves)

  add_test(NAME burgers.phase3.waves COMMAND test_phase3_waves)
  set_tests_properties(burgers.phase3.waves PROPERTIES
    LABELS "phase3;verification")

  add_executable(test_phase4_kernels tests/test_phase4_kernels.cpp)
  target_link_libraries(test_phase4_kernels PRIVATE burgers::core)
  burgers_configure_target(test_phase4_kernels)

  add_test(NAME burgers.phase4.kernels COMMAND test_phase4_kernels)
  set_tests_properties(burgers.phase4.kernels PROPERTIES
    LABELS "phase4;verification")

  add_executable(test_phase4_matrix tests/test_phase4_matrix.cpp)
  target_link_libraries(test_phase4_matrix PRIVATE burgers::core)
  burgers_configure_target(test_phase4_matrix)

  add_test(NAME burgers.phase4.matrix COMMAND test_phase4_matrix)
  set_tests_properties(burgers.phase4.matrix PROPERTIES
    LABELS "phase4;verification")

  add_executable(test_phase4_waves_dissipation
    tests/test_phase4_waves_dissipation.cpp)
  target_link_libraries(test_phase4_waves_dissipation PRIVATE burgers::core)
  burgers_configure_target(test_phase4_waves_dissipation)

  add_test(NAME burgers.phase4.waves_dissipation
    COMMAND test_phase4_waves_dissipation)
  set_tests_properties(burgers.phase4.waves_dissipation PROPERTIES
    LABELS "phase4;verification")

  add_executable(test_phase5_forcing tests/test_phase5_forcing.cpp)
  target_link_libraries(test_phase5_forcing PRIVATE burgers::core)
  burgers_configure_target(test_phase5_forcing)

  add_test(NAME burgers.phase5.forcing COMMAND test_phase5_forcing)
  set_tests_properties(burgers.phase5.forcing PROPERTIES
    LABELS "phase5;verification")

  add_executable(test_phase5_solver tests/test_phase5_solver.cpp)
  target_link_libraries(test_phase5_solver PRIVATE burgers::core)
  burgers_configure_target(test_phase5_solver)

  add_test(NAME burgers.phase5.solver COMMAND test_phase5_solver)
  set_tests_properties(burgers.phase5.solver PROPERTIES
    LABELS "phase5;verification")

  add_executable(test_phase7_closures tests/test_phase7_closures.cpp)
  target_link_libraries(test_phase7_closures PRIVATE burgers::core)
  burgers_configure_target(test_phase7_closures)

  add_test(NAME burgers.phase7.closures COMMAND test_phase7_closures)
  set_tests_properties(burgers.phase7.closures PROPERTIES
    LABELS "phase7;verification")

  add_executable(test_phase8_prescribed_forcing
    tests/test_phase8_prescribed_forcing.cpp)
  target_link_libraries(test_phase8_prescribed_forcing PRIVATE burgers::core)
  burgers_configure_target(test_phase8_prescribed_forcing)

  add_test(NAME burgers.phase8.prescribed_forcing
    COMMAND test_phase8_prescribed_forcing)
  set_tests_properties(burgers.phase8.prescribed_forcing PROPERTIES
    LABELS "phase8;verification")
endif()
