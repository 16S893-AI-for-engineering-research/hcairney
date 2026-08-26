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
  src/Config.cpp
  src/RunMetadata.cpp)
add_library(burgers::core ALIAS burgers_core)

target_include_directories(burgers_core
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(burgers_core PRIVATE
  BURGERS_BUILD_CONFIGURATION=\"$<CONFIG>\")
burgers_configure_target(burgers_core)

add_executable(run_solver tools/run_solver.cpp)
target_link_libraries(run_solver PRIVATE burgers::core)
burgers_configure_target(run_solver)

if(BUILD_TESTING)
  add_executable(test_phase0_scaffolding tests/test_phase0_scaffolding.cpp)
  target_link_libraries(test_phase0_scaffolding PRIVATE burgers::core)
  burgers_configure_target(test_phase0_scaffolding)

  add_test(NAME burgers.phase0.scaffolding COMMAND test_phase0_scaffolding)
endif()
