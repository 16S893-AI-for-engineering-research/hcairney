#include "burgers/Config.h"
#include "burgers/ConvectiveFlux.h"
#include "burgers/Grid.h"
#include "burgers/Reconstruction.h"
#include "burgers/State.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if(!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expectNear(double actual,
                double expected,
                double tolerance,
                const std::string& message) {
  if(std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " (expected " << expected
              << ", got " << actual << ", tolerance " << tolerance
              << ")\n";
    ++failures;
  }
}

template<typename Exception, typename Function>
void expectThrows(Function function, const std::string& message) {
  try {
    function();
  } catch(const Exception&) {
    return;
  } catch(const std::exception& error) {
    std::cerr << "FAIL: " << message << " (wrong exception: "
              << error.what() << ")\n";
    ++failures;
    return;
  }
  std::cerr << "FAIL: " << message << " (no exception)\n";
  ++failures;
}

const std::vector<burgers::Limiter>& musclLimiters() {
  static const std::vector<burgers::Limiter> limiters{
    burgers::Limiter::Minmod,
    burgers::Limiter::MonotonizedCentral,
    burgers::Limiter::VanLeer};
  return limiters;
}

void checkLimiterValues() {
  expectNear(burgers::musclLimitedSlope(
               1.0, 2.0, burgers::Limiter::Minmod),
             1.0,
             0.0,
             "Minmod must select the smaller same-sign difference");
  expectNear(burgers::musclLimitedSlope(
               1.0, 2.0, burgers::Limiter::MonotonizedCentral),
             1.5,
             0.0,
             "MC must retain an admissible centered difference");
  expectNear(burgers::musclLimitedSlope(
               1.0, 2.0, burgers::Limiter::VanLeer),
             4.0 / 3.0,
             2.0e-15,
             "Van Leer must use the harmonic same-sign difference");

  for(const burgers::Limiter limiter : musclLimiters()) {
    expectNear(burgers::musclLimitedSlope(1.0, -0.25, limiter),
               0.0,
               0.0,
               "a limiter must zero a slope across an extremum");
    expectNear(burgers::musclLimitedSlope(-2.0, -1.0, limiter),
               -burgers::musclLimitedSlope(2.0, 1.0, limiter),
               0.0,
               "a limiter must be sign symmetric");
  }

  expectThrows<std::invalid_argument>(
    []() {
      static_cast<void>(burgers::musclLimitedSlope(
        1.0, 1.0, burgers::Limiter::None));
    },
    "MUSCL must reject limiter none");
  expectThrows<std::invalid_argument>(
    []() {
      static_cast<void>(burgers::musclLimitedSlope(
        std::numeric_limits<double>::quiet_NaN(),
        1.0,
        burgers::Limiter::Minmod));
    },
    "a limiter must reject non-finite input");
}

void checkMusclReconstruction() {
  const burgers::Grid grid(0.0, 6.0, 6);
  const burgers::State constant(grid, -1.25);
  const burgers::State varying(
    burgers::State::Container{0.0, 1.0, 2.0, 1.5, -0.5, -1.0});

  for(const burgers::Limiter limiter : musclLimiters()) {
    burgers::FaceStates constant_faces(grid.cellCount());
    burgers::reconstructMuscl(grid, constant, limiter, constant_faces);
    for(std::size_t face = 0; face < grid.cellCount(); ++face) {
      expectNear(constant_faces.left[face],
                 -1.25,
                 0.0,
                 "MUSCL must exactly preserve a constant left state");
      expectNear(constant_faces.right[face],
                 -1.25,
                 0.0,
                 "MUSCL must exactly preserve a constant right state");
    }

    std::vector<double> slopes(grid.cellCount(), 0.0);
    burgers::computeMusclSlopes(grid, varying, limiter, slopes);
    burgers::FaceStates faces(grid.cellCount());
    burgers::reconstructMuscl(grid, varying, limiter, faces);
    for(std::size_t face = 0; face < grid.cellCount(); ++face) {
      const std::size_t right = grid.neighbor(face, 1);
      const double lower = std::min(varying[face], varying[right]);
      const double upper = std::max(varying[face], varying[right]);
      expect(faces.left[face] >= lower - 2.0e-15 &&
               faces.left[face] <= upper + 2.0e-15,
             "limited MUSCL left state must remain within adjacent data");
      expect(faces.right[face] >= lower - 2.0e-15 &&
               faces.right[face] <= upper + 2.0e-15,
             "limited MUSCL right state must remain within adjacent data");
    }
    expectNear(faces.left.back(),
               varying[varying.size() - 1] +
                 0.5 * slopes[varying.size() - 1],
               0.0,
               "MUSCL left reconstruction must wrap at the periodic face");
    expectNear(faces.right.back(),
               varying[0] - 0.5 * slopes[0],
               0.0,
               "MUSCL right reconstruction must wrap at the periodic face");
  }

  std::vector<double> wrong_slopes(grid.cellCount() - 1, 0.0);
  expectThrows<std::invalid_argument>(
    [&grid, &varying, &wrong_slopes]() {
      burgers::computeMusclSlopes(
        grid, varying, burgers::Limiter::Minmod, wrong_slopes);
    },
    "MUSCL must reject a mismatched slope array");
}

void checkRusanovFlux() {
  for(const double state : {-3.0, -0.5, 0.0, 1.25, 4.0}) {
    expectNear(burgers::rusanovFlux(state, state),
               burgers::burgersPhysicalFlux(state),
               2.0e-15,
               "Rusanov flux must be consistent");
  }
  expectNear(burgers::rusanovFlux(2.0, 0.0),
             3.0,
             0.0,
             "Rusanov must use the maximum local Burgers wave speed");
  expectNear(burgers::rusanovFlux(-1.0, 2.0),
             -1.75,
             0.0,
             "Rusanov must diffuse a transonic jump");

  burgers::FaceStates faces(2);
  faces.left[0] = 2.0;
  faces.right[0] = 0.0;
  faces.left[1] = -1.0;
  faces.right[1] = 2.0;
  std::vector<double> fluxes(2, 0.0);
  burgers::computeConvectiveFluxes(
    faces, burgers::ConvectiveFlux::Rusanov, fluxes);
  expectNear(fluxes[0], 3.0, 0.0,
             "Rusanov array dispatch must evaluate each face");
  expectNear(fluxes[1], -1.75, 0.0,
             "Rusanov array dispatch must preserve face ordering");

  expectThrows<std::invalid_argument>(
    []() {
      static_cast<void>(burgers::rusanovFlux(
        0.0, std::numeric_limits<double>::infinity()));
    },
    "Rusanov must reject a non-finite state");
  std::vector<double> wrong_fluxes(1, 0.0);
  expectThrows<std::invalid_argument>(
    [&faces, &wrong_fluxes]() {
      burgers::computeRusanovFluxes(faces, wrong_fluxes);
    },
    "Rusanov must reject a mismatched flux array");
}

void checkConfigurationContracts() {
  burgers::RunConfig muscl = burgers::makeDefaultRunConfig();
  muscl.numerical_method.reconstruction = burgers::Reconstruction::Muscl;
  const std::vector<std::string> missing_limiter = burgers::validate(muscl);
  expect(!missing_limiter.empty(),
         "MUSCL configuration must require an explicit limiter");

  burgers::RunConfig piecewise = burgers::makeDefaultRunConfig();
  piecewise.numerical_method.limiter = burgers::Limiter::Minmod;
  const std::vector<std::string> extraneous_limiter =
    burgers::validate(piecewise);
  expect(!extraneous_limiter.empty(),
         "piecewise reconstruction must reject an extraneous limiter");

  for(const burgers::Limiter limiter : musclLimiters()) {
    muscl.numerical_method.limiter = limiter;
    expect(burgers::validate(muscl).empty(),
           "every exposed MUSCL limiter must be a valid configuration");
  }
}

}  // namespace

int main() {
  checkLimiterValues();
  checkMusclReconstruction();
  checkRusanovFlux();
  checkConfigurationContracts();

  if(failures != 0) {
    std::cerr << failures << " Phase 4 kernel check(s) failed\n";
    return 1;
  }
  std::cout << "Phase 4 kernel checks passed\n";
  return 0;
}
