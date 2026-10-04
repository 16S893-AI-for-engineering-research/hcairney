#include "burgers/ConvectiveFlux.h"
#include "burgers/Grid.h"
#include "burgers/Reconstruction.h"
#include "burgers/State.h"
#include "burgers/ViscousFlux.h"

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
              << ", got " << actual << ")\n";
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

}  // namespace

int main() {
  const double tolerance = 1.0e-14;

  expectNear(burgers::godunovFlux(2.0, 1.0), 2.0, tolerance,
             "a right-going shock must select the left flux");
  expectNear(burgers::godunovFlux(-1.0, -2.0), 2.0, tolerance,
             "a left-going shock must select the right flux");
  expectNear(burgers::godunovFlux(1.0, -1.0), 0.5, tolerance,
             "a stationary shock must have its physical shock flux");
  expectNear(burgers::godunovFlux(1.0, 2.0), 0.5, tolerance,
             "a right-going rarefaction must select the left flux");
  expectNear(burgers::godunovFlux(-2.0, -1.0), 0.5, tolerance,
             "a left-going rarefaction must select the right flux");
  expectNear(burgers::godunovFlux(-1.0, 2.0), 0.0, tolerance,
             "a transonic rarefaction must use the zero flux");
  expectNear(burgers::godunovFlux(-3.0, -3.0), 4.5, tolerance,
             "equal states must recover the physical flux");

  const burgers::Grid grid(0.0, 3.0, 3);
  const burgers::State state(burgers::State::Container{1.0, 2.0, 3.0});
  burgers::FaceStates face_states(grid.cellCount());
  burgers::reconstructPiecewiseConstant(grid, state, face_states);
  expectNear(face_states.left[0], 1.0, 0.0,
             "face zero left state must come from cell zero");
  expectNear(face_states.right[0], 2.0, 0.0,
             "face zero right state must come from cell one");
  expectNear(face_states.left[2], 3.0, 0.0,
             "last face left state must come from the last cell");
  expectNear(face_states.right[2], 1.0, 0.0,
             "last face right state must wrap periodically");

  std::vector<double> convective_fluxes(grid.cellCount(), 0.0);
  burgers::computeGodunovFluxes(face_states, convective_fluxes);
  expectNear(convective_fluxes[0], 0.5, tolerance,
             "Godunov face flux must use reconstructed states");
  expectNear(convective_fluxes[1], 2.0, tolerance,
             "Godunov face flux must be evaluated independently per face");
  expectNear(convective_fluxes[2], 4.5, tolerance,
             "periodic shock flux must use the final and first cells");

  std::vector<double> viscous_fluxes(grid.cellCount(), 0.0);
  burgers::computeMolecularViscousFluxes(
    grid, state, 0.5, viscous_fluxes);
  expectNear(viscous_fluxes[0], 0.5, tolerance,
             "centered viscous flux must use the face gradient");
  expectNear(viscous_fluxes[1], 0.5, tolerance,
             "centered viscous flux must be consistent on uniform slopes");
  expectNear(viscous_fluxes[2], -1.0, tolerance,
             "centered viscous flux must wrap periodically");

  const burgers::State constant(grid, -2.0);
  burgers::reconstructPiecewiseConstant(grid, constant, face_states);
  for(std::size_t face = 0; face < face_states.size(); ++face) {
    expectNear(face_states.left[face], -2.0, 0.0,
               "constant reconstruction must preserve every left state");
    expectNear(face_states.right[face], -2.0, 0.0,
               "constant reconstruction must preserve every right state");
  }

  expectThrows<std::invalid_argument>(
    []() {
      static_cast<void>(burgers::godunovFlux(
        std::numeric_limits<double>::quiet_NaN(), 1.0));
    },
    "Godunov flux must reject non-finite input");
  expectThrows<std::invalid_argument>(
    [&grid, &state]() {
      std::vector<double> wrong_size(2, 0.0);
      burgers::computeMolecularViscousFluxes(
        grid, state, 0.1, wrong_size);
    },
    "viscous flux must reject a mismatched output array");
  expectThrows<std::invalid_argument>(
    [&grid, &state, &viscous_fluxes]() {
      burgers::computeMolecularViscousFluxes(
        grid, state, -0.1, viscous_fluxes);
    },
    "viscous flux must reject negative molecular viscosity");

  if(failures != 0) {
    std::cerr << failures << " Phase 2 flux check(s) failed\n";
    return 1;
  }

  std::cout << "Phase 2 flux checks passed\n";
  return 0;
}
