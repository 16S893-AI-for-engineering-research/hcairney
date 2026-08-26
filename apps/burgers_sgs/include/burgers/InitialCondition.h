#pragma once

#include "burgers/Config.h"
#include "burgers/Grid.h"
#include "burgers/State.h"

#include <cstdint>
#include <functional>

namespace burgers {

// An initial-condition callback fills finite-volume cell averages on a grid.
using InitialCondition = std::function<void(const Grid&, State&)>;

InitialCondition constantInitialCondition(double value);
InitialCondition sinusoidalInitialCondition(double mean,
                                             double amplitude,
                                             int wavenumber,
                                             double phase);
InitialCondition randomInitialCondition(double mean,
                                         double amplitude,
                                         std::uint64_t seed);
InitialCondition periodicTwoStateInitialCondition(double left_state,
                                                   double right_state);

InitialCondition makeInitialCondition(const InitialConditionConfig& config,
                                      std::uint64_t seed);
void initializeState(const Grid& grid,
                     State& state,
                     const InitialConditionConfig& config,
                     std::uint64_t seed);

}  // namespace burgers
