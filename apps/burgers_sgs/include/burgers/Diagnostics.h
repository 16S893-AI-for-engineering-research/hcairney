#pragma once

#include "burgers/Grid.h"
#include "burgers/State.h"

namespace burgers {

double integral(const Grid& grid, const State& state);
double mean(const Grid& grid, const State& state);
double l1Norm(const Grid& grid, const State& state);
double l2Norm(const Grid& grid, const State& state);
double lInfinityNorm(const Grid& grid, const State& state);

}  // namespace burgers
