#pragma once

#include "burgers/Forcing.h"
#include "burgers/Grid.h"
#include "burgers/State.h"

#include <vector>

namespace burgers {

struct ErrorNorms {
  double l1 = 0.0;
  double l2 = 0.0;
  double l_infinity = 0.0;
};

struct UnforcedEnergyBudgetRate {
  double energy_rate = 0.0;
  double molecular_dissipation = 0.0;
  double numerical_dissipation = 0.0;
};

struct ForcingPower {
  double manufactured = 0.0;
  double deterministic = 0.0;
  double stochastic = 0.0;
  double total = 0.0;
};

struct ForcedEnergyBudgetRate {
  double energy_rate = 0.0;
  ForcingPower power;
  double molecular_dissipation = 0.0;
  double numerical_dissipation = 0.0;
};

double integral(const Grid& grid, const State& state);
double mean(const Grid& grid, const State& state);
double l1Norm(const Grid& grid, const State& state);
double l2Norm(const Grid& grid, const State& state);
double lInfinityNorm(const Grid& grid, const State& state);
double kineticEnergy(const Grid& grid, const State& state);
double spatialVariance(const Grid& grid, const State& state);
std::vector<double> energySpectrum(const Grid& grid, const State& state);
double molecularDissipation(const Grid& grid,
                            const State& state,
                            double molecular_viscosity);
double powerInput(const Grid& grid,
                  const State& state,
                  const State& forcing);
ForcingPower forcingPower(const Grid& grid,
                          const State& state,
                          const ForcingFields& forcing);
ForcedEnergyBudgetRate forcedEnergyBudgetRate(
  const Grid& grid,
  const State& state,
  const State& derivative,
  const ForcingFields& forcing,
  double molecular_viscosity);
ErrorNorms errorNorms(const Grid& grid,
                      const State& numerical,
                      const State& reference);
UnforcedEnergyBudgetRate unforcedEnergyBudgetRate(
  const Grid& grid,
  const State& state,
  const State& derivative,
  double molecular_viscosity);

}  // namespace burgers
