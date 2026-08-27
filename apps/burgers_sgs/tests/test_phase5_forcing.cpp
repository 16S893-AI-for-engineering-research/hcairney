#include "burgers/Config.h"
#include "burgers/Diagnostics.h"
#include "burgers/Forcing.h"
#include "burgers/Grid.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
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

double maximumDifference(const burgers::State& left,
                         const burgers::State& right) {
  double result = 0.0;
  for(std::size_t cell = 0; cell < left.size(); ++cell) {
    result = std::max(result, std::abs(left[cell] - right[cell]));
  }
  return result;
}

burgers::ForcingConfig stochasticConfig(double spectral_exponent = 0.0) {
  burgers::ForcingConfig config;
  config.type = burgers::ForcingType::StochasticFourierOu;
  config.stochastic = burgers::makeLowModeOuForcingConfig();
  config.stochastic.spectral_exponent = spectral_exponent;
  config.stochastic.stationary_rms = 0.1;
  config.stochastic.correlation_time = 0.5;
  config.stochastic.clock_interval = 0.05;
  return config;
}

void advanceOneClock(burgers::Forcing& forcing, double time) {
  const double time_step = forcing.clockInterval();
  forcing.prepareStep(time, time_step);
  forcing.commitStep(time + time_step);
}

void checkPresets() {
  const burgers::StochasticForcingConfig low =
    burgers::makeLowModeOuForcingConfig();
  const burgers::StochasticForcingConfig cy =
    burgers::makeChekhlovYakhotForcingConfig();
  expect(low.wavenumbers == std::vector<int>({1, 2, 3}),
         "low-mode OU preset must contain modes 1:3");
  expectNear(low.spectral_exponent, 0.0, 0.0,
             "low-mode OU preset must use p=0");
  expect(cy.wavenumbers.size() == 8 && cy.wavenumbers.front() == 1 &&
           cy.wavenumbers.back() == 8,
         "Chekhlov-Yakhot-type preset must contain modes 1:8");
  expectNear(cy.spectral_exponent, 1.0, 0.0,
             "Chekhlov-Yakhot-type preset must use p=1");
  expectNear(low.clock_interval, low.correlation_time / 10.0, 0.0,
             "default forcing clock must resolve tau with ten intervals");
}

void checkDeterministicCellAverages() {
  const double pi = 3.1415926535897932384626433832795;
  const burgers::Grid grid(0.0, 2.0 * pi, 64);
  burgers::ForcingConfig config;
  config.type = burgers::ForcingType::DeterministicFourier;
  config.deterministic.modes = {{1, 0.1, 0.0}};
  burgers::Forcing forcing(grid, config, 0.01, 123u);
  burgers::ForcingFields fields(grid);
  forcing.evaluate(0.0, fields);

  const double width = grid.cellWidth();
  const double expected_first =
    0.1 * (1.0 - std::cos(width)) / width;
  expectNear(fields.deterministic[0], expected_first, 2.0e-16,
             "deterministic sine forcing must use analytic cell averages");
  expectNear(burgers::mean(grid, fields.deterministic), 0.0, 2.0e-17,
             "deterministic forcing must have zero discrete mean");
  expectNear(maximumDifference(fields.deterministic, fields.total),
             0.0, 0.0,
             "single deterministic component must equal total forcing");

  const std::vector<double> spectrum =
    burgers::energySpectrum(grid, fields.deterministic);
  double spectral_energy = 0.0;
  for(const double value : spectrum) {
    spectral_energy += value;
  }
  expectNear(spectral_energy,
             burgers::kineticEnergy(grid, fields.deterministic),
             2.0e-15,
             "one-sided energy spectrum must satisfy discrete Parseval "
             "balance");
}

void checkSeedAndGridReproducibility() {
  const double pi = 3.1415926535897932384626433832795;
  const burgers::Grid coarse_grid(0.0, 2.0 * pi, 64);
  const burgers::Grid fine_grid(0.0, 2.0 * pi, 128);
  const burgers::ForcingConfig config = stochasticConfig(1.0);
  burgers::Forcing first(coarse_grid, config, 0.01, 771u);
  burgers::Forcing second(coarse_grid, config, 0.01, 771u);
  burgers::Forcing different(coarse_grid, config, 0.01, 772u);
  burgers::Forcing fine(fine_grid, config, 0.01, 771u);
  burgers::ForcingFields first_fields(coarse_grid);
  burgers::ForcingFields second_fields(coarse_grid);
  burgers::ForcingFields different_fields(coarse_grid);
  burgers::ForcingFields fine_fields(fine_grid);
  first.evaluate(0.0, first_fields);
  second.evaluate(0.0, second_fields);
  different.evaluate(0.0, different_fields);
  fine.evaluate(0.0, fine_fields);

  expectNear(maximumDifference(first_fields.stochastic,
                               second_fields.stochastic),
             0.0, 0.0,
             "equal seeds must produce bitwise-identical forcing fields");
  expect(maximumDifference(first_fields.stochastic,
                           different_fields.stochastic) > 1.0e-10,
         "different seeds must produce different forcing fields");
  for(std::size_t cell = 0; cell < coarse_grid.cellCount(); ++cell) {
    const double restricted =
      0.5 * (fine_fields.stochastic[2 * cell] +
             fine_fields.stochastic[2 * cell + 1]);
    expectNear(restricted, first_fields.stochastic[cell], 2.0e-15,
               "one Fourier realization must restrict consistently across "
               "grids");
  }
  expectNear(burgers::mean(coarse_grid, first_fields.stochastic),
             0.0, 2.0e-17,
             "each stochastic forcing realization must have zero mean");
}

void checkClockAndRestart() {
  const double pi = 3.1415926535897932384626433832795;
  const burgers::Grid grid(0.0, 2.0 * pi, 64);
  const burgers::ForcingConfig config = stochasticConfig();
  burgers::Forcing uninterrupted(grid, config, 0.01, 891u);
  burgers::ForcingFields at_start(grid);
  burgers::ForcingFields within_interval(grid);
  uninterrupted.evaluate(0.0, at_start);
  uninterrupted.prepareStep(0.0, 0.04);
  uninterrupted.evaluate(0.04, within_interval);
  expectNear(maximumDifference(at_start.stochastic,
                               within_interval.stochastic),
             0.0, 0.0,
             "piecewise-constant forcing must not change within a clock "
             "interval");
  uninterrupted.commitStep(0.04);

  uninterrupted.prepareStep(0.04, 0.01);
  uninterrupted.commitStep(0.05);
  burgers::ForcingFields after_tick(grid);
  uninterrupted.evaluate(0.05, after_tick);
  expect(maximumDifference(at_start.stochastic, after_tick.stochastic) >
           1.0e-12,
         "OU coefficients must change at a forcing-clock boundary");

  for(std::size_t tick = 1; tick < 5; ++tick) {
    advanceOneClock(uninterrupted, 0.05 * static_cast<double>(tick));
  }
  const std::string saved = uninterrupted.serializeStochasticState();
  burgers::Forcing restarted(grid, config, 0.01, 891u);
  restarted.restoreStochasticState(saved);
  burgers::ForcingFields saved_uninterrupted(grid);
  burgers::ForcingFields saved_restarted(grid);
  uninterrupted.evaluate(0.25, saved_uninterrupted);
  restarted.evaluate(0.25, saved_restarted);
  expectNear(maximumDifference(saved_uninterrupted.stochastic,
                               saved_restarted.stochastic),
             0.0, 0.0,
             "restored forcing must reproduce its checkpoint field");

  advanceOneClock(uninterrupted, 0.25);
  advanceOneClock(restarted, 0.25);
  uninterrupted.evaluate(0.30, saved_uninterrupted);
  restarted.evaluate(0.30, saved_restarted);
  expectNear(maximumDifference(saved_uninterrupted.stochastic,
                               saved_restarted.stochastic),
             0.0, 0.0,
             "restored forcing must continue the exact random sequence");
}

void checkComposition() {
  const double pi = 3.1415926535897932384626433832795;
  const burgers::Grid grid(0.0, 2.0 * pi, 64);
  burgers::ForcingConfig config = stochasticConfig(1.0);
  config.type = burgers::ForcingType::Composite;
  config.deterministic.modes = {{1, 0.1, 0.0}};
  burgers::Forcing forcing(grid, config, 0.01, 44u);
  burgers::ForcingFields fields(grid);
  forcing.evaluate(0.0, fields);
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    expectNear(fields.total[cell],
               fields.deterministic[cell] + fields.stochastic[cell],
               2.0e-16,
               "composite forcing must be the sum of separately exposed "
               "components");
  }
  expectNear(burgers::mean(grid, fields.deterministic), 0.0, 2.0e-17,
             "composite deterministic component must be zero mean");
  expectNear(burgers::mean(grid, fields.stochastic), 0.0, 2.0e-17,
             "composite stochastic component must be zero mean");
}

double recoverCosineCoefficient(const burgers::Grid& grid,
                                const burgers::State& field,
                                int mode) {
  const double argument =
    0.5 * static_cast<double>(mode) * grid.cellWidth();
  const double cell_average_factor = std::sin(argument) / argument;
  double projection = 0.0;
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    projection += field[cell] *
      std::cos(static_cast<double>(mode) * grid.cellCenter(cell));
  }
  return 2.0 * projection /
         (static_cast<double>(grid.cellCount()) * cell_average_factor);
}

void checkOuStatistics() {
  const double pi = 3.1415926535897932384626433832795;
  const burgers::Grid grid(0.0, 2.0 * pi, 128);
  const burgers::ForcingConfig config = stochasticConfig(1.0);
  burgers::Forcing forcing(grid, config, 0.01, 2025u);
  burgers::ForcingFields fields(grid);
  const std::size_t sample_count = 10000;
  std::vector<double> mode_one(sample_count, 0.0);
  double mode_one_square_sum = 0.0;
  double mode_three_square_sum = 0.0;
  double continuum_spatial_square_sum = 0.0;

  for(std::size_t sample = 0; sample < sample_count; ++sample) {
    const double time =
      static_cast<double>(sample) * config.stochastic.clock_interval;
    forcing.evaluate(time, fields);
    const double coefficient_one =
      recoverCosineCoefficient(grid, fields.stochastic, 1);
    const double coefficient_three =
      recoverCosineCoefficient(grid, fields.stochastic, 3);
    mode_one[sample] = coefficient_one;
    mode_one_square_sum += coefficient_one * coefficient_one;
    mode_three_square_sum += coefficient_three * coefficient_three;

    double cell_square_sum = 0.0;
    for(const double value : fields.stochastic) {
      cell_square_sum += value * value;
    }
    continuum_spatial_square_sum +=
      cell_square_sum / static_cast<double>(grid.cellCount());
    advanceOneClock(forcing, time);
  }

  const double variance_ratio =
    mode_one_square_sum / mode_three_square_sum;
  std::cout << "OU p=1 variance ratio V1/V3=" << variance_ratio << '\n';
  expect(variance_ratio > 2.4 && variance_ratio < 3.6,
         "p=1 OU forcing must produce coefficient variance proportional "
         "to k^-1");

  double lag_product = 0.0;
  double lag_left_square = 0.0;
  double lag_right_square = 0.0;
  for(std::size_t sample = 0; sample + 1 < sample_count; ++sample) {
    lag_product += mode_one[sample] * mode_one[sample + 1];
    lag_left_square += mode_one[sample] * mode_one[sample];
    lag_right_square += mode_one[sample + 1] * mode_one[sample + 1];
  }
  const double measured_correlation =
    lag_product / std::sqrt(lag_left_square * lag_right_square);
  const double expected_correlation = std::exp(
    -config.stochastic.clock_interval /
    config.stochastic.correlation_time);
  std::cout << "OU lag-one correlation measured=" << measured_correlation
            << " expected=" << expected_correlation << '\n';
  expectNear(measured_correlation, expected_correlation, 0.025,
             "clocked OU forcing must reproduce its configured lag-one "
             "correlation");

  const double measured_cell_rms = std::sqrt(
    continuum_spatial_square_sum / static_cast<double>(sample_count));
  std::cout << "OU stationary cell RMS measured=" << measured_cell_rms
            << " configured=" << config.stochastic.stationary_rms << '\n';
  expectNear(measured_cell_rms,
             config.stochastic.stationary_rms,
             0.012,
             "normalized OU modes must reproduce the requested stationary "
             "RMS up to finite-volume attenuation and sampling error");
}

}  // namespace

int main() {
  checkPresets();
  checkDeterministicCellAverages();
  checkSeedAndGridReproducibility();
  checkClockAndRestart();
  checkComposition();
  checkOuStatistics();

  if(failures != 0) {
    std::cerr << failures << " Phase 5 forcing check(s) failed\n";
    return 1;
  }
  std::cout << "Phase 5 forcing checks passed\n";
  return 0;
}
