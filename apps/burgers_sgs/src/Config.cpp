#include "burgers/Config.h"

#include <cmath>
#include <set>

namespace burgers {
namespace {

bool isFinite(double value) {
  return std::isfinite(value) != 0;
}

void requireFinite(std::vector<std::string>& errors,
                   double value,
                   const char* name) {
  if(!isFinite(value)) {
    errors.emplace_back(std::string(name) + " must be finite");
  }
}

}  // namespace

RunConfig makeDefaultRunConfig() {
  return RunConfig{};
}

RunConfig makePhase5ValidationRunConfig() {
  RunConfig config;
  config.grid.cell_count = 256;
  config.initial_condition.mean = 0.0;
  config.initial_condition.amplitude = 0.1;
  config.viscosity.molecular = 0.01;
  config.numerical_method.reconstruction = Reconstruction::Muscl;
  config.numerical_method.convective_flux = ConvectiveFlux::Godunov;
  config.numerical_method.limiter = Limiter::MonotonizedCentral;
  config.time_integration.final_time = 50.0;
  config.forcing.type = ForcingType::Composite;
  config.forcing.deterministic.modes = {{1, 0.1, 0.0}};
  config.forcing.stochastic = makeLowModeOuForcingConfig();
  config.output.history_interval = 0.05;
  config.output.statistics_start_time = 10.0;
  return config;
}

StochasticForcingConfig makeLowModeOuForcingConfig() {
  return StochasticForcingConfig{};
}

StochasticForcingConfig makeChekhlovYakhotForcingConfig() {
  StochasticForcingConfig config;
  config.wavenumbers = {1, 2, 3, 4, 5, 6, 7, 8};
  config.spectral_exponent = 1.0;
  return config;
}

std::vector<std::string> validate(const RunConfig& config) {
  std::vector<std::string> errors;

  requireFinite(errors, config.grid.x_begin, "grid.x_begin");
  requireFinite(errors, config.grid.x_end, "grid.x_end");
  if(isFinite(config.grid.x_begin) && isFinite(config.grid.x_end) &&
     config.grid.x_end <= config.grid.x_begin) {
    errors.emplace_back("grid.x_end must be greater than grid.x_begin");
  }
  if(config.grid.cell_count == 0) {
    errors.emplace_back("grid.cell_count must be positive");
  }

  requireFinite(errors, config.initial_condition.mean,
                "initial_condition.mean");
  requireFinite(errors, config.initial_condition.amplitude,
                "initial_condition.amplitude");
  requireFinite(errors, config.initial_condition.phase,
                "initial_condition.phase");
  requireFinite(errors, config.initial_condition.left_state,
                "initial_condition.left_state");
  requireFinite(errors, config.initial_condition.right_state,
                "initial_condition.right_state");
  if(config.initial_condition.type == InitialConditionType::Sinusoidal &&
     config.initial_condition.wavenumber == 0) {
    errors.emplace_back(
      "initial_condition.wavenumber must be nonzero for sinusoidal data");
  }
  if(config.initial_condition.type == InitialConditionType::Random &&
     isFinite(config.initial_condition.amplitude) &&
     config.initial_condition.amplitude < 0.0) {
    errors.emplace_back(
      "initial_condition.amplitude must be nonnegative for random data");
  }

  requireFinite(errors, config.viscosity.molecular,
                "viscosity.molecular");
  if(isFinite(config.viscosity.molecular) &&
     config.viscosity.molecular < 0.0) {
    errors.emplace_back("viscosity.molecular must be nonnegative");
  }

  if(config.numerical_method.reconstruction ==
       Reconstruction::PiecewiseConstant &&
     config.numerical_method.limiter != Limiter::None) {
    errors.emplace_back(
      "numerical_method.limiter must be none for piecewise-constant "
      "reconstruction");
  }
  if(config.numerical_method.reconstruction == Reconstruction::Muscl &&
     config.numerical_method.limiter == Limiter::None) {
    errors.emplace_back(
      "numerical_method.limiter must be selected for MUSCL reconstruction");
  }

  requireFinite(errors, config.time_integration.initial_time,
                "time_integration.initial_time");
  requireFinite(errors, config.time_integration.final_time,
                "time_integration.final_time");
  requireFinite(errors, config.time_integration.advective_cfl,
                "time_integration.advective_cfl");
  requireFinite(errors, config.time_integration.diffusive_cfl,
                "time_integration.diffusive_cfl");
  if(isFinite(config.time_integration.initial_time) &&
     isFinite(config.time_integration.final_time) &&
     config.time_integration.final_time <
       config.time_integration.initial_time) {
    errors.emplace_back(
      "time_integration.final_time must not precede initial_time");
  }
  if(isFinite(config.time_integration.advective_cfl) &&
     config.time_integration.advective_cfl <= 0.0) {
    errors.emplace_back("time_integration.advective_cfl must be positive");
  }
  if(isFinite(config.time_integration.diffusive_cfl) &&
     config.time_integration.diffusive_cfl <= 0.0) {
    errors.emplace_back("time_integration.diffusive_cfl must be positive");
  }
  if(config.time_integration.maximum_steps == 0) {
    errors.emplace_back("time_integration.maximum_steps must be positive");
  }

  requireFinite(errors, config.forcing.manufactured.amplitude,
                "forcing.manufactured.amplitude");
  requireFinite(errors, config.forcing.manufactured.phase,
                "forcing.manufactured.phase");
  requireFinite(errors, config.forcing.manufactured.decay_rate,
                "forcing.manufactured.decay_rate");
  if(config.forcing.manufactured.wavenumber == 0) {
    errors.emplace_back(
      "forcing.manufactured.wavenumber must be nonzero");
  }
  if(isFinite(config.forcing.manufactured.decay_rate) &&
     config.forcing.manufactured.decay_rate <= 0.0) {
    errors.emplace_back(
      "forcing.manufactured.decay_rate must be positive");
  }
  if(config.forcing.type == ForcingType::Manufactured &&
     isFinite(config.grid.x_begin) && isFinite(config.grid.x_end) &&
     config.grid.x_end > config.grid.x_begin &&
     config.forcing.manufactured.wavenumber != 0) {
    const double pi = 3.1415926535897932384626433832795;
    const double cycles =
      static_cast<double>(config.forcing.manufactured.wavenumber) *
      (config.grid.x_end - config.grid.x_begin) / (2.0 * pi);
    const double nearest_integer = std::round(cycles);
    const double tolerance = 1.0e-12 * std::fmax(1.0, std::abs(cycles));
    if(std::abs(cycles - nearest_integer) > tolerance) {
      errors.emplace_back(
        "manufactured forcing wavenumber must be periodic on the grid domain");
    }
  }

  const bool deterministic_forcing_enabled =
    config.forcing.type == ForcingType::DeterministicFourier ||
    config.forcing.type == ForcingType::Composite;
  const bool stochastic_forcing_enabled =
    config.forcing.type == ForcingType::StochasticFourierOu ||
    config.forcing.type == ForcingType::Composite;

  for(std::size_t index = 0;
      index < config.forcing.deterministic.modes.size(); ++index) {
    const FourierModeConfig& mode =
      config.forcing.deterministic.modes[index];
    const std::string prefix =
      "forcing.deterministic.modes[" + std::to_string(index) + "]";
    if(mode.wavenumber <= 0) {
      errors.emplace_back(prefix + ".wavenumber must be positive");
    } else if(deterministic_forcing_enabled &&
              static_cast<std::size_t>(mode.wavenumber) >=
                config.grid.cell_count / 2 + config.grid.cell_count % 2) {
      errors.emplace_back(prefix + ".wavenumber must be below grid Nyquist");
    }
    requireFinite(errors, mode.amplitude, (prefix + ".amplitude").c_str());
    requireFinite(errors, mode.phase, (prefix + ".phase").c_str());
  }
  if(deterministic_forcing_enabled &&
     config.forcing.deterministic.modes.empty()) {
    errors.emplace_back(
      "forcing.deterministic.modes must not be empty when deterministic "
      "forcing is enabled");
  }
  std::set<int> deterministic_wavenumbers;
  for(const FourierModeConfig& mode : config.forcing.deterministic.modes) {
    if(mode.wavenumber > 0 &&
       !deterministic_wavenumbers.insert(mode.wavenumber).second) {
      errors.emplace_back(
        "forcing.deterministic.modes must have unique wavenumbers");
      break;
    }
  }

  std::set<int> stochastic_wavenumbers;
  for(std::size_t index = 0;
      index < config.forcing.stochastic.wavenumbers.size(); ++index) {
    const int wavenumber = config.forcing.stochastic.wavenumbers[index];
    if(wavenumber <= 0) {
      errors.emplace_back(
        "forcing.stochastic.wavenumbers[" + std::to_string(index) +
        "] must be positive");
    } else if(stochastic_forcing_enabled &&
              static_cast<std::size_t>(wavenumber) >=
                config.grid.cell_count / 2 + config.grid.cell_count % 2) {
      errors.emplace_back(
        "forcing.stochastic.wavenumbers[" + std::to_string(index) +
        "] must be below grid Nyquist");
    } else if(!stochastic_wavenumbers.insert(wavenumber).second) {
      errors.emplace_back(
        "forcing.stochastic.wavenumbers must be unique");
    }
  }
  if(stochastic_forcing_enabled &&
     config.forcing.stochastic.wavenumbers.empty()) {
    errors.emplace_back(
      "forcing.stochastic.wavenumbers must not be empty when stochastic "
      "forcing is enabled");
  }
  requireFinite(errors, config.forcing.stochastic.spectral_exponent,
                "forcing.stochastic.spectral_exponent");
  requireFinite(errors, config.forcing.stochastic.stationary_rms,
                "forcing.stochastic.stationary_rms");
  requireFinite(errors, config.forcing.stochastic.correlation_time,
                "forcing.stochastic.correlation_time");
  requireFinite(errors, config.forcing.stochastic.clock_interval,
                "forcing.stochastic.clock_interval");
  requireFinite(errors, config.forcing.stochastic.clock_reference_time,
                "forcing.stochastic.clock_reference_time");
  if(isFinite(config.forcing.stochastic.spectral_exponent) &&
     config.forcing.stochastic.spectral_exponent < 0.0) {
    errors.emplace_back(
      "forcing.stochastic.spectral_exponent must be nonnegative");
  }
  if(isFinite(config.forcing.stochastic.stationary_rms) &&
     config.forcing.stochastic.stationary_rms < 0.0) {
    errors.emplace_back(
      "forcing.stochastic.stationary_rms must be nonnegative");
  }
  if(stochastic_forcing_enabled &&
     isFinite(config.forcing.stochastic.stationary_rms) &&
     config.forcing.stochastic.stationary_rms == 0.0) {
    errors.emplace_back(
      "forcing.stochastic.stationary_rms must be positive when stochastic "
      "forcing is enabled");
  }
  if(isFinite(config.forcing.stochastic.correlation_time) &&
     config.forcing.stochastic.correlation_time <= 0.0) {
    errors.emplace_back(
      "forcing.stochastic.correlation_time must be positive");
  }
  if(isFinite(config.forcing.stochastic.clock_interval) &&
     config.forcing.stochastic.clock_interval <= 0.0) {
    errors.emplace_back(
      "forcing.stochastic.clock_interval must be positive");
  }
  if(stochastic_forcing_enabled &&
     isFinite(config.forcing.stochastic.clock_reference_time) &&
     isFinite(config.time_integration.initial_time) &&
     config.forcing.stochastic.clock_reference_time >
       config.time_integration.initial_time) {
    errors.emplace_back(
      "forcing.stochastic.clock_reference_time must not follow the initial "
      "simulation time");
  }
  if((deterministic_forcing_enabled || stochastic_forcing_enabled) &&
     !config.forcing.remove_discrete_mean) {
    errors.emplace_back(
      "forcing.remove_discrete_mean must be true for production forcing");
  }

  requireFinite(errors, config.closure.static_coefficient,
                "closure.static_coefficient");
  requireFinite(errors, config.closure.minimum_coefficient,
                "closure.minimum_coefficient");
  requireFinite(errors, config.closure.maximum_coefficient,
                "closure.maximum_coefficient");
  requireFinite(errors, config.closure.test_filter_ratio,
                "closure.test_filter_ratio");
  requireFinite(errors, config.closure.denominator_regularization,
                "closure.denominator_regularization");
  if(isFinite(config.closure.minimum_coefficient) &&
     config.closure.minimum_coefficient < 0.0) {
    errors.emplace_back("closure.minimum_coefficient must be nonnegative");
  }
  if(isFinite(config.closure.minimum_coefficient) &&
     isFinite(config.closure.maximum_coefficient) &&
     config.closure.maximum_coefficient <
       config.closure.minimum_coefficient) {
    errors.emplace_back(
      "closure.maximum_coefficient must not be below minimum_coefficient");
  }
  if(isFinite(config.closure.static_coefficient) &&
     isFinite(config.closure.minimum_coefficient) &&
     isFinite(config.closure.maximum_coefficient) &&
     (config.closure.static_coefficient <
        config.closure.minimum_coefficient ||
      config.closure.static_coefficient >
        config.closure.maximum_coefficient)) {
    errors.emplace_back(
      "closure.static_coefficient must lie within the coefficient bounds");
  }
  if(isFinite(config.closure.test_filter_ratio) &&
     config.closure.test_filter_ratio <= 1.0) {
    errors.emplace_back("closure.test_filter_ratio must be greater than one");
  }
  if(isFinite(config.closure.denominator_regularization) &&
     config.closure.denominator_regularization < 0.0) {
    errors.emplace_back(
      "closure.denominator_regularization must be nonnegative");
  }

  if(config.output.directory.empty()) {
    errors.emplace_back("output.directory must not be empty");
  }
  if(config.output.metadata_filename.empty()) {
    errors.emplace_back("output.metadata_filename must not be empty");
  }
  if(config.output.scalar_history_filename.empty()) {
    errors.emplace_back("output.scalar_history_filename must not be empty");
  }
  if(config.output.final_profile_filename.empty()) {
    errors.emplace_back("output.final_profile_filename must not be empty");
  }
  if(config.output.profile_history_filename.empty()) {
    errors.emplace_back("output.profile_history_filename must not be empty");
  }
  if(config.output.spectrum_filename.empty()) {
    errors.emplace_back("output.spectrum_filename must not be empty");
  }
  requireFinite(errors, config.output.history_interval,
                "output.history_interval");
  requireFinite(errors, config.output.statistics_start_time,
                "output.statistics_start_time");
  if(isFinite(config.output.history_interval) &&
     config.output.history_interval <= 0.0) {
    errors.emplace_back("output.history_interval must be positive");
  }
  if(isFinite(config.output.statistics_start_time) &&
     isFinite(config.time_integration.initial_time) &&
     isFinite(config.time_integration.final_time) &&
     (config.output.statistics_start_time <
        config.time_integration.initial_time ||
      config.output.statistics_start_time >
        config.time_integration.final_time)) {
    errors.emplace_back(
      "output.statistics_start_time must lie within the simulation time "
      "interval");
  }

  return errors;
}

const char* toString(InitialConditionType value) {
  switch(value) {
    case InitialConditionType::Constant: return "constant";
    case InitialConditionType::Sinusoidal: return "sinusoidal";
    case InitialConditionType::Random: return "random";
    case InitialConditionType::PeriodicTwoState: return "periodic_two_state";
  }
  return "unknown";
}

const char* toString(Reconstruction value) {
  switch(value) {
    case Reconstruction::PiecewiseConstant: return "piecewise_constant";
    case Reconstruction::Muscl: return "muscl";
  }
  return "unknown";
}

const char* toString(ConvectiveFlux value) {
  switch(value) {
    case ConvectiveFlux::Godunov: return "godunov";
    case ConvectiveFlux::Rusanov: return "rusanov";
  }
  return "unknown";
}

const char* toString(Limiter value) {
  switch(value) {
    case Limiter::None: return "none";
    case Limiter::Minmod: return "minmod";
    case Limiter::MonotonizedCentral: return "monotonized_central";
    case Limiter::VanLeer: return "van_leer";
  }
  return "unknown";
}

const char* toString(TimeIntegrator value) {
  switch(value) {
    case TimeIntegrator::SspRk3: return "ssp_rk3";
  }
  return "unknown";
}

const char* toString(ForcingType value) {
  switch(value) {
    case ForcingType::None: return "none";
    case ForcingType::Manufactured: return "manufactured";
    case ForcingType::DeterministicFourier: return "deterministic_fourier";
    case ForcingType::StochasticFourierOu: return "stochastic_fourier_ou";
    case ForcingType::Composite: return "composite";
  }
  return "unknown";
}

const char* toString(ClosureType value) {
  switch(value) {
    case ClosureType::NoClosure: return "no_closure";
    case ClosureType::StaticSmagorinsky: return "static_smagorinsky";
    case ClosureType::DynamicSmagorinsky: return "dynamic_smagorinsky";
    case ClosureType::PrescribedCoefficientField:
      return "prescribed_coefficient_field";
  }
  return "unknown";
}

const char* toString(FaceViscosityAveraging value) {
  switch(value) {
    case FaceViscosityAveraging::Arithmetic: return "arithmetic";
    case FaceViscosityAveraging::Harmonic: return "harmonic";
  }
  return "unknown";
}

}  // namespace burgers
