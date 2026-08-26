#include "burgers/Config.h"

#include <cmath>

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

  for(std::size_t index = 0;
      index < config.forcing.deterministic.modes.size(); ++index) {
    const FourierModeConfig& mode =
      config.forcing.deterministic.modes[index];
    const std::string prefix =
      "forcing.deterministic.modes[" + std::to_string(index) + "]";
    if(mode.wavenumber == 0) {
      errors.emplace_back(prefix + ".wavenumber must be nonzero");
    }
    requireFinite(errors, mode.amplitude, (prefix + ".amplitude").c_str());
    requireFinite(errors, mode.phase, (prefix + ".phase").c_str());
  }
  for(std::size_t index = 0;
      index < config.forcing.stochastic.wavenumbers.size(); ++index) {
    if(config.forcing.stochastic.wavenumbers[index] == 0) {
      errors.emplace_back(
        "forcing.stochastic.wavenumbers[" + std::to_string(index) +
        "] must be nonzero");
    }
  }
  requireFinite(errors, config.forcing.stochastic.standard_deviation,
                "forcing.stochastic.standard_deviation");
  requireFinite(errors, config.forcing.stochastic.correlation_time,
                "forcing.stochastic.correlation_time");
  if(isFinite(config.forcing.stochastic.standard_deviation) &&
     config.forcing.stochastic.standard_deviation < 0.0) {
    errors.emplace_back(
      "forcing.stochastic.standard_deviation must be nonnegative");
  }
  if(isFinite(config.forcing.stochastic.correlation_time) &&
     config.forcing.stochastic.correlation_time <= 0.0) {
    errors.emplace_back(
      "forcing.stochastic.correlation_time must be positive");
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
  requireFinite(errors, config.output.history_interval,
                "output.history_interval");
  requireFinite(errors, config.output.profile_interval,
                "output.profile_interval");
  if(isFinite(config.output.history_interval) &&
     config.output.history_interval <= 0.0) {
    errors.emplace_back("output.history_interval must be positive");
  }
  if(isFinite(config.output.profile_interval) &&
     config.output.profile_interval <= 0.0) {
    errors.emplace_back("output.profile_interval must be positive");
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

