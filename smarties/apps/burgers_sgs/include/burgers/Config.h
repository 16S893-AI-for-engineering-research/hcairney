#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace burgers {

enum class InitialConditionType {
  Constant,
  Sinusoidal,
  Random,
  PeriodicTwoState
};

enum class Reconstruction {
  PiecewiseConstant,
  Muscl
};

enum class ConvectiveFlux {
  Godunov,
  Rusanov
};

enum class Limiter {
  None,
  Minmod,
  MonotonizedCentral,
  VanLeer
};

enum class TimeIntegrator {
  SspRk3
};

enum class ForcingType {
  None,
  Manufactured,
  DeterministicFourier,
  StochasticFourierOu,
  Composite
};

enum class ClosureType {
  NoClosure,
  StaticSmagorinsky,
  DynamicSmagorinsky,
  PrescribedCoefficientField
};

enum class FaceViscosityAveraging {
  Arithmetic,
  Harmonic
};

struct GridConfig {
  double x_begin = 0.0;
  double x_end = 6.283185307179586476925286766559;
  std::size_t cell_count = 64;
};

struct InitialConditionConfig {
  InitialConditionType type = InitialConditionType::Sinusoidal;
  double mean = 0.0;
  double amplitude = 1.0;
  int wavenumber = 1;
  double phase = 0.0;
  double left_state = 1.0;
  double right_state = -1.0;
};

struct ViscosityConfig {
  double molecular = 0.01;
  FaceViscosityAveraging face_averaging =
    FaceViscosityAveraging::Arithmetic;
};

struct NumericalMethodConfig {
  Reconstruction reconstruction = Reconstruction::PiecewiseConstant;
  ConvectiveFlux convective_flux = ConvectiveFlux::Godunov;
  Limiter limiter = Limiter::None;
};

struct TimeIntegrationConfig {
  TimeIntegrator integrator = TimeIntegrator::SspRk3;
  double initial_time = 0.0;
  double final_time = 1.0;
  double advective_cfl = 0.4;
  double diffusive_cfl = 0.2;
  std::size_t maximum_steps = 1000000;
};

struct FourierModeConfig {
  int wavenumber = 1;
  double amplitude = 0.0;
  double phase = 0.0;
};

struct DeterministicForcingConfig {
  std::vector<FourierModeConfig> modes;
};

struct StochasticForcingConfig {
  // Positive Fourier-mode indices. The default is the low-mode OU preset.
  std::vector<int> wavenumbers{1, 2, 3};
  // The stationary quadrature variances satisfy V_k proportional to
  // k^(-spectral_exponent) and sum to stationary_rms^2.
  double spectral_exponent = 0.0;
  double stationary_rms = 0.1;
  double correlation_time = 0.5;
  // Exact OU updates occur at these physical-time intervals and the
  // coefficient field is held fixed between updates.
  double clock_interval = 0.05;
  double clock_reference_time = 0.0;
};

struct ManufacturedForcingConfig {
  double amplitude = 1.0;
  int wavenumber = 1;
  double phase = 0.0;
  double decay_rate = 1.0;
};

struct ForcingConfig {
  ForcingType type = ForcingType::None;
  ManufacturedForcingConfig manufactured;
  DeterministicForcingConfig deterministic;
  StochasticForcingConfig stochastic;
  bool remove_discrete_mean = true;
};

struct ClosureConfig {
  ClosureType type = ClosureType::NoClosure;
  // Configuration and metadata use C_S. Closure kernels square it once and
  // store C_S^2 internally.
  double static_coefficient = 0.0;
  double minimum_coefficient = 0.0;
  double maximum_coefficient = 1.0;
  // Reserved for a possible future dynamic model.
  double test_filter_ratio = 2.0;
  double denominator_regularization = 1.0e-12;
  // Retained in schema version 1 for compatibility; Phase 7 requires true.
  bool store_squared_coefficient = true;
};

struct OutputConfig {
  std::string directory = ".";
  std::string metadata_filename = "burgers_run_metadata.json";
  std::string scalar_history_filename = "burgers_history.csv";
  std::string final_profile_filename = "burgers_final_profile.csv";
  std::string profile_history_filename = "burgers_profiles.csv";
  std::string spectrum_filename = "burgers_mean_spectrum.csv";
  double history_interval = 0.1;
  double profile_interval = 0.1;
  bool write_initial_profile = true;
  bool write_profile_history = true;
  bool write_online_spectrum = true;
  double statistics_start_time = 0.0;
};

struct RandomConfig {
  std::uint64_t seed = 5489u;
};

struct RunConfig {
  GridConfig grid;
  InitialConditionConfig initial_condition;
  ViscosityConfig viscosity;
  NumericalMethodConfig numerical_method;
  TimeIntegrationConfig time_integration;
  ForcingConfig forcing;
  ClosureConfig closure;
  OutputConfig output;
  RandomConfig random;
};

RunConfig makeDefaultRunConfig();
RunConfig makePhase5ValidationRunConfig();
StochasticForcingConfig makeLowModeOuForcingConfig();
StochasticForcingConfig makeChekhlovYakhotForcingConfig();
std::vector<std::string> validate(const RunConfig& config);

const char* toString(InitialConditionType value);
const char* toString(Reconstruction value);
const char* toString(ConvectiveFlux value);
const char* toString(Limiter value);
const char* toString(TimeIntegrator value);
const char* toString(ForcingType value);
const char* toString(ClosureType value);
const char* toString(FaceViscosityAveraging value);

}  // namespace burgers
