#include "burgers/ConfigIO.h"

#include "json.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace burgers {
namespace {

using Json = nlohmann::json;

const unsigned int config_schema_version = 1;

std::invalid_argument configError(const std::string& message) {
  return std::invalid_argument("invalid run configuration: " + message);
}

const Json& member(const Json& object,
                   const std::string& name,
                   const std::string& path) {
  const Json::const_iterator found = object.find(name);
  if(found == object.end()) {
    throw configError(path + "." + name + " is required");
  }
  return *found;
}

const Json& objectMember(const Json& object,
                         const std::string& name,
                         const std::string& path) {
  const Json& value = member(object, name, path);
  if(!value.is_object()) {
    throw configError(path + "." + name + " must be an object");
  }
  return value;
}

void requireObject(const Json& value, const std::string& path) {
  if(!value.is_object()) {
    throw configError(path + " must be an object");
  }
}

void requireKeys(const Json& object,
                 std::initializer_list<const char*> allowed,
                 const std::string& path) {
  std::set<std::string> allowed_keys;
  for(const char* key : allowed) {
    allowed_keys.insert(key);
  }
  for(Json::const_iterator item = object.begin(); item != object.end(); ++item) {
    if(allowed_keys.count(item.key()) == 0) {
      throw configError(path + " contains unknown field " + item.key());
    }
  }
}

std::string stringMember(const Json& object,
                         const std::string& name,
                         const std::string& path) {
  const Json& value = member(object, name, path);
  if(!value.is_string()) {
    throw configError(path + "." + name + " must be a string");
  }
  return value.get<std::string>();
}

bool boolMember(const Json& object,
                const std::string& name,
                const std::string& path) {
  const Json& value = member(object, name, path);
  if(!value.is_boolean()) {
    throw configError(path + "." + name + " must be a boolean");
  }
  return value.get<bool>();
}

double doubleMember(const Json& object,
                    const std::string& name,
                    const std::string& path) {
  const Json& value = member(object, name, path);
  if(!value.is_number()) {
    throw configError(path + "." + name + " must be a number");
  }
  return value.get<double>();
}

std::int64_t signedIntegerMember(const Json& object,
                                 const std::string& name,
                                 const std::string& path) {
  const Json& value = member(object, name, path);
  if(!value.is_number_integer()) {
    throw configError(path + "." + name + " must be an integer");
  }
  if(value.is_number_unsigned()) {
    const std::uint64_t converted = value.get<std::uint64_t>();
    if(converted > static_cast<std::uint64_t>(
                     std::numeric_limits<std::int64_t>::max())) {
      throw configError(path + "." + name + " is too large");
    }
    return static_cast<std::int64_t>(converted);
  }
  return value.get<std::int64_t>();
}

std::uint64_t unsignedIntegerMember(const Json& object,
                                    const std::string& name,
                                    const std::string& path) {
  const Json& value = member(object, name, path);
  if(!value.is_number_integer()) {
    throw configError(path + "." + name + " must be a nonnegative integer");
  }
  if(value.is_number_unsigned()) {
    return value.get<std::uint64_t>();
  }
  const std::int64_t converted = value.get<std::int64_t>();
  if(converted < 0) {
    throw configError(path + "." + name + " must be nonnegative");
  }
  return static_cast<std::uint64_t>(converted);
}

template<typename Enum>
Enum enumError(const std::string& path, const std::string& value) {
  throw configError(path + " has unsupported value " + value);
}

InitialConditionType parseInitialConditionType(const std::string& value,
                                               const std::string& path) {
  if(value == "constant") return InitialConditionType::Constant;
  if(value == "sinusoidal") return InitialConditionType::Sinusoidal;
  if(value == "random") return InitialConditionType::Random;
  if(value == "periodic_two_state") return InitialConditionType::PeriodicTwoState;
  return enumError<InitialConditionType>(path, value);
}

Reconstruction parseReconstruction(const std::string& value,
                                   const std::string& path) {
  if(value == "piecewise_constant") return Reconstruction::PiecewiseConstant;
  if(value == "muscl") return Reconstruction::Muscl;
  return enumError<Reconstruction>(path, value);
}

ConvectiveFlux parseConvectiveFlux(const std::string& value,
                                   const std::string& path) {
  if(value == "godunov") return ConvectiveFlux::Godunov;
  if(value == "rusanov") return ConvectiveFlux::Rusanov;
  return enumError<ConvectiveFlux>(path, value);
}

Limiter parseLimiter(const std::string& value, const std::string& path) {
  if(value == "none") return Limiter::None;
  if(value == "minmod") return Limiter::Minmod;
  if(value == "monotonized_central") return Limiter::MonotonizedCentral;
  if(value == "van_leer") return Limiter::VanLeer;
  return enumError<Limiter>(path, value);
}

TimeIntegrator parseTimeIntegrator(const std::string& value,
                                   const std::string& path) {
  if(value == "ssp_rk3") return TimeIntegrator::SspRk3;
  return enumError<TimeIntegrator>(path, value);
}

ForcingType parseForcingType(const std::string& value,
                             const std::string& path) {
  if(value == "none") return ForcingType::None;
  if(value == "manufactured") return ForcingType::Manufactured;
  if(value == "deterministic_fourier") return ForcingType::DeterministicFourier;
  if(value == "stochastic_fourier_ou") return ForcingType::StochasticFourierOu;
  if(value == "composite") return ForcingType::Composite;
  return enumError<ForcingType>(path, value);
}

ClosureType parseClosureType(const std::string& value,
                             const std::string& path) {
  if(value == "no_closure") return ClosureType::NoClosure;
  if(value == "static_smagorinsky") return ClosureType::StaticSmagorinsky;
  if(value == "dynamic_smagorinsky") return ClosureType::DynamicSmagorinsky;
  if(value == "prescribed_coefficient_field") {
    return ClosureType::PrescribedCoefficientField;
  }
  return enumError<ClosureType>(path, value);
}

FaceViscosityAveraging parseFaceAveraging(const std::string& value,
                                          const std::string& path) {
  if(value == "arithmetic") return FaceViscosityAveraging::Arithmetic;
  if(value == "harmonic") return FaceViscosityAveraging::Harmonic;
  return enumError<FaceViscosityAveraging>(path, value);
}

int checkedInt(std::int64_t value, const std::string& path) {
  if(value < std::numeric_limits<int>::min() ||
     value > std::numeric_limits<int>::max()) {
    throw configError(path + " is outside the supported integer range");
  }
  return static_cast<int>(value);
}

std::size_t checkedSize(std::uint64_t value, const std::string& path) {
  if(value > static_cast<std::uint64_t>(
               std::numeric_limits<std::size_t>::max())) {
    throw configError(path + " is outside the supported size range");
  }
  return static_cast<std::size_t>(value);
}

void readConfiguration(const Json& root, RunConfig& config) {
  const Json& grid = objectMember(root, "grid", "configuration");
  requireKeys(grid, {"x_begin", "x_end", "cell_count"}, "configuration.grid");
  config.grid.x_begin = doubleMember(grid, "x_begin", "configuration.grid");
  config.grid.x_end = doubleMember(grid, "x_end", "configuration.grid");
  config.grid.cell_count = checkedSize(
    unsignedIntegerMember(grid, "cell_count", "configuration.grid"),
    "configuration.grid.cell_count");

  const Json& initial = objectMember(root, "initial_condition", "configuration");
  requireKeys(initial,
              {"type", "mean", "amplitude", "wavenumber", "phase",
               "left_state", "right_state"},
              "configuration.initial_condition");
  config.initial_condition.type = parseInitialConditionType(
    stringMember(initial, "type", "configuration.initial_condition"),
    "configuration.initial_condition.type");
  config.initial_condition.mean = doubleMember(initial, "mean", "configuration.initial_condition");
  config.initial_condition.amplitude = doubleMember(initial, "amplitude", "configuration.initial_condition");
  config.initial_condition.wavenumber = checkedInt(
    signedIntegerMember(initial, "wavenumber", "configuration.initial_condition"),
    "configuration.initial_condition.wavenumber");
  config.initial_condition.phase = doubleMember(initial, "phase", "configuration.initial_condition");
  config.initial_condition.left_state = doubleMember(initial, "left_state", "configuration.initial_condition");
  config.initial_condition.right_state = doubleMember(initial, "right_state", "configuration.initial_condition");

  const Json& viscosity = objectMember(root, "viscosity", "configuration");
  requireKeys(viscosity, {"molecular", "face_averaging"}, "configuration.viscosity");
  config.viscosity.molecular = doubleMember(viscosity, "molecular", "configuration.viscosity");
  config.viscosity.face_averaging = parseFaceAveraging(
    stringMember(viscosity, "face_averaging", "configuration.viscosity"),
    "configuration.viscosity.face_averaging");

  const Json& method = objectMember(root, "numerical_method", "configuration");
  requireKeys(method, {"reconstruction", "convective_flux", "limiter"},
              "configuration.numerical_method");
  config.numerical_method.reconstruction = parseReconstruction(
    stringMember(method, "reconstruction", "configuration.numerical_method"),
    "configuration.numerical_method.reconstruction");
  config.numerical_method.convective_flux = parseConvectiveFlux(
    stringMember(method, "convective_flux", "configuration.numerical_method"),
    "configuration.numerical_method.convective_flux");
  config.numerical_method.limiter = parseLimiter(
    stringMember(method, "limiter", "configuration.numerical_method"),
    "configuration.numerical_method.limiter");

  const Json& time = objectMember(root, "time_integration", "configuration");
  requireKeys(time,
              {"integrator", "initial_time", "final_time", "advective_cfl",
               "diffusive_cfl", "maximum_steps"},
              "configuration.time_integration");
  config.time_integration.integrator = parseTimeIntegrator(
    stringMember(time, "integrator", "configuration.time_integration"),
    "configuration.time_integration.integrator");
  config.time_integration.initial_time = doubleMember(time, "initial_time", "configuration.time_integration");
  config.time_integration.final_time = doubleMember(time, "final_time", "configuration.time_integration");
  config.time_integration.advective_cfl = doubleMember(time, "advective_cfl", "configuration.time_integration");
  config.time_integration.diffusive_cfl = doubleMember(time, "diffusive_cfl", "configuration.time_integration");
  config.time_integration.maximum_steps = checkedSize(
    unsignedIntegerMember(time, "maximum_steps", "configuration.time_integration"),
    "configuration.time_integration.maximum_steps");

  const Json& forcing = objectMember(root, "forcing", "configuration");
  requireKeys(forcing,
              {"type", "remove_discrete_mean", "manufactured",
               "deterministic", "stochastic"},
              "configuration.forcing");
  config.forcing.type = parseForcingType(
    stringMember(forcing, "type", "configuration.forcing"),
    "configuration.forcing.type");
  config.forcing.remove_discrete_mean = boolMember(
    forcing, "remove_discrete_mean", "configuration.forcing");

  const Json& manufactured = objectMember(forcing, "manufactured", "configuration.forcing");
  requireKeys(manufactured, {"amplitude", "wavenumber", "phase", "decay_rate"},
              "configuration.forcing.manufactured");
  config.forcing.manufactured.amplitude = doubleMember(manufactured, "amplitude", "configuration.forcing.manufactured");
  config.forcing.manufactured.wavenumber = checkedInt(
    signedIntegerMember(manufactured, "wavenumber", "configuration.forcing.manufactured"),
    "configuration.forcing.manufactured.wavenumber");
  config.forcing.manufactured.phase = doubleMember(manufactured, "phase", "configuration.forcing.manufactured");
  config.forcing.manufactured.decay_rate = doubleMember(manufactured, "decay_rate", "configuration.forcing.manufactured");

  const Json& deterministic = objectMember(forcing, "deterministic", "configuration.forcing");
  requireKeys(deterministic, {"modes"}, "configuration.forcing.deterministic");
  const Json& modes = member(deterministic, "modes", "configuration.forcing.deterministic");
  if(!modes.is_array()) {
    throw configError("configuration.forcing.deterministic.modes must be an array");
  }
  config.forcing.deterministic.modes.clear();
  for(std::size_t index = 0; index < modes.size(); ++index) {
    const Json& mode = modes[index];
    const std::string path = "configuration.forcing.deterministic.modes[" +
      std::to_string(index) + "]";
    requireObject(mode, path);
    requireKeys(mode, {"wavenumber", "amplitude", "phase"}, path);
    FourierModeConfig parsed;
    parsed.wavenumber = checkedInt(signedIntegerMember(mode, "wavenumber", path), path + ".wavenumber");
    parsed.amplitude = doubleMember(mode, "amplitude", path);
    parsed.phase = doubleMember(mode, "phase", path);
    config.forcing.deterministic.modes.push_back(parsed);
  }

  const Json& stochastic = objectMember(forcing, "stochastic", "configuration.forcing");
  requireKeys(stochastic,
              {"wavenumbers", "spectral_exponent", "stationary_rms",
               "correlation_time", "clock_interval", "clock_reference_time"},
              "configuration.forcing.stochastic");
  const Json& wavenumbers = member(stochastic, "wavenumbers", "configuration.forcing.stochastic");
  if(!wavenumbers.is_array()) {
    throw configError("configuration.forcing.stochastic.wavenumbers must be an array");
  }
  config.forcing.stochastic.wavenumbers.clear();
  for(std::size_t index = 0; index < wavenumbers.size(); ++index) {
    const Json& mode = wavenumbers[index];
    const std::string path =
      "configuration.forcing.stochastic.wavenumbers[" +
      std::to_string(index) + "]";
    if(!mode.is_number_integer()) {
      throw configError("configuration.forcing.stochastic.wavenumbers[" +
                        std::to_string(index) + "] must be an integer");
    }
    if(mode.is_number_unsigned()) {
      const std::uint64_t value = mode.get<std::uint64_t>();
      if(value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw configError(path + " is outside the supported integer range");
      }
      config.forcing.stochastic.wavenumbers.push_back(
        static_cast<int>(value));
    } else {
      config.forcing.stochastic.wavenumbers.push_back(
        checkedInt(mode.get<std::int64_t>(), path));
    }
  }
  config.forcing.stochastic.spectral_exponent = doubleMember(stochastic, "spectral_exponent", "configuration.forcing.stochastic");
  config.forcing.stochastic.stationary_rms = doubleMember(stochastic, "stationary_rms", "configuration.forcing.stochastic");
  config.forcing.stochastic.correlation_time = doubleMember(stochastic, "correlation_time", "configuration.forcing.stochastic");
  config.forcing.stochastic.clock_interval = doubleMember(stochastic, "clock_interval", "configuration.forcing.stochastic");
  config.forcing.stochastic.clock_reference_time = doubleMember(stochastic, "clock_reference_time", "configuration.forcing.stochastic");

  const Json& closure = objectMember(root, "closure", "configuration");
  requireKeys(closure,
              {"type", "static_coefficient", "minimum_coefficient",
               "maximum_coefficient", "test_filter_ratio",
               "denominator_regularization", "store_squared_coefficient"},
              "configuration.closure");
  config.closure.type = parseClosureType(
    stringMember(closure, "type", "configuration.closure"),
    "configuration.closure.type");
  config.closure.static_coefficient = doubleMember(closure, "static_coefficient", "configuration.closure");
  config.closure.minimum_coefficient = doubleMember(closure, "minimum_coefficient", "configuration.closure");
  config.closure.maximum_coefficient = doubleMember(closure, "maximum_coefficient", "configuration.closure");
  config.closure.test_filter_ratio = doubleMember(closure, "test_filter_ratio", "configuration.closure");
  config.closure.denominator_regularization = doubleMember(closure, "denominator_regularization", "configuration.closure");
  config.closure.store_squared_coefficient = boolMember(closure, "store_squared_coefficient", "configuration.closure");

  const Json& output = objectMember(root, "output", "configuration");
  requireKeys(output,
              {"directory", "metadata_filename", "scalar_history_filename",
               "final_profile_filename", "profile_history_filename",
               "spectrum_filename", "history_interval", "profile_interval",
               "write_initial_profile", "write_profile_history",
               "write_online_spectrum", "statistics_start_time"},
              "configuration.output");
  config.output.directory = stringMember(output, "directory", "configuration.output");
  config.output.metadata_filename = stringMember(output, "metadata_filename", "configuration.output");
  config.output.scalar_history_filename = stringMember(output, "scalar_history_filename", "configuration.output");
  config.output.final_profile_filename = stringMember(output, "final_profile_filename", "configuration.output");
  config.output.profile_history_filename = stringMember(output, "profile_history_filename", "configuration.output");
  config.output.spectrum_filename = stringMember(output, "spectrum_filename", "configuration.output");
  config.output.history_interval = doubleMember(output, "history_interval", "configuration.output");
  config.output.profile_interval = doubleMember(output, "profile_interval", "configuration.output");
  config.output.write_initial_profile = boolMember(output, "write_initial_profile", "configuration.output");
  config.output.write_profile_history = boolMember(output, "write_profile_history", "configuration.output");
  config.output.write_online_spectrum = boolMember(output, "write_online_spectrum", "configuration.output");
  config.output.statistics_start_time = doubleMember(output, "statistics_start_time", "configuration.output");

  const Json& random = objectMember(root, "random", "configuration");
  requireKeys(random, {"seed"}, "configuration.random");
  config.random.seed = unsignedIntegerMember(random, "seed", "configuration.random");
}

Json configurationJson(const RunConfig& config) {
  Json root;
  root["schema_version"] = config_schema_version;
  Json& value = root["configuration"];
  value["grid"] = {{"x_begin", config.grid.x_begin},
                   {"x_end", config.grid.x_end},
                   {"cell_count", config.grid.cell_count}};
  value["initial_condition"] = {
    {"type", toString(config.initial_condition.type)},
    {"mean", config.initial_condition.mean},
    {"amplitude", config.initial_condition.amplitude},
    {"wavenumber", config.initial_condition.wavenumber},
    {"phase", config.initial_condition.phase},
    {"left_state", config.initial_condition.left_state},
    {"right_state", config.initial_condition.right_state}};
  value["viscosity"] = {
    {"molecular", config.viscosity.molecular},
    {"face_averaging", toString(config.viscosity.face_averaging)}};
  value["numerical_method"] = {
    {"reconstruction", toString(config.numerical_method.reconstruction)},
    {"convective_flux", toString(config.numerical_method.convective_flux)},
    {"limiter", toString(config.numerical_method.limiter)}};
  value["time_integration"] = {
    {"integrator", toString(config.time_integration.integrator)},
    {"initial_time", config.time_integration.initial_time},
    {"final_time", config.time_integration.final_time},
    {"advective_cfl", config.time_integration.advective_cfl},
    {"diffusive_cfl", config.time_integration.diffusive_cfl},
    {"maximum_steps", config.time_integration.maximum_steps}};

  Json deterministic_modes = Json::array();
  for(const FourierModeConfig& mode : config.forcing.deterministic.modes) {
    deterministic_modes.push_back({{"wavenumber", mode.wavenumber},
                                   {"amplitude", mode.amplitude},
                                   {"phase", mode.phase}});
  }
  value["forcing"] = {
    {"type", toString(config.forcing.type)},
    {"remove_discrete_mean", config.forcing.remove_discrete_mean},
    {"manufactured", {
      {"amplitude", config.forcing.manufactured.amplitude},
      {"wavenumber", config.forcing.manufactured.wavenumber},
      {"phase", config.forcing.manufactured.phase},
      {"decay_rate", config.forcing.manufactured.decay_rate}}},
    {"deterministic", {{"modes", deterministic_modes}}},
    {"stochastic", {
      {"wavenumbers", config.forcing.stochastic.wavenumbers},
      {"spectral_exponent", config.forcing.stochastic.spectral_exponent},
      {"stationary_rms", config.forcing.stochastic.stationary_rms},
      {"correlation_time", config.forcing.stochastic.correlation_time},
      {"clock_interval", config.forcing.stochastic.clock_interval},
      {"clock_reference_time", config.forcing.stochastic.clock_reference_time}}}};
  value["closure"] = {
    {"type", toString(config.closure.type)},
    {"static_coefficient", config.closure.static_coefficient},
    {"minimum_coefficient", config.closure.minimum_coefficient},
    {"maximum_coefficient", config.closure.maximum_coefficient},
    {"test_filter_ratio", config.closure.test_filter_ratio},
    {"denominator_regularization", config.closure.denominator_regularization},
    {"store_squared_coefficient", config.closure.store_squared_coefficient}};
  value["output"] = {
    {"directory", config.output.directory},
    {"metadata_filename", config.output.metadata_filename},
    {"scalar_history_filename", config.output.scalar_history_filename},
    {"final_profile_filename", config.output.final_profile_filename},
    {"profile_history_filename", config.output.profile_history_filename},
    {"spectrum_filename", config.output.spectrum_filename},
    {"history_interval", config.output.history_interval},
    {"profile_interval", config.output.profile_interval},
    {"write_initial_profile", config.output.write_initial_profile},
    {"write_profile_history", config.output.write_profile_history},
    {"write_online_spectrum", config.output.write_online_spectrum},
    {"statistics_start_time", config.output.statistics_start_time}};
  value["random"] = {{"seed", config.random.seed}};
  return root;
}

}  // namespace

RunConfig loadRunConfig(const std::string& path) {
  std::ifstream input(path);
  if(!input) {
    throw std::runtime_error("unable to open run configuration: " + path);
  }

  Json document;
  try {
    input >> document;
  } catch(const Json::exception& error) {
    throw configError(std::string("unable to parse JSON: ") + error.what());
  }
  requireObject(document, "document");
  requireKeys(document, {"schema_version", "configuration"}, "document");
  const std::uint64_t schema = unsignedIntegerMember(
    document, "schema_version", "document");
  if(schema != config_schema_version) {
    throw configError("document.schema_version must be " +
                      std::to_string(config_schema_version));
  }
  const Json& configuration = objectMember(document, "configuration", "document");
  requireKeys(configuration,
              {"grid", "initial_condition", "viscosity", "numerical_method",
               "time_integration", "forcing", "closure", "output", "random"},
              "configuration");

  RunConfig config;
  readConfiguration(configuration, config);
  const std::vector<std::string> errors = validate(config);
  if(!errors.empty()) {
    std::ostringstream message;
    for(std::size_t index = 0; index < errors.size(); ++index) {
      if(index != 0) message << "; ";
      message << errors[index];
    }
    throw configError(message.str());
  }
  return config;
}

std::string serializeRunConfig(const RunConfig& config) {
  return configurationJson(config).dump(2) + "\n";
}

}  // namespace burgers
