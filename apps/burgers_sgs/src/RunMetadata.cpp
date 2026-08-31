#include "burgers/RunMetadata.h"

#include "burgers/BuildInfo.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

#ifndef BURGERS_BUILD_CONFIGURATION
#define BURGERS_BUILD_CONFIGURATION "unknown"
#endif

namespace burgers {
namespace {

std::string jsonString(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for(const unsigned char character : value) {
    switch(character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if(character < 0x20u) {
          output << "\\u00" << std::hex << std::setw(2)
                 << std::setfill('0') << static_cast<unsigned int>(character)
                 << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
  return output.str();
}

const char* jsonBool(bool value) {
  return value ? "true" : "false";
}

std::string utcTimestamp() {
  const std::time_t now = std::time(nullptr);
  const std::tm* utc = std::gmtime(&now);
  if(utc == nullptr) {
    return "unknown";
  }

  std::ostringstream output;
  output << std::put_time(utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

void writeFourierModes(std::ostream& output,
                       const std::vector<FourierModeConfig>& modes) {
  output << '[';
  for(std::size_t index = 0; index < modes.size(); ++index) {
    if(index != 0) {
      output << ',';
    }
    output << "{\"wavenumber\":" << modes[index].wavenumber
           << ",\"amplitude\":" << modes[index].amplitude
           << ",\"phase\":" << modes[index].phase << '}';
  }
  output << ']';
}

void writeWavenumbers(std::ostream& output,
                      const std::vector<int>& wavenumbers) {
  output << '[';
  for(std::size_t index = 0; index < wavenumbers.size(); ++index) {
    if(index != 0) {
      output << ',';
    }
    output << wavenumbers[index];
  }
  output << ']';
}

}  // namespace

BuildMetadata currentBuildMetadata() {
  BuildMetadata metadata;
  metadata.project_version = BURGERS_PROJECT_VERSION;
  metadata.compiler_id = BURGERS_COMPILER_ID;
  metadata.compiler_version = BURGERS_COMPILER_VERSION;
  metadata.build_configuration = BURGERS_BUILD_CONFIGURATION;
  metadata.cmake_version = BURGERS_CMAKE_VERSION;
  metadata.cmake_generator = BURGERS_CMAKE_GENERATOR;
  metadata.source_revision = BURGERS_GIT_REVISION;
  metadata.source_dirty = BURGERS_SOURCE_DIRTY;
#ifdef NDEBUG
  metadata.assertions_enabled = false;
#else
  metadata.assertions_enabled = true;
#endif
#ifdef BURGERS_SANITIZERS_ENABLED
  metadata.sanitizers_enabled = true;
#else
  metadata.sanitizers_enabled = false;
#endif
  return metadata;
}

const char* toString(RunStatus value) {
  switch(value) {
    case RunStatus::ScaffoldComplete: return "scaffold_complete";
    case RunStatus::Completed: return "completed";
    case RunStatus::InvalidConfiguration: return "invalid_configuration";
    case RunStatus::Failed: return "failed";
  }
  return "unknown";
}

std::string metadataFilePath(const OutputConfig& output) {
  if(output.directory.empty() || output.directory == ".") {
    return std::string("./") + output.metadata_filename;
  }
  const char last = output.directory.back();
  if(last == '/' || last == '\\') {
    return output.directory + output.metadata_filename;
  }
  return output.directory + '/' + output.metadata_filename;
}

std::string serializeRunMetadata(const RunConfig& config,
                                 const RunResultMetadata& result) {
  const BuildMetadata build = currentBuildMetadata();
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);

  output << "{\n"
         << "  \"schema_version\": 6,\n"
         << "  \"created_utc\": " << jsonString(utcTimestamp()) << ",\n"
         << "  \"phase\": " << result.phase << ",\n"
         << "  \"build\": {\n"
         << "    \"project_version\": "
         << jsonString(build.project_version) << ",\n"
         << "    \"compiler_id\": " << jsonString(build.compiler_id)
         << ",\n"
         << "    \"compiler_version\": "
         << jsonString(build.compiler_version) << ",\n"
         << "    \"build_configuration\": "
         << jsonString(build.build_configuration) << ",\n"
         << "    \"cmake_version\": " << jsonString(build.cmake_version)
         << ",\n"
         << "    \"cmake_generator\": "
         << jsonString(build.cmake_generator) << ",\n"
         << "    \"source_revision\": "
         << jsonString(build.source_revision) << ",\n"
         << "    \"source_dirty\": " << jsonString(build.source_dirty)
         << ",\n"
         << "    \"assertions_enabled\": "
         << jsonBool(build.assertions_enabled) << ",\n"
         << "    \"sanitizers_enabled\": "
         << jsonBool(build.sanitizers_enabled) << ",\n"
         << "    \"smarties_linked\": false\n"
         << "  },\n"
         << "  \"configuration\": {\n"
         << "    \"grid\": {\"x_begin\":" << config.grid.x_begin
         << ",\"x_end\":" << config.grid.x_end
         << ",\"cell_count\":" << config.grid.cell_count << "},\n"
         << "    \"initial_condition\": {\"type\":"
         << jsonString(toString(config.initial_condition.type))
         << ",\"mean\":" << config.initial_condition.mean
         << ",\"amplitude\":" << config.initial_condition.amplitude
         << ",\"wavenumber\":" << config.initial_condition.wavenumber
         << ",\"phase\":" << config.initial_condition.phase
         << ",\"left_state\":" << config.initial_condition.left_state
         << ",\"right_state\":" << config.initial_condition.right_state
         << "},\n"
         << "    \"viscosity\": {\"molecular\":"
         << config.viscosity.molecular << ",\"face_averaging\":"
         << jsonString(toString(config.viscosity.face_averaging)) << "},\n"
         << "    \"numerical_method\": {\"reconstruction\":"
         << jsonString(toString(config.numerical_method.reconstruction))
         << ",\"convective_flux\":"
         << jsonString(toString(config.numerical_method.convective_flux))
         << ",\"limiter\":"
         << jsonString(toString(config.numerical_method.limiter)) << "},\n"
         << "    \"time_integration\": {\"integrator\":"
         << jsonString(toString(config.time_integration.integrator))
         << ",\"initial_time\":" << config.time_integration.initial_time
         << ",\"final_time\":" << config.time_integration.final_time
         << ",\"advective_cfl\":"
         << config.time_integration.advective_cfl
         << ",\"diffusive_cfl\":"
         << config.time_integration.diffusive_cfl
         << ",\"maximum_steps\":"
         << config.time_integration.maximum_steps << "},\n"
         << "    \"forcing\": {\"type\":"
         << jsonString(toString(config.forcing.type))
         << ",\"remove_discrete_mean\":"
         << jsonBool(config.forcing.remove_discrete_mean)
         << ",\"deterministic_modes\":";
  writeFourierModes(output, config.forcing.deterministic.modes);
  output << ",\"stochastic_wavenumbers\":";
  writeWavenumbers(output, config.forcing.stochastic.wavenumbers);
  output << ",\"manufactured_amplitude\":"
         << config.forcing.manufactured.amplitude
         << ",\"manufactured_wavenumber\":"
         << config.forcing.manufactured.wavenumber
         << ",\"manufactured_phase\":"
         << config.forcing.manufactured.phase
         << ",\"manufactured_decay_rate\":"
         << config.forcing.manufactured.decay_rate
         << ",\"stochastic_spectral_exponent\":"
         << config.forcing.stochastic.spectral_exponent
         << ",\"stochastic_stationary_rms\":"
         << config.forcing.stochastic.stationary_rms
         << ",\"stochastic_correlation_time\":"
         << config.forcing.stochastic.correlation_time
         << ",\"stochastic_clock_interval\":"
         << config.forcing.stochastic.clock_interval
         << ",\"stochastic_clock_reference_time\":"
         << config.forcing.stochastic.clock_reference_time << "},\n"
         << "    \"closure\": {\"type\":"
         << jsonString(toString(config.closure.type))
         << ",\"static_coefficient\":"
         << config.closure.static_coefficient
         << ",\"minimum_coefficient\":"
         << config.closure.minimum_coefficient
         << ",\"maximum_coefficient\":"
         << config.closure.maximum_coefficient
         << ",\"test_filter_ratio\":"
         << config.closure.test_filter_ratio
         << ",\"denominator_regularization\":"
         << config.closure.denominator_regularization
         << ",\"store_squared_coefficient\":"
         << jsonBool(config.closure.store_squared_coefficient) << "},\n"
         << "    \"output\": {\"directory\":"
         << jsonString(config.output.directory)
         << ",\"metadata_filename\":"
         << jsonString(config.output.metadata_filename)
         << ",\"scalar_history_filename\":"
         << jsonString(config.output.scalar_history_filename)
         << ",\"final_profile_filename\":"
         << jsonString(config.output.final_profile_filename)
         << ",\"profile_history_filename\":"
         << jsonString(config.output.profile_history_filename)
         << ",\"spectrum_filename\":"
         << jsonString(config.output.spectrum_filename)
         << ",\"history_interval\":" << config.output.history_interval
         << ",\"profile_interval\":" << config.output.profile_interval
         << ",\"write_initial_profile\":"
         << jsonBool(config.output.write_initial_profile)
         << ",\"write_profile_history\":"
         << jsonBool(config.output.write_profile_history)
         << ",\"write_online_spectrum\":"
         << jsonBool(config.output.write_online_spectrum)
         << ",\"statistics_start_time\":"
         << config.output.statistics_start_time
         << "},\n"
         << "    \"random\": {\"seed\":" << config.random.seed << "}\n"
         << "  },\n"
         << "  \"result\": {\n"
         << "    \"status\": " << jsonString(toString(result.status))
         << ",\n"
         << "    \"message\": " << jsonString(result.message) << ",\n"
         << "    \"numerical_advancement_performed\": "
         << jsonBool(result.numerical_advancement_performed) << ",\n"
         << "    \"final_time\": " << result.final_time << ",\n"
         << "    \"timestep_count\": " << result.timestep_count << ",\n"
         << "    \"rejected_step_count\": "
         << result.rejected_step_count << ",\n"
         << "    \"shortened_final_step_count\": "
         << result.shortened_final_step_count << ",\n"
         << "    \"forcing_clock_step_count\": "
         << result.forcing_clock_step_count << ",\n"
         << "    \"statistics_sample_count\": "
         << result.statistics_sample_count << ",\n"
         << "    \"deterministic_work\": "
         << result.deterministic_work << ",\n"
         << "    \"stochastic_work\": "
         << result.stochastic_work << ",\n"
         << "    \"manufactured_work\": "
         << result.manufactured_work << ",\n"
         << "    \"molecular_dissipation\": "
         << result.molecular_dissipation << ",\n"
         << "    \"sgs_dissipation\": "
         << result.sgs_dissipation << ",\n"
         << "    \"numerical_dissipation\": "
         << result.numerical_dissipation << ",\n"
         << "    \"energy_change\": "
         << result.energy_change << "\n"
         << "  }\n"
         << "}\n";
  return output.str();
}

void writeRunMetadata(const std::string& path,
                      const RunConfig& config,
                      const RunResultMetadata& result) {
  std::ofstream output(path);
  if(!output) {
    throw std::runtime_error("unable to open metadata file: " + path);
  }

  output << serializeRunMetadata(config, result);
  if(!output) {
    throw std::runtime_error("unable to write metadata file: " + path);
  }
}

}  // namespace burgers
