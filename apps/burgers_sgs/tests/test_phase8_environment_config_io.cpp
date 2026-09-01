#include "burgers/environment/EnvironmentConfigIO.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if(!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template<typename Function>
void expectInvalid(Function function, const std::string& message) {
  try {
    function();
  } catch(const std::invalid_argument&) {
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

void writeText(const std::string& path, const std::string& text) {
  std::ofstream output(path);
  output << text;
  if(!output) throw std::runtime_error("unable to write test RL config");
}

std::string replaceOnce(std::string text,
                        const std::string& original,
                        const std::string& replacement) {
  const std::size_t location = text.find(original);
  if(location == std::string::npos) {
    throw std::runtime_error("test JSON fragment was not found");
  }
  text.replace(location, original.size(), replacement);
  return text;
}

}  // namespace

int main() {
  const std::string example =
    std::string(BURGERS_TEST_SOURCE_DIR) + "/configs/burgers_rl.json";
  const std::string temporary = "burgers_test_rl_config.json";
  try {
    const burgers::environment::EnvironmentApplicationConfig config =
      burgers::environment::loadEnvironmentApplicationConfig(example);
    expect(config.environment.solver.grid.cell_count == 64u,
           "RL config must load its referenced solver configuration");
    expect(config.environment.observation_radius == 1u,
           "RL config must preserve the observation radius");
    expect(config.environment.action.forcing_scale == 0.5,
           "RL config must preserve the action scale");
    expect(config.evaluation_seeds.size() == 4u,
           "RL config must preserve the held-out seed list");
    expect(config.solver_config_path.find("configs/les_test.json") !=
             std::string::npos,
           "solver path must resolve relative to the RL document");
    expect(config.environment.target_metadata_path.find(
             "configs/dns_target_metadata.json") !=
             std::string::npos,
           "target path must resolve relative to the RL document");

    const std::string serialized =
      burgers::environment::serializeEnvironmentApplicationConfig(config);
    writeText(temporary, serialized);
    const burgers::environment::EnvironmentApplicationConfig restored =
      burgers::environment::loadEnvironmentApplicationConfig(temporary);
    expect(restored.environment.decision_interval == 0.1,
           "resolved RL configuration must round trip");

    const std::string unknown = replaceOnce(
      serialized, "\"environment\": {",
      "\"environment\": {\n    \"mystery\": 1,");
    writeText(temporary, unknown);
    expectInvalid(
      [&temporary]() {
        burgers::environment::loadEnvironmentApplicationConfig(temporary);
      },
      "strict RL loader must reject unknown fields");
  } catch(const std::exception& error) {
    std::cerr << "FAIL: unexpected RL configuration exception: "
              << error.what() << '\n';
    ++failures;
  }
  std::remove(temporary.c_str());

  if(failures != 0) {
    std::cerr << failures << " environment-config check(s) failed\n";
    return 1;
  }
  std::cout << "Phase 8 environment-config checks passed\n";
  return 0;
}
