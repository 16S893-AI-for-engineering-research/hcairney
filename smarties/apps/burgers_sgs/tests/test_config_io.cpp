#include "burgers/Config.h"
#include "burgers/ConfigIO.h"

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
  if(!output) {
    throw std::runtime_error("unable to write test configuration");
  }
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
  const std::string path = "burgers_test_config_io.json";
  burgers::RunConfig original = burgers::makePhase5ValidationRunConfig();
  original.grid.cell_count = 512;
  original.output.profile_interval = 0.2;
  original.output.write_online_spectrum = false;
  original.closure.type = burgers::ClosureType::StaticSmagorinsky;
  original.closure.static_coefficient = 0.27;
  original.random.seed = 982451653u;

  try {
    const std::string serialized = burgers::serializeRunConfig(original);
    writeText(path, serialized);
    const burgers::RunConfig restored = burgers::loadRunConfig(path);
    expect(restored.grid.cell_count == original.grid.cell_count,
           "configuration round trip must preserve cell count");
    expect(restored.output.profile_interval ==
             original.output.profile_interval,
           "configuration round trip must preserve profile interval");
    expect(!restored.output.write_online_spectrum,
           "configuration round trip must preserve spectrum output switch");
    expect(restored.closure.type == burgers::ClosureType::StaticSmagorinsky,
           "configuration round trip must preserve closure type");
    expect(restored.closure.static_coefficient == 0.27,
           "configuration round trip must preserve C_S");
    expect(restored.closure.store_squared_coefficient,
           "configuration round trip must preserve the internal C_S-squared "
           "representation");
    expect(restored.random.seed == original.random.seed,
           "configuration round trip must preserve the full seed");

    const std::string unknown = replaceOnce(
      serialized, "\"grid\": {", "\"grid\": {\n      \"mystery\": 1,");
    writeText(path, unknown);
    expectInvalid([&path]() { burgers::loadRunConfig(path); },
                  "strict loader must reject unknown fields");

    const std::string wrong_type = replaceOnce(
      serialized, "\"cell_count\": 512", "\"cell_count\": \"512\"");
    writeText(path, wrong_type);
    expectInvalid([&path]() { burgers::loadRunConfig(path); },
                  "strict loader must reject incorrect JSON types");

    const std::string wrong_schema = replaceOnce(
      serialized, "\"schema_version\": 1", "\"schema_version\": 99");
    writeText(path, wrong_schema);
    expectInvalid([&path]() { burgers::loadRunConfig(path); },
                  "strict loader must reject unsupported schema versions");
  } catch(const std::exception& error) {
    std::cerr << "FAIL: unexpected configuration I/O exception: "
              << error.what() << '\n';
    ++failures;
  }
  std::remove(path.c_str());

  if(failures != 0) {
    std::cerr << failures << " configuration I/O check(s) failed\n";
    return 1;
  }
  std::cout << "Configuration I/O checks passed\n";
  return 0;
}
