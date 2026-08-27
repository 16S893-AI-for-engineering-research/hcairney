#include "burgers/Config.h"
#include "burgers/RunMetadata.h"

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

}  // namespace

int main() {
  const burgers::RunConfig config = burgers::makeDefaultRunConfig();
  const std::vector<std::string> errors = burgers::validate(config);
  expect(errors.empty(), "default configuration must be valid");

  expect(std::string(burgers::toString(config.numerical_method.reconstruction))
           == "piecewise_constant",
         "default reconstruction must be piecewise constant");
  expect(std::string(burgers::toString(config.numerical_method.convective_flux))
           == "godunov",
         "default convective flux must be Godunov");
  expect(std::string(burgers::toString(config.closure.type)) == "no_closure",
         "default closure must be disabled");
  expect(std::string(burgers::toString(config.forcing.type)) == "none",
         "default forcing must be disabled");

  const burgers::RunResultMetadata result;
  const std::string metadata = burgers::serializeRunMetadata(config, result);
  expect(metadata.find("\"schema_version\": 2") != std::string::npos,
         "metadata must contain its schema version");
  expect(metadata.find("\"smarties_linked\": false") != std::string::npos,
         "metadata must document SMARTIES independence");
  expect(metadata.find("\"seed\":5489") != std::string::npos,
         "metadata must contain the random seed");
  expect(metadata.find("\"numerical_advancement_performed\": false") !=
           std::string::npos,
         "metadata must identify the Phase 0 no-op run");
  expect(burgers::metadataFilePath(config.output) ==
           "./burgers_run_metadata.json",
         "default metadata path must be reproducible");

  if(failures != 0) {
    std::cerr << failures << " Phase 0 scaffolding check(s) failed\n";
    return 1;
  }

  std::cout << "Phase 0 scaffolding checks passed\n";
  return 0;
}
