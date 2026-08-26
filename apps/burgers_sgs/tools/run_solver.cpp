#include "burgers/Config.h"
#include "burgers/RunMetadata.h"

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string joinErrors(const std::vector<std::string>& errors) {
  std::ostringstream message;
  for(std::size_t index = 0; index < errors.size(); ++index) {
    if(index != 0) {
      message << "; ";
    }
    message << errors[index];
  }
  return message.str();
}

}  // namespace

int main() {
  const burgers::RunConfig config = burgers::makeDefaultRunConfig();
  burgers::RunResultMetadata result;
  const std::vector<std::string> errors = burgers::validate(config);

  if(!errors.empty()) {
    result.status = burgers::RunStatus::InvalidConfiguration;
    result.message = joinErrors(errors);
  }

  const std::string metadata_path = burgers::metadataFilePath(config.output);
  try {
    burgers::writeRunMetadata(metadata_path, config, result);
  } catch(const std::exception& error) {
    std::cerr << "burgers_sgs: " << error.what() << '\n';
    return 2;
  }

  if(!errors.empty()) {
    std::cerr << "burgers_sgs: invalid configuration: "
              << result.message << '\n'
              << "Metadata written to " << metadata_path << '\n';
    return 1;
  }

  std::cout << "burgers_sgs Phase 0 scaffold is operational.\n"
            << "No PDE was advanced. Metadata written to "
            << metadata_path << '\n';
  return 0;
}

