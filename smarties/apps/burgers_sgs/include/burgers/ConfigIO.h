#pragma once

#include "burgers/Config.h"

#include <string>

namespace burgers {

// Load a complete, versioned RunConfig document. The loader is deliberately
// strict: every field must be present and unknown fields are rejected.
RunConfig loadRunConfig(const std::string& path);

// Serialize the complete configuration document accepted by loadRunConfig().
std::string serializeRunConfig(const RunConfig& config);

}  // namespace burgers
