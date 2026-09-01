#pragma once

#include "burgers/Grid.h"

#include <cstddef>
#include <string>
#include <vector>

namespace burgers {
namespace environment {

class TargetProfile {
public:
  static TargetProfile loadAccepted(const std::string& metadata_path);

  const std::string& metadataPath() const noexcept;
  const std::string& profileHash() const noexcept;
  const std::string& spectrumHash() const noexcept;
  std::size_t cellCount() const noexcept;
  double xBegin() const noexcept;
  double xEnd() const noexcept;
  const std::vector<double>& meanVelocity() const noexcept;

  std::vector<double> restrictMeanVelocity(const Grid& grid) const;

private:
  std::string metadata_path_;
  std::string profile_hash_;
  std::string spectrum_hash_;
  double x_begin_ = 0.0;
  double x_end_ = 0.0;
  std::vector<double> cell_centers_;
  std::vector<double> mean_velocity_;
};

}  // namespace environment
}  // namespace burgers
