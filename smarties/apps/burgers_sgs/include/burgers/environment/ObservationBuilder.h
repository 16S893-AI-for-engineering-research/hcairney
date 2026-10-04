#pragma once

#include "burgers/Grid.h"
#include "burgers/State.h"

#include <cstddef>
#include <vector>

namespace burgers {
namespace environment {

class ObservationBuilder {
public:
  ObservationBuilder(const Grid& grid,
                     double molecular_viscosity,
                     std::size_t stencil_radius);

  std::size_t stencilRadius() const noexcept;
  std::size_t observationSize() const noexcept;

  std::vector<double> featureField(const State& velocity) const;
  std::vector<std::vector<double>> build(const State& velocity) const;

private:
  Grid grid_;
  double molecular_viscosity_;
  std::size_t stencil_radius_;
};

}  // namespace environment
}  // namespace burgers
