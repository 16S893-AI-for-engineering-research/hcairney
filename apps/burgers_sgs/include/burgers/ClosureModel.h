#pragma once

#include "burgers/Config.h"
#include "burgers/Grid.h"
#include "burgers/State.h"

#include <memory>
#include <vector>

namespace burgers {

// Coefficients are represented internally as C_S^2. Public configuration and
// prescribed-field setters use C_S so metadata remains directly comparable to
// the conventional static Smagorinsky coefficient.
struct ClosureFields {
  explicit ClosureFields(const Grid& grid);

  State coefficient_squared;
  State eddy_viscosity;
};

class ClosureModel {
public:
  virtual ~ClosureModel() = default;

  virtual ClosureType type() const noexcept = 0;
  virtual void evaluate(const Grid& grid,
                        const State& velocity,
                        ClosureFields& fields) const = 0;

  // The default implementation rejects the operation. Only the prescribed
  // closure accepts a field, and it holds that field until it is replaced.
  virtual void setCoefficientField(const std::vector<double>& coefficients);
};

void computeCellCenteredGradients(const Grid& grid,
                                  const State& velocity,
                                  State& gradients);

std::unique_ptr<ClosureModel> makeClosureModel(
  const Grid& grid,
  const ClosureConfig& config);

}  // namespace burgers
