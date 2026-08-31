#include "burgers/ClosureModel.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace burgers {
namespace {

void requireCompatible(const Grid& grid,
                       const State& state,
                       const char* name) {
  if(state.size() != grid.cellCount()) {
    throw std::invalid_argument(std::string(name) +
                                " size does not match the grid");
  }
}

void evaluateFromCoefficientSquared(const Grid& grid,
                                    const State& velocity,
                                    const State& coefficient_squared,
                                    ClosureFields& fields) {
  requireCompatible(grid, velocity, "closure velocity");
  requireCompatible(grid, coefficient_squared,
                    "closure coefficient field");
  requireCompatible(grid, fields.coefficient_squared,
                    "closure output coefficient field");
  requireCompatible(grid, fields.eddy_viscosity,
                    "closure output eddy-viscosity field");

  State gradients(grid);
  computeCellCenteredGradients(grid, velocity, gradients);
  const double delta_squared = grid.cellWidth() * grid.cellWidth();
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const double coefficient = coefficient_squared[cell];
    if(std::isfinite(coefficient) == 0 || coefficient < 0.0) {
      throw std::runtime_error(
        "closure coefficient-squared field contains an invalid value at cell " +
        std::to_string(cell));
    }
    const double eddy_viscosity =
      coefficient * delta_squared * std::abs(gradients[cell]);
    if(std::isfinite(eddy_viscosity) == 0) {
      throw std::runtime_error(
        "closure produced a non-finite eddy viscosity at cell " +
        std::to_string(cell));
    }
    fields.coefficient_squared[cell] = coefficient;
    fields.eddy_viscosity[cell] = eddy_viscosity;
  }
}

class NoClosure final : public ClosureModel {
public:
  ClosureType type() const noexcept override {
    return ClosureType::NoClosure;
  }

  void evaluate(const Grid& grid,
                const State& velocity,
                ClosureFields& fields) const override {
    requireCompatible(grid, velocity, "closure velocity");
    requireCompatible(grid, fields.coefficient_squared,
                      "closure output coefficient field");
    requireCompatible(grid, fields.eddy_viscosity,
                      "closure output eddy-viscosity field");
    fields.coefficient_squared.fill(0.0);
    fields.eddy_viscosity.fill(0.0);
  }
};

class StaticSmagorinsky final : public ClosureModel {
public:
  StaticSmagorinsky(const Grid& grid, double coefficient)
    : coefficient_squared_(grid, coefficient * coefficient) {
    if(std::isfinite(coefficient) == 0 || coefficient < 0.0 ||
       std::isfinite(coefficient * coefficient) == 0) {
      throw std::invalid_argument(
        "static Smagorinsky coefficient must be finite, nonnegative, and "
        "safely representable when squared");
    }
  }

  ClosureType type() const noexcept override {
    return ClosureType::StaticSmagorinsky;
  }

  void evaluate(const Grid& grid,
                const State& velocity,
                ClosureFields& fields) const override {
    evaluateFromCoefficientSquared(
      grid, velocity, coefficient_squared_, fields);
  }

private:
  State coefficient_squared_;
};

class PrescribedCoefficientField final : public ClosureModel {
public:
  PrescribedCoefficientField(const Grid& grid,
                             double minimum_coefficient,
                             double maximum_coefficient)
    : coefficient_squared_(grid, 0.0),
      minimum_coefficient_(minimum_coefficient),
      maximum_coefficient_(maximum_coefficient),
      field_is_set_(false) {}

  ClosureType type() const noexcept override {
    return ClosureType::PrescribedCoefficientField;
  }

  void setCoefficientField(
    const std::vector<double>& coefficients) override {
    if(coefficients.size() != coefficient_squared_.size()) {
      throw std::invalid_argument(
        "prescribed coefficient field size does not match the solver grid");
    }
    std::vector<double> squared_coefficients(coefficients.size(), 0.0);
    for(std::size_t cell = 0; cell < coefficients.size(); ++cell) {
      const double coefficient = coefficients[cell];
      if(std::isfinite(coefficient) == 0 ||
         coefficient < minimum_coefficient_ ||
         coefficient > maximum_coefficient_) {
        throw std::invalid_argument(
          "prescribed coefficient at cell " + std::to_string(cell) +
          " must be finite and lie within the configured C_S bounds");
      }
      const double squared = coefficient * coefficient;
      if(std::isfinite(squared) == 0) {
        throw std::invalid_argument(
          "prescribed coefficient at cell " + std::to_string(cell) +
          " is not safely representable when squared");
      }
      squared_coefficients[cell] = squared;
    }
    coefficient_squared_.cellAverages() = std::move(squared_coefficients);
    field_is_set_ = true;
  }

  void evaluate(const Grid& grid,
                const State& velocity,
                ClosureFields& fields) const override {
    if(!field_is_set_) {
      throw std::runtime_error(
        "prescribed coefficient field must be set before evaluating the "
        "closure or advancing the solver");
    }
    evaluateFromCoefficientSquared(
      grid, velocity, coefficient_squared_, fields);
  }

private:
  State coefficient_squared_;
  double minimum_coefficient_;
  double maximum_coefficient_;
  bool field_is_set_;
};

}  // namespace

ClosureFields::ClosureFields(const Grid& grid)
  : coefficient_squared(grid, 0.0), eddy_viscosity(grid, 0.0) {}

void ClosureModel::setCoefficientField(
  const std::vector<double>& coefficients) {
  static_cast<void>(coefficients);
  throw std::logic_error(
    "coefficient fields can only be set on prescribed_coefficient_field");
}

void computeCellCenteredGradients(const Grid& grid,
                                  const State& velocity,
                                  State& gradients) {
  requireCompatible(grid, velocity, "gradient input");
  requireCompatible(grid, gradients, "gradient output");
  const double inverse_two_width = 0.5 / grid.cellWidth();
  for(std::size_t cell = 0; cell < grid.cellCount(); ++cell) {
    const std::size_t left = grid.neighbor(cell, -1);
    const std::size_t right = grid.neighbor(cell, 1);
    gradients[cell] =
      (velocity[right] - velocity[left]) * inverse_two_width;
  }
}

std::unique_ptr<ClosureModel> makeClosureModel(
  const Grid& grid,
  const ClosureConfig& config) {
  switch(config.type) {
    case ClosureType::NoClosure:
      return std::unique_ptr<ClosureModel>(new NoClosure());
    case ClosureType::StaticSmagorinsky:
      return std::unique_ptr<ClosureModel>(
        new StaticSmagorinsky(grid, config.static_coefficient));
    case ClosureType::PrescribedCoefficientField:
      return std::unique_ptr<ClosureModel>(new PrescribedCoefficientField(
        grid, config.minimum_coefficient, config.maximum_coefficient));
    case ClosureType::DynamicSmagorinsky:
      throw std::invalid_argument(
        "dynamic_smagorinsky is intentionally unsupported in Phase 7");
  }
  throw std::invalid_argument("unknown closure type");
}

}  // namespace burgers
