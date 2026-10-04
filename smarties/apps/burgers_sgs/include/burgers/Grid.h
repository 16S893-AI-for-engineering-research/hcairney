#pragma once

#include "burgers/Config.h"

#include <cstddef>
#include <vector>

namespace burgers {

class Grid {
public:
  explicit Grid(const GridConfig& config);
  Grid(double x_begin, double x_end, std::size_t cell_count);

  double xBegin() const noexcept;
  double xEnd() const noexcept;
  double length() const noexcept;
  std::size_t cellCount() const noexcept;
  double cellWidth() const noexcept;
  double cellWidth(std::size_t cell) const;
  double cellCenter(std::size_t cell) const;

  const std::vector<double>& cellCenters() const noexcept;
  const std::vector<double>& cellWidths() const noexcept;

  std::size_t periodicIndex(std::ptrdiff_t index) const noexcept;
  std::size_t neighbor(std::size_t cell, std::ptrdiff_t offset) const;

private:
  double x_begin_;
  double x_end_;
  double length_;
  double cell_width_;
  std::vector<double> cell_centers_;
  std::vector<double> cell_widths_;
};

}  // namespace burgers
