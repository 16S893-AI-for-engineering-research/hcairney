#include "burgers/Grid.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace burgers {

Grid::Grid(const GridConfig& config)
  : Grid(config.x_begin, config.x_end, config.cell_count) {}

Grid::Grid(double x_begin, double x_end, std::size_t cell_count)
  : x_begin_(x_begin),
    x_end_(x_end),
    length_(x_end - x_begin),
    cell_width_(0.0) {
  if(std::isfinite(x_begin_) == 0 || std::isfinite(x_end_) == 0) {
    throw std::invalid_argument("grid endpoints must be finite");
  }
  if(x_end_ <= x_begin_) {
    throw std::invalid_argument("grid end must be greater than grid begin");
  }
  if(std::isfinite(length_) == 0) {
    throw std::invalid_argument("grid length must be finite");
  }
  if(cell_count == 0) {
    throw std::invalid_argument("grid cell count must be positive");
  }
  const std::size_t maximum_signed_size =
    static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if(cell_count > maximum_signed_size) {
    throw std::invalid_argument(
      "grid cell count is too large for signed periodic indexing");
  }

  cell_width_ = length_ / static_cast<double>(cell_count);
  cell_centers_.resize(cell_count);
  cell_widths_.assign(cell_count, cell_width_);
  for(std::size_t cell = 0; cell < cell_count; ++cell) {
    cell_centers_[cell] =
      x_begin_ + (static_cast<double>(cell) + 0.5) * cell_width_;
  }
}

double Grid::xBegin() const noexcept {
  return x_begin_;
}

double Grid::xEnd() const noexcept {
  return x_end_;
}

double Grid::length() const noexcept {
  return length_;
}

std::size_t Grid::cellCount() const noexcept {
  return cell_centers_.size();
}

double Grid::cellWidth() const noexcept {
  return cell_width_;
}

double Grid::cellWidth(std::size_t cell) const {
  return cell_widths_.at(cell);
}

double Grid::cellCenter(std::size_t cell) const {
  return cell_centers_.at(cell);
}

const std::vector<double>& Grid::cellCenters() const noexcept {
  return cell_centers_;
}

const std::vector<double>& Grid::cellWidths() const noexcept {
  return cell_widths_;
}

std::size_t Grid::periodicIndex(std::ptrdiff_t index) const noexcept {
  const std::ptrdiff_t count =
    static_cast<std::ptrdiff_t>(cell_centers_.size());
  const std::ptrdiff_t remainder = index % count;
  return static_cast<std::size_t>(remainder < 0 ? remainder + count
                                                : remainder);
}

std::size_t Grid::neighbor(std::size_t cell, std::ptrdiff_t offset) const {
  if(cell >= cellCount()) {
    throw std::out_of_range("grid cell index is out of range");
  }
  const std::size_t wrapped_offset = periodicIndex(offset);
  return (cell + wrapped_offset) % cellCount();
}

}  // namespace burgers
