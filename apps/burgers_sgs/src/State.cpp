#include "burgers/State.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace burgers {

State::State(std::size_t cell_count, double value)
  : cell_averages_(cell_count, value) {
  if(cell_count == 0) {
    throw std::invalid_argument("state cell count must be positive");
  }
}

State::State(const Grid& grid, double value)
  : State(grid.cellCount(), value) {}

State::State(Container cell_averages)
  : cell_averages_(std::move(cell_averages)) {
  if(cell_averages_.empty()) {
    throw std::invalid_argument("state cell averages must not be empty");
  }
}

std::size_t State::size() const noexcept {
  return cell_averages_.size();
}

bool State::empty() const noexcept {
  return cell_averages_.empty();
}

double* State::data() noexcept {
  return cell_averages_.data();
}

const double* State::data() const noexcept {
  return cell_averages_.data();
}

double& State::operator[](std::size_t cell) noexcept {
  return cell_averages_[cell];
}

const double& State::operator[](std::size_t cell) const noexcept {
  return cell_averages_[cell];
}

double& State::at(std::size_t cell) {
  return cell_averages_.at(cell);
}

const double& State::at(std::size_t cell) const {
  return cell_averages_.at(cell);
}

State::iterator State::begin() noexcept {
  return cell_averages_.begin();
}

State::const_iterator State::begin() const noexcept {
  return cell_averages_.begin();
}

State::const_iterator State::cbegin() const noexcept {
  return cell_averages_.cbegin();
}

State::iterator State::end() noexcept {
  return cell_averages_.end();
}

State::const_iterator State::end() const noexcept {
  return cell_averages_.end();
}

State::const_iterator State::cend() const noexcept {
  return cell_averages_.cend();
}

void State::fill(double value) {
  std::fill(cell_averages_.begin(), cell_averages_.end(), value);
}

State::Container& State::cellAverages() noexcept {
  return cell_averages_;
}

const State::Container& State::cellAverages() const noexcept {
  return cell_averages_;
}

}  // namespace burgers
