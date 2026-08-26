#pragma once

#include "burgers/Grid.h"

#include <cstddef>
#include <vector>

namespace burgers {

class State {
public:
  using Container = std::vector<double>;
  using iterator = Container::iterator;
  using const_iterator = Container::const_iterator;

  explicit State(std::size_t cell_count, double value = 0.0);
  explicit State(const Grid& grid, double value = 0.0);
  explicit State(Container cell_averages);

  std::size_t size() const noexcept;
  bool empty() const noexcept;

  double* data() noexcept;
  const double* data() const noexcept;

  double& operator[](std::size_t cell) noexcept;
  const double& operator[](std::size_t cell) const noexcept;
  double& at(std::size_t cell);
  const double& at(std::size_t cell) const;

  iterator begin() noexcept;
  const_iterator begin() const noexcept;
  const_iterator cbegin() const noexcept;
  iterator end() noexcept;
  const_iterator end() const noexcept;
  const_iterator cend() const noexcept;

  void fill(double value);
  Container& cellAverages() noexcept;
  const Container& cellAverages() const noexcept;

private:
  Container cell_averages_;
};

}  // namespace burgers
