#pragma once

#include "burgers/BurgersSolver.h"
#include "burgers/Config.h"
#include "burgers/State.h"

#include <cstddef>
#include <memory>

namespace burgers {

// Writes the scalar, profile, final-profile, and optional spectrum products
// shared by standalone runs and recorded SMARTIES evaluation episodes.
class RunOutputRecorder {
public:
  explicit RunOutputRecorder(const RunConfig& config);
  ~RunOutputRecorder();

  RunOutputRecorder(const RunOutputRecorder&) = delete;
  RunOutputRecorder& operator=(const RunOutputRecorder&) = delete;

  void begin(const BurgersSolver& solver,
             const State& state,
             double initial_time,
             double final_time);
  double nextEventTime() const noexcept;
  void observeAdvance(const BurgersSolver& solver,
                      const State& state,
                      const AdvanceResult& advance);
  void finish(const BurgersSolver& solver,
              const State& state,
              double time);

  const AdvanceResult& totalAdvance() const noexcept;
  std::size_t statisticsSampleCount() const noexcept;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

}  // namespace burgers
