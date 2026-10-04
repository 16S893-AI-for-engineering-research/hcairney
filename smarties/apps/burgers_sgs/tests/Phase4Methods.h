#pragma once

#include "burgers/Config.h"

#include <vector>

namespace phase4_test {

struct MethodCase {
  burgers::Reconstruction reconstruction;
  burgers::ConvectiveFlux flux;
  burgers::Limiter limiter;
  const char* name;
  int expected_spatial_order;
};

inline const std::vector<MethodCase>& methodCases() {
  static const std::vector<MethodCase> cases{
    {burgers::Reconstruction::PiecewiseConstant,
     burgers::ConvectiveFlux::Godunov,
     burgers::Limiter::None,
     "piecewise-godunov",
     1},
    {burgers::Reconstruction::PiecewiseConstant,
     burgers::ConvectiveFlux::Rusanov,
     burgers::Limiter::None,
     "piecewise-rusanov",
     1},
    {burgers::Reconstruction::Muscl,
     burgers::ConvectiveFlux::Godunov,
     burgers::Limiter::MonotonizedCentral,
     "muscl-mc-godunov",
     2},
    {burgers::Reconstruction::Muscl,
     burgers::ConvectiveFlux::Godunov,
     burgers::Limiter::Minmod,
     "muscl-minmod-godunov",
     2},
    {burgers::Reconstruction::Muscl,
     burgers::ConvectiveFlux::Godunov,
     burgers::Limiter::VanLeer,
     "muscl-vanleer-godunov",
     2},
    {burgers::Reconstruction::Muscl,
     burgers::ConvectiveFlux::Rusanov,
     burgers::Limiter::MonotonizedCentral,
     "muscl-mc-rusanov",
     2},
    {burgers::Reconstruction::Muscl,
     burgers::ConvectiveFlux::Rusanov,
     burgers::Limiter::Minmod,
     "muscl-minmod-rusanov",
     2},
    {burgers::Reconstruction::Muscl,
     burgers::ConvectiveFlux::Rusanov,
     burgers::Limiter::VanLeer,
     "muscl-vanleer-rusanov",
     2}};
  return cases;
}

inline void selectMethod(burgers::RunConfig& config,
                         const MethodCase& method) {
  config.numerical_method.reconstruction = method.reconstruction;
  config.numerical_method.convective_flux = method.flux;
  config.numerical_method.limiter = method.limiter;
}

}  // namespace phase4_test
