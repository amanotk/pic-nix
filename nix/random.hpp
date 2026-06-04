// -*- C++ -*-
#ifndef _RANDOM_HPP_
#define _RANDOM_HPP_

#include "nix.hpp"

NIX_NAMESPACE_BEGIN

using rand_uniform = std::uniform_real_distribution<float64>;
using rand_normal  = std::normal_distribution<float64>;
using rand_poisson = std::poisson_distribution<int>;
using rand_gamma   = std::gamma_distribution<float64>;

///
/// @brief Relativistic Maxwell-Juttner Distribution
///
/// This class draws random numbers from the relativistic Maxwell-Juttner distribution with
/// a finite drift in x direction. In other words, the distribution is isotropic in the rest
/// frame and is Lorentz boosted to the lab frame.
/// Since the drift direction is limited to x, an appropriate coordinate transformation is
/// required to generate a distribution drifting in a different direction.
///
/// Reference
/// - S. Zenitani, Physics of Plasmas 22, 042116 (2015).
/// - S. Zenitani and S. Nakano, Physics of Plasmas 29, 113904 (2022).
///
class MaxwellJuttner
{
private:
  static inline constexpr float64 a = 0.56;
  static inline constexpr float64 b = 0.35;

  float64                 temperature;
  float64                 drift;
  rand_uniform            uniform;
  std::vector<rand_gamma> gamma;
  std::vector<float64>    table;

  void initialize()
  {
    table = {
        0,
        std::sqrt(M_PI),
        a * std::sqrt(2 * temperature),
        b * 1.5 * std::sqrt(M_PI) * temperature,
        std::sqrt(2 * temperature) * (2 * temperature),
    };

    // calculate cumulative sum and normalize to 1
    std::partial_sum(table.begin(), table.end(), table.begin());
    std::transform(table.begin(), table.end(), table.begin(),
                   [&](float64 x) { return x / table.back(); });

    // initialize gamma distribution
    gamma = {
        rand_gamma(1.5, temperature),
        rand_gamma(2.0, temperature),
        rand_gamma(2.5, temperature),
        rand_gamma(3.0, temperature),
    };
  }

public:
  MaxwellJuttner(float64 temperature, float64 drift = 0) : temperature(temperature), drift(drift)
  {
    initialize();
  }

  void set_temperature(float64 temperature)
  {
    this->temperature = temperature;
    initialize();
  }

  void set_drift(float64 drift)
  {
    this->drift = drift;
  }

  void reset()
  {
    uniform.reset();
    for (auto& g : gamma) {
      g.reset();
    }
  }

  template <typename Random>
  void operator()(Random& random, float64& ux, float64& uy, float64& uz)
  {
    // Lorentz factor with the modified Canfield method
    bool    status = false;
    float64 xx     = -1;
    do {
      float64 r1    = uniform(random);
      float64 r2    = uniform(random);
      int     index = std::upper_bound(table.begin(), table.end(), r1) - table.begin() - 1;
      xx            = gamma[index](random);
      float64 R1    = (xx + 1) * std::sqrt(xx + 2);
      float64 R2    = std::sqrt(2) + a * std::sqrt(xx) + b * std::sqrt(2) * xx + std::pow(xx, 1.5);
      status        = (r2 > R1 / R2);
    } while (status);

    // isotropic distribution in the rest frame
    float64 uu = std::sqrt(xx * (xx + 2));
    float64 r3 = uniform(random);
    float64 r4 = uniform(random);

    ux = uu * (2 * r3 - 1);
    uy = uu * 2 * std::sqrt(r3 * (1 - r3)) * std::cos(2 * M_PI * r4);
    uz = uu * 2 * std::sqrt(r3 * (1 - r3)) * std::sin(2 * M_PI * r4);

    // Lorentz boost to the lab frame with the flipping method
    ux = lorentz_boost(random, drift, ux, uy, uz);
  }

  template <typename Random>
  std::tuple<float64, float64, float64> operator()(Random& random)
  {
    float64 ux, uy, uz;
    (*this)(random, ux, uy, uz);
    return std::make_tuple(ux, uy, uz);
  }

  template <typename Random>
  float64 lorentz_boost(Random& random, float64 u0, float64 ux, float64 uy, float64 uz)
  {
    float64 rr = uniform(random);
    float64 Gm = std::sqrt(1 + u0 * u0);
    float64 Vx = u0 / Gm;
    float64 gm = std::sqrt(1 + ux * ux + uy * uy + uz * uz);
    float64 vx = ux / gm;

    ux = (-Vx * vx > rr) ? -ux : +ux;
    return Gm * (ux + Vx * gm);
  }
};

///
/// @brief Ring distribution in the perpendicular plane
///
/// This class draws the magnitude |v_perp| of a 2D ring (non-central Rayleigh)
/// distribution with ring radius v_ring and thermal spread vt_perp.
/// The angle in the perpendicular plane is then distributed uniformly in [0, 2π).
///
/// The PDF is P(|v|) ∝ |v| * exp(-(|v| - v_ring)^2 / (2 * vt_perp^2))
///
/// Reference
/// - The inverse transform method with Newton's method, adapted from
///   S. Zenitani, "Ring" class in hybrid3d (2024).
///
class MaxwellianRing
{
private:
  float64      v_ring;
  float64      vt_perp;
  rand_uniform uniform;

  static inline constexpr float64 sqrt_2  = 1.4142135623730951;
  static inline constexpr float64 sqrt_pi = 1.7724538509055160;

public:
  MaxwellianRing(float64 v_ring, float64 vt_perp) : v_ring(v_ring), vt_perp(vt_perp)
  {
  }

  void set_vring(float64 v_ring)
  {
    this->v_ring = v_ring;
  }

  void set_vtperp(float64 vt_perp)
  {
    this->vt_perp = vt_perp;
  }

  void reset()
  {
    uniform.reset();
  }

  template <typename Random>
  float64 operator()(Random& random)
  {
    // handle vanishing thermal spread
    if (vt_perp <= 1.0e-14) {
      return v_ring;
    }

    // inverse transform sampling with Newton's method
    const int max_iter = 5;
    float64   yy       = uniform(random);
    float64   xx       = 0.5 * (v_ring + std::sqrt(v_ring * v_ring + 4 * vt_perp * vt_perp));
    float64   V        = v_ring / (sqrt_2 * vt_perp);
    float64   F0       = vt_perp * vt_perp * (std::exp(-V * V) + sqrt_pi * V * (1 + std::erf(V)));

    for (int i = 0; i < max_iter; i++) {
      float64 x  = (xx - v_ring) / (sqrt_2 * vt_perp);
      float64 FF = F0 - vt_perp * vt_perp * (std::exp(-x * x) + sqrt_pi * V * (1 - std::erf(x)));
      float64 dF = xx * std::exp(-x * x);
      xx += (yy * F0 - FF) / dF;
    }

    return xx;
  }
};

NIX_NAMESPACE_END

#endif
