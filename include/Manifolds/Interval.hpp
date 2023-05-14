#pragma once
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>
#include <Manifolds/ManifoldBase.hpp>
#include <fpm/fixed.hpp>

namespace manifolds {

class CanonicInterval
    : public ManifoldInheritanceHelper<CanonicInterval, Reals> {
public:
  using base_t = ManifoldInheritanceHelper<CanonicInterval, Reals>;

  CanonicInterval(double in) : base_t() {
    if (not contains(in))
      throw std::invalid_argument(
          "cannot be initialized by number ouside interval");
    this->repr() = in;
  }

  CanonicInterval() = default;
  CanonicInterval(const CanonicInterval &that) = default;
  CanonicInterval(CanonicInterval &&that) = default;

  using base_t::operator=;

  const CanonicInterval &operator=(double in) {
    if (not contains(in))
      throw std::invalid_argument(
          "cannot be initialized by number ouside interval");
    this->repr() = in;
    return *this;
  }

  operator const double &() const { return this->crepr(); }

  bool contains(const Reals &other) {
    if (other <= -1.0 - 1.0e-9 or 1.0 + 1.0e-9 <= other)
      return false;
    return true;
  }
};

template <std::size_t T>
class ZeroToNInterval
    : public ManifoldInheritanceHelper<CanonicInterval, Reals> {
public:
  using base_t = ManifoldInheritanceHelper<CanonicInterval, Reals>;

  ZeroToNInterval(double in) : base_t() {
    if (not contains(in))
      throw std::invalid_argument(
          "cannot be initialized by number ouside interval");
    this->repr() = in;
  }

  ZeroToNInterval() = default;
  ZeroToNInterval(const ZeroToNInterval &that) = default;
  ZeroToNInterval(ZeroToNInterval &&that) = default;

  using base_t::operator=;

  const ZeroToNInterval &operator=(double in) {
    if (not contains(in))
      throw std::invalid_argument(
          "cannot be initialized by number ouside interval");
    this->repr() = in;
    return *this;
  }

  operator const double &() const { return this->crepr(); }

  bool contains(const Reals &other) {
    if (other <= 0.0 - 1.0e-9 or T + 1.0e-9 <= other)
      return false;
    return true;
  }
};
} // namespace manifolds
