#pragma once
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>
#include <Manifolds/ManifoldBase.hpp>

namespace manifolds {

/* class CanonicInterval */
/*     : public ManifoldInheritanceHelper<CanonicInterval, Reals> { */
/* public: */
/*   using base_t = ManifoldInheritanceHelper<CanonicInterval, Reals>; */

/*   CanonicInterval(double in) : base_t() { */
/*     if (not contains(in)) */
/*       throw std::invalid_argument( */
/*           "cannot be initialized by number ouside interval"); */
/*     this->repr() = in; */
/*   } */

/*   CanonicInterval() = default; */
/*   CanonicInterval(const CanonicInterval &that) = default; */
/*   CanonicInterval(CanonicInterval &&that) = default; */

/*   using base_t::operator=; */

/*   const CanonicInterval &operator=(double in) { */
/*     if (not contains(in)) */
/*       throw std::invalid_argument( */
/*           "cannot be initialized by number ouside interval"); */
/*     this->repr() = in; */
/*     return *this; */
/*   } */

/*   operator const double &() const { return this->crepr(); } */

/*   bool contains(const Reals &other) { */
/*     if (other <= -1.0 - 1.0e-9 or 1.0 + 1.0e-9 <= other) */
/*       return false; */
/*     return true; */
/*   } */
/* }; */

/* template <std::size_t T> */
/* class ZeroToNInterval */
/*     : public ManifoldInheritanceHelper<CanonicInterval, Reals> { */
/* public: */
/*   using base_t = ManifoldInheritanceHelper<CanonicInterval, Reals>; */

/*   ZeroToNInterval(double in) : base_t() { */
/*     if (not contains(in)) */
/*       throw std::invalid_argument( */
/*           "cannot be initialized by number ouside interval"); */
/*     this->repr() = in; */
/*   } */

/*   ZeroToNInterval() = default; */
/*   ZeroToNInterval(const ZeroToNInterval &that) = default; */
/*   ZeroToNInterval(ZeroToNInterval &&that) = default; */

/*   using base_t::operator=; */

/*   const ZeroToNInterval &operator=(double in) { */
/*     if (not contains(in)) */
/*       throw std::invalid_argument( */
/*           "cannot be initialized by number ouside interval"); */
/*     this->repr() = in; */
/*     return *this; */
/*   } */

/*   operator const double &() const { return this->crepr(); } */

/*   bool contains(const Reals &other) { */
/*     if (other <= 0.0 - 1.0e-9 or T + 1.0e-9 <= other) */
/*       return false; */
/*     return true; */
/*   } */
/* }; */

class Interval : public ManifoldInheritanceHelper<Interval, LinearManifold<2>> {
public:
  using base_t = ManifoldInheritanceHelper<Interval, LinearManifold<2>>;
  using base_t::base_t;
  Interval(const std::pair<double, double> _pair)
      : base_t(Eigen::Vector2d({_pair.first, _pair.second})) {}

  Interval &operator=(const std::pair<double, double> &_pair) {
    this->repr() = Eigen::Vector2d({_pair.first, _pair.second});
    return *this;
  }
  double length() const {
    const auto &vec = std::get<Eigen::Vector2d>(this->crepr());
    return vec(1) - vec(0);
  }

  double first() const {

    const auto &vec = std::get<Eigen::Vector2d>(this->crepr());
    return vec(0);
  }
  double second() const {

    const auto &vec = std::get<Eigen::Vector2d>(this->crepr());
    return vec(1);
  }

  bool contains(double val) const {
    const auto &vec = std::get<Eigen::Vector2d>(this->crepr());
    return vec(0) - std::numeric_limits<double>::epsilon() <= val and
           val <= vec(1) + -std::numeric_limits<double>::epsilon();
  }

  double get_random() const {
    double lambda = (Eigen::Vector<double, 1>::Random()(0) + 1.0) / 2.0;
    return first() * lambda + (1.0 - lambda) * second();
  }
};

template <std::size_t NumberOfIntervals>
class IntervalPartition
    : public Manifold<LinearManifoldAtlas<NumberOfIntervals, 1>, false> {
public:
  using base_t = Manifold<LinearManifoldAtlas<NumberOfIntervals, 1>, false>;
  using base_t::base_t;
  IntervalPartition(
      const Interval &_interval,
      Eigen::Ref<const Eigen::Vector<double, NumberOfIntervals>> in)
      : base_t(), interval_(_interval) {
    this->repr() = in;
  }
  IntervalPartition(const Interval &_interval)
      : base_t(), interval_(_interval) {
    this->repr().array() = interval_.length() / NumberOfIntervals;
  }
  IntervalPartition(const std::pair<double, double> &_interval)
      : base_t(), interval_(_interval) {
    this->repr().array() = interval_.length() / NumberOfIntervals;
  }

  IntervalPartition &
  operator=(Eigen::Ref<const Eigen::Vector<double, NumberOfIntervals>> in) {

    this->repr() = in;
    return *this;
  }

  std::size_t subinterval_index(double val) const {
    double t0 = interval_.first();
    double t1;
    for (std::size_t i = 0; i < NumberOfIntervals; i++) {
      t1 = t0 + this->crepr()(i);
      if (t0 - std::numeric_limits<double>::epsilon() <= val and
          val <= t1 + std::numeric_limits<double>::epsilon())
        return i;
      t0 = t1;
    }
    throw std::invalid_argument("outside interval");
  }

  std::size_t subinterval_length(std::size_t indx) const {
    return this->crepr()(indx);
  }

  const Interval &interval() const { return interval_; }

private:
  Interval interval_;
};

} // namespace manifolds
