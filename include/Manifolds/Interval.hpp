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

class Interval
    : public detail::Clonable<Interval, DenseMatrixManifold<2, 1, false>> {

  template <std::size_t NumberOfIntervals> friend class IntervalPartition;

public:
  using base_t = detail::Clonable<Interval, DenseMatrixManifold<2, 1, false>>;
  using base_t::base_t;

  static Interval random() {
    Eigen::Vector2d val = Eigen::Vector2d::Random();
    val(1) = val(0) + val(1);
    return Interval(val);
  }
  static Interval from_repr(const Eigen::Vector2d &val) {
    if (val(1) < val(0))
      throw std::invalid_argument("invalid interval");
    return Interval(val);
  }
  static Interval from_repr(Eigen::Vector2d &&val) {
    if (val(1) < val(0))
      throw std::invalid_argument("invalid interval");
    return Interval(std::move(val));
  }

  Interval(double a, double b) : base_t(Eigen::Vector2d({a, b})) {
    if (b < a)
      throw std::invalid_argument("invalid interval");
  }

  Interval(const Interval &) = default;
  Interval(Interval &&) = default;
  Interval &operator=(const Interval &) = default;
  Interval &operator=(Interval &&) = default;

  double length() const {
    const auto &vec = this->crepr();
    return vec(1) - vec(0);
  }

  double first() const {

    const auto &vec = this->crepr();
    return vec(0);
  }
  double second() const {

    const auto &vec = this->crepr();
    return vec(1);
  }

  bool contains(double val) const {
    const auto &vec = this->crepr();
    return vec(0) - std::numeric_limits<double>::epsilon() <= val and
           val <= vec(1) + -std::numeric_limits<double>::epsilon();
  }

  double get_random() const {
    double lambda = (Eigen::Vector<double, 1>::Random()(0) + 1.0) / 2.0;
    return first() * lambda + (1.0 - lambda) * second();
  }

  std::tuple<double, double> as_tuple() {
    return std::make_tuple<double, double>(first(), second());
  }

  Eigen::VectorXd sized_linspace(std::size_t n) {
    return Eigen::VectorXd::LinSpaced(n, first(), second());
  }
  Eigen::VectorXd spaced_linspace(double step) {
    std::size_t n = ((first() - second()) / step) + 1;
    return Eigen::VectorXd::LinSpaced(n, first(), first() + step * (n - 1));
  }
};

template <std::size_t NumberOfIntervals>
class IntervalPartition
    : public detail::Clonable<
          IntervalPartition<NumberOfIntervals>,
          DenseMatrixManifold<NumberOfIntervals, 1, false>> {
public:
  using base_t =
      detail::Clonable<IntervalPartition<NumberOfIntervals>,
                       DenseMatrixManifold<NumberOfIntervals, 1, false>>;

  using base_t::base_t;

  IntervalPartition(
      const Interval &_interval,
      Eigen::Ref<const Eigen::Vector<double, NumberOfIntervals>> in)
      : base_t(), interval_(_interval) {
    this->repr() = in;
  }
  IntervalPartition(double a, double b) : base_t(), interval_(a, b) {
    this->repr().array() = interval_.length() / NumberOfIntervals;
  }

  explicit IntervalPartition(const Interval &_interval)
      : base_t(), interval_(_interval) {
    this->repr().array() = interval_.length() / NumberOfIntervals;
  }

  IntervalPartition &
  operator=(Eigen::Ref<const Eigen::Vector<double, NumberOfIntervals>> in) {

    this->repr() = in;
    return *this;
  }
  IntervalPartition(const IntervalPartition &) = default;
  IntervalPartition(IntervalPartition &&) = default;
  IntervalPartition &operator=(const IntervalPartition &) = default;
  IntervalPartition &operator=(IntervalPartition &&) = default;

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
    throw std::invalid_argument("outside interval \n interval is [" +
                                std::to_string(interval_.first()) + ", " +
                                std::to_string(interval_.second()) +
                                "] \n"
                                "and value is" +
                                std::to_string(val));
  }

  std::tuple<std::size_t, double>
  subinterval_index_and_canonic_value(double val) const {
    double t0 = interval_.first();
    double t1;
    for (std::size_t i = 0; i < NumberOfIntervals; i++) {
      t1 = t0 + this->crepr()(i);
      if (t0 - std::numeric_limits<double>::epsilon() <= val and
          val <= t1 + std::numeric_limits<double>::epsilon()) {

        return {i, 2 * (val - t0) / (t1 - t0) - 1.0};
      }
      t0 = t1;
    }
    throw std::invalid_argument("outside interval \n interval is [" +
                                std::to_string(interval_.first()) + ", " +
                                std::to_string(interval_.second()) +
                                "] \n"
                                "and value is" +
                                std::to_string(val));
  }

  double subinterval_length(std::size_t indx) const {
    return this->crepr()(indx);
  }

  const Interval &interval() const { return interval_; }

private:
  Interval interval_;
};

} // namespace manifolds
