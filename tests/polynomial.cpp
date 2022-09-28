

#include <Manifolds/LinearManifolds.hpp>
#include <Manifolds/Map.hpp>
#include <gsplines/Collocation/GaussLobattoLagrange.hpp>
#include <gtest/gtest.h>
#include <numeric> // std::iota
#include <random>

// This example shows how to automatically generate a chart in the sphere from
//  two points.
//
//  The charts are spherical coordinates with inclination and azimuth  with
//  respect to a frame of coordinates generated from two points.
//  Two points are used to generate charts and respective parametrizations.
//  The chart should cover both points.
//
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> inclination_dist(0, 3.1);
std::uniform_real_distribution<double> azimuth(-3.1, 3.1);

using namespace manifolds;

class Polynomial
    : public MapInheritanceHelper<Polynomial, Map<Reals<>, Reals<>>> {

public:
  typedef MapInheritanceHelper<Polynomial, Map<Reals<>, Reals<>>> base_t;

  Polynomial(double _t0, double _t1, const gsplines::basis::Basis &_base,
             const Eigen::VectorXd &_x)
      : base_t(), t0_(_t0), t1_(_t1), base_(_base.clone()), coeff_(_x),
        basis_buffer_(base_->get_dim()) {

    if (t1_ < t0_)
      throw std::invalid_argument("");
    if (_x.size() != (long)_base.get_dim())
      throw std::invalid_argument("");
  }

  Polynomial(const Polynomial &_other)
      : base_t(_other), t0_(_other.t0_), t1_(_other.t1_),
        base_(_other.base_->clone()), coeff_(_other.coeff_),
        basis_buffer_(_other.basis_buffer_) {}

  Polynomial(Polynomial &&_other)
      : base_t(_other), t0_(_other.t0_), t1_(_other.t1_),
        base_(_other.base_->move_clone()), coeff_(std::move(_other.coeff_)),
        basis_buffer_(std::move(_other.basis_buffer_)) {}

  virtual ~Polynomial() = default;

private:
  double t0_;
  double t1_;
  std::unique_ptr<gsplines::basis::Basis> base_;
  Eigen::VectorXd coeff_;
  mutable Eigen::VectorXd basis_buffer_;

  bool value_on_repr(const double &_in, double &_out) const override {

    double t = _in;
    if (t < t0_)
      t = t0_;
    if (t > t1_)
      t = t1_;
    double s = 2.0 * (t - t0_) / (t1_ - t0_) - 1.0;
    base_->eval_on_window(s, t1_ - t0_, basis_buffer_);
    _out = coeff_.transpose() * basis_buffer_;

    return true;
  }

  bool diff_from_repr(const double &_in, Eigen::MatrixXd &_out) const override {
    double t = _in;
    if (t < t0_)
      t = t0_;
    if (t > t1_)
      t = t1_;
    double s = 2.0 * (t - t0_) / (t1_ - t0_) - 1.0;
    base_->eval_derivative_on_window(s, t1_ - t0_, 1, basis_buffer_);
    _out(0, 0) = coeff_.transpose() * basis_buffer_;
    return true;
  }
};
template <long N>
class VectorPolynomial
    : public MapInheritanceHelper<VectorPolynomial<N>, Map<Reals<>, Reals<N>>> {

public:
  typedef MapInheritanceHelper<VectorPolynomial<N>, Map<Reals<>, Reals<N>>>
      base_t;
  VectorPolynomial(double _t0, double _t1, const gsplines::basis::Basis &_base,
                   const Eigen::VectorXd &_x)
      : base_t(), t0_(_t0), t1_(_t1), base_(_base.clone()), coeff_(_x),
        basis_buffer_(base_->get_dim()) {

    if (t1_ < t0_)
      throw std::invalid_argument("");
    if (_x.size() != (long)_base.get_dim() * N)
      throw std::invalid_argument("");
  }

  VectorPolynomial(const VectorPolynomial<N> &_other)
      : base_t(_other), t0_(_other.t0_), t1_(_other.t1_),
        base_(_other.base_->clone()), coeff_(_other.coeff_),
        basis_buffer_(_other.basis_buffer_) {}

  VectorPolynomial(VectorPolynomial<N> &&_other)
      : base_t(_other), t0_(_other.t0_), t1_(_other.t1_),
        base_(_other.base_->move_clone()), coeff_(std::move(_other.coeff_)),
        basis_buffer_(std::move(_other.basis_buffer_)) {}

  virtual ~VectorPolynomial() = default;

private:
  double t0_;
  double t1_;
  std::unique_ptr<gsplines::basis::Basis> base_;
  Eigen::VectorXd coeff_;
  mutable Eigen::VectorXd basis_buffer_;

  bool value_on_repr(const double &_in,
                     Eigen::Matrix<double, N, 1> &_out) const override {

    double t = _in;
    if (t < t0_)
      t = t0_;
    if (t > t1_)
      t = t1_;
    double s = 2.0 * (t - t0_) / (t1_ - t0_) - 1.0;
    base_->eval_on_window(s, t1_ - t0_, basis_buffer_);

    for (int i = 0; i < N; i++) {
      _out(i) = basis_buffer_.transpose() *
                coeff_.segment(i * base_->get_dim(),
                               i * base_->get_dim() + base_->get_dim());
    }

    return true;
  }

  bool diff_from_repr(const double &_in, Eigen::MatrixXd &_out) const override {
    double t = _in;
    if (t < t0_)
      t = t0_;
    if (t > t1_)
      t = t1_;
    double s = 2.0 * (t - t0_) / (t1_ - t0_) - 1.0;
    base_->eval_derivative_on_window(s, t1_ - t0_, 1, basis_buffer_);
    _out(0, 0) = coeff_.transpose() * basis_buffer_;
    return true;
  }
};

TEST(Manifolds, Sphere) {

  std::size_t nc = 10;
  VectorPolynomial<2>(0, 1, gsplines::basis::BasisLagrangeGaussLobatto(nc),
                      Eigen::VectorXd::Random(nc * 2));
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
