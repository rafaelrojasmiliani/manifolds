
#include <Manifolds/MGSpline.hpp>
#include <gsplines/Basis/BasisLegendre.hpp>
#include <gsplines/GSpline.hpp>
#include <gsplines/Interpolator.hpp>
#include <gsplines/Tools.hpp>
#include <iostream>
#include <random>

namespace manifolds {

MGSpline::MGSpline(const MGSpline &that)
    : coefficients_(that.coefficients_),
      domain_interval_lengths_(that.domain_interval_lengths_),
      basis_(that.basis_->clone()), basis_buffer_(basis_->get_dim()) {}

MGSpline::MGSpline(MGSpline &&that)
    : FunctionInheritanceHelper(std::move(that)),
      coefficients_(std::move(that.coefficients_)),
      domain_interval_lengths_(std::move(that.domain_interval_lengths_)),
      basis_(that.basis_->move_clone()), basis_buffer_(basis_->get_dim()) {}

MGSpline::MGSpline(std::pair<double, double> _domain, std::size_t _codom_dim,
                   std::size_t _n_intervals, const basis::Basis &_basis,
                   const Eigen::Ref<const Eigen::VectorXd> _coefficents,
                   const Eigen::Ref<const Eigen::VectorXd> _tauv,
                   const std::string &_name)
    : FunctionInheritanceHelper(_domain, _codom_dim, _name),
      coefficients_(_coefficents), domain_interval_lengths_(_tauv),
      basis_(_basis.clone()), basis_buffer_(basis_->get_dim()) {

  if (coefficients_.size() !=
      (long)(_n_intervals * basis_->get_dim() * _codom_dim)) {
    throw std::invalid_argument(
        "MGSpline instantation Error: The number of coefficients is "
        "incorrect "
        "req base dim: " +
        std::to_string(basis_->get_dim()) +
        " req codom dim: " + std::to_string(get_codom_dim()) +
        " req num inter: " + std::to_string(get_number_of_intervals()) +
        ". However, the number of coeff was " +
        std::to_string(coefficients_.size()));
  }
}

MGSpline::MGSpline(std::pair<double, double> _domain, std::size_t _codom_dim,
                   std::size_t _n_intervals, const basis::Basis &_basis,
                   Eigen::VectorXd &&_coefficents, Eigen::VectorXd &&_tauv,
                   const std::string &_name)
    : FunctionInheritanceHelper(_domain, _codom_dim, _name),
      coefficients_(std::move(_coefficents)),
      domain_interval_lengths_(std::move(_tauv)), basis_(_basis.clone()),
      basis_buffer_(basis_->get_dim()) {

  if (coefficients_.size() !=
      (long)(_n_intervals * basis_->get_dim() * _codom_dim)) {
    throw std::invalid_argument(
        "MGSpline instantation Error: The number of coefficients is "
        "incorrect "
        "req base dim: " +
        std::to_string(basis_->get_dim()) +
        " req codom dim: " + std::to_string(get_codom_dim()) +
        " req num inter: " + std::to_string(get_number_of_intervals()) +
        ". However, the number of coeff was " +
        std::to_string(coefficients_.size()));
  }
}

void MGSpline::value_impl(
    const Eigen::Ref<const Eigen::VectorXd> _domain_points,
    Eigen::Ref<Eigen::MatrixXd> _result) const {

  std::size_t result_cols(_domain_points.size());
  std::size_t current_interval = 0;
  double s, tau;

  std::size_t i, j;

  for (i = 0; i < result_cols; i++) {
    current_interval = get_interval(_domain_points(i));
    s = interval_to_window(_domain_points(i), current_interval);
    tau = domain_interval_lengths_(current_interval);
    basis_->eval_on_window(s, tau, basis_buffer_);
    for (j = 0; j < get_codom_dim(); j++) {
      _result(i, j) =
          coefficient_segment(current_interval, j).adjoint() * basis_buffer_;
    }
  }
}

std::size_t MGSpline::get_interval(double _domain_point) const {
  double left_breakpoint = get_domain().first;
  double right_breakpoint;
  if (_domain_point <= left_breakpoint)
    return 0;

  for (std::size_t i = 0; i < get_number_of_intervals(); i++) {
    right_breakpoint = left_breakpoint + domain_interval_lengths_(i);
    if (left_breakpoint < _domain_point and _domain_point <= right_breakpoint) {
      return i;
    }
    left_breakpoint = right_breakpoint;
  }
  return get_number_of_intervals() - 1;
}

Eigen::Ref<const Eigen::VectorXd>
MGSpline::coefficient_segment(std::size_t _interval,
                              std::size_t _component) const {
  int i0 = _interval * basis_->get_dim() * get_codom_dim() +
           basis_->get_dim() * _component;
  return coefficients_.segment(i0, basis_->get_dim());
}

Eigen::Ref<Eigen::VectorXd>
MGSpline::coefficient_segment(std::size_t _interval, std::size_t _component) {
  int i0 = _interval * basis_->get_dim() * get_codom_dim() +
           basis_->get_dim() * _component;
  return coefficients_.segment(i0, basis_->get_dim());
}

double MGSpline::interval_to_window(double _domain_point,
                                    std::size_t _interval) const {
  double left_breakpoint =
      get_domain().first +
      domain_interval_lengths_.segment(0, _interval).array().sum();
  return 2.0 * (_domain_point - left_breakpoint) /
             domain_interval_lengths_[_interval] -
         1.0;
}

bool MGSpline::operator==(const MGSpline &_that) const {
  return tools::approx_equal(*this, _that, 1.0e-7);
}
bool MGSpline::operator!=(const MGSpline &_that) const {
  return not(*this == _that);
}

const Eigen::Ref<const Eigen::VectorXd>
get_coefficient_segment(const Eigen::Ref<const Eigen::VectorXd> _coefficients,
                        basis::Basis &_basis, std::size_t /*_num_interval*/,
                        std::size_t _codom_dim, std::size_t _interval,
                        std::size_t _component) {

  int i0 =
      _interval * _basis.get_dim() * _codom_dim + _basis.get_dim() * _component;
  return _coefficients.segment(i0, _basis.get_dim());
}
Eigen::VectorXd MGSpline::get_domain_breakpoints() const {

  double time_instant = get_domain().first;
  Eigen::VectorXd result(get_number_of_intervals() + 1);
  result(0) = time_instant;
  for (std::size_t i = 1; i < get_number_of_intervals() + 1; i++) {
    time_instant += domain_interval_lengths_(i - 1);
    result(i) = time_instant;
  }
  return result;
}

Eigen::MatrixXd MGSpline::get_waypoints() const {
  Eigen::MatrixXd result(get_number_of_intervals() + 1, get_codom_dim());
  MGSpline::value(get_domain_breakpoints(), result);
  return result;
}

bool MGSpline::same_vector_space(const MGSpline &_that) const {
  return get_basis() == _that.get_basis() and
         get_codom_dim() == _that.get_codom_dim() and
         tools::approx_equal(get_interval_lengths(),
                             _that.get_interval_lengths(), 1.0e-8);
}

} // namespace manifolds
