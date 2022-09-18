#pragma once
#include <Manifolds/Map.hpp>
#include <eigen3/Eigen/Core>

#include <gsplines/Basis/Basis.hpp>
#include <gsplines/Collocation/GaussLobattoLagrange.hpp>
#include <memory>
#include <vector>

namespace manifolds {

template <typename M> class MGSpline {

protected:
  Eigen::VectorXd coefficients_;
  Eigen::VectorXd domain_interval_lengths_;

  std::unique_ptr<gsplines::basis::Basis> basis_;
  mutable Eigen::VectorXd basis_buffer_;
  double interval_to_window(double _t, std::size_t _interval) const;

  Eigen::Ref<const Eigen::VectorXd>
  coefficient_segment(std::size_t _interval, std::size_t _component) const;

  Eigen::Ref<Eigen::VectorXd> coefficient_segment(std::size_t _interval,
                                                  std::size_t _component);

  std::size_t get_interval(double _domain_point) const;

public:
  MGSpline(std::pair<double, double> _domain, std::size_t _codom_dim,
           std::size_t _n_intervals, const gsplines::basis::Basis &_basis,
           const Eigen::Ref<const Eigen::VectorXd> _coefficents,
           const Eigen::Ref<const Eigen::VectorXd> _tauv,
           const std::string &_name = "MGSpline");

  MGSpline(std::pair<double, double> _domain, std::size_t _codom_dim,
           std::size_t _n_intervals, const gsplines::basis::Basis &_basis,
           Eigen::VectorXd &&_coefficents, Eigen::VectorXd &&_tauv,
           const std::string &_name = "MGSpline");

  MGSpline(const MGSpline &that)
      : coefficients_(that.coefficients_),
        domain_interval_lengths_(that.domain_interval_lengths_),
        basis_(that.basis_->clone()), basis_buffer_(basis_->get_dim()) {}

  MGSpline(MGSpline &&that)
      : coefficients_(std::move(that.coefficients_)),
        domain_interval_lengths_(std::move(that.domain_interval_lengths_)),
        basis_(that.basis_->move_clone()), basis_buffer_(basis_->get_dim()) {}

public:
  void value(const Eigen::Ref<const Eigen::VectorXd> _domain_points,
             Eigen::Ref<Eigen::MatrixXd> _result) const;

  Eigen::VectorXd get_domain_breakpoints() const;

  Eigen::MatrixXd get_waypoints() const;

  virtual ~MGSpline() = default;
  const Eigen::VectorXd &get_coefficients() const { return coefficients_; }

  const Eigen::VectorXd &get_interval_lengths() const {
    return domain_interval_lengths_;
  }

  const std::string &get_basis_name() const { return basis_->get_name(); }

  std::size_t get_number_of_intervals() const {
    return domain_interval_lengths_.size();
  }

  std::size_t get_basis_dim() const { return basis_->get_dim(); }

  const gsplines::basis::Basis &get_basis() const { return *basis_; }

  bool same_vector_space(const MGSpline &_that) const;

  bool operator==(const MGSpline &_that) const;

  bool operator!=(const MGSpline &_that) const;

protected:
  MGSpline *deriv_impl(std::size_t) const override {
    throw std::runtime_error("Base class cannot be used");
    return nullptr;
  }
};
} // namespace manifolds
