#pragma once

#include <Manifolds/Map.hpp>
namespace manifolds {

template <typename Manifold> class Parametrization;

template <typename DomainT>
class Chart : public Map<DomainT, Eigen::Matrix<double, DomainT::dim, 1>> {

  friend DomainT;

private:
  typedef Eigen::Matrix<double, DomainT::dim, 1> Rn;
  typedef Eigen::Matrix<double, DomainT::dim, DomainT::dim> Rnxn;

  typedef std::function<Rn(const DomainT &)> chart_function_t;

  typedef std::function<DomainT(const Rn &)> parametrization_function_t;

  typedef std::function<Rnxn(const DomainT &)> char_differential_;

  chart_function_t chart_;
  parametrization_function_t parametrization_;

public:
  typedef Eigen::Matrix<double, DomainT::dim, 1> Codomain;

  Chart(std::function<Codomain(const DomainT &)> _fun,
        topology::CartensianInterval<DomainT::dim>);

  virtual Codomain operator()(const DomainT &_domain) const = 0;

  virtual Parametrization<DomainT> inverse() const = 0;
};

template <typename Manifold>
class Parametrization
    : public Map<Eigen::Matrix<double, Manifold::dim, 1>, Manifold> {
private:
  std::function<Manifold(const Eigen::Matrix<double, Manifold::dim, 1> &)>
      param_;

  topology::CartensianInterval<Manifold::dim> domain_;

public:
  Parametrization(const std::function<Manifold(
                      const Eigen::Matrix<double, Manifold::dim, 1> &)> &_param,
                  const topology::CartensianInterval<Manifold::dim> &_domain)

      : param_(_param), domain_(_domain) {}
  Manifold operator()(const Eigen::Matrix<double, Manifold::dim, 1> &_in) {
    if (not domain_.contains(_in))
      throw std::invalid_argument("");
    return param_(_in);
  }

  bool in_domain(const Eigen::Matrix<double, Manifold::dim, 1> &_in) {
    return domain_.contains(_in);
  }
};
} // namespace manifolds
