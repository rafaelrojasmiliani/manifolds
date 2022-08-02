#pragma once

#include <Manifolds/Topology/Open.hpp>

#include <Eigen/Core>

namespace manifolds {

template <typename DomainT, typename CodomainT> class Map {
public:
  typedef DomainT Domain;
  typedef CodomainT Codomain;
  CodomainT operator()(const DomainT &_domain) const = 0;
};

template <typename DomainT>
class Chart : public Map<DomainT, Eigen::Matrix<double, DomainT::dim, 1>> {

  friend DomainT;

public:
  typedef Eigen::Matrix<double, DomainT::dim, 1> Codomain;

  Chart(std::function<bool(const DomainT &)> _fun,
        topology::CartensianInterval<DomainT::dim>);

  Codomain operator()(const DomainT &_domain) const = 0;
};
} // namespace manifolds
