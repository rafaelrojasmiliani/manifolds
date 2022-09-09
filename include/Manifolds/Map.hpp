#pragma once

#include <Manifolds/Manifold.hpp>
#include <Manifolds/Topology/Open.hpp>

#include <Eigen/Core>
#include <memory>

namespace manifolds {
/*
class MapBase {

public:
  virtual std::shared_ptr<Manifold>
  value_ptr(std::shared_ptr<const Manifold> &_in) const = 0;

  virtual Eigen::MatrixXd
  diff_ptr(std::shared_ptr<const Manifold> &_in) const = 0;
};

template <typename DomainT, typename CodomainT> class MapComposition;

template <typename DomainT, typename CodomainT> class Map : public MapBase {
public:
  typedef DomainT Domain;
  typedef CodomainT Codomain;

  Map(const std::function<CodomainT(const DomainT &)> _fun);
  Map(const std::function<CodomainT(const DomainT &)> _fun,
      const std::function<DomainT(const CodomainT &)> _inverse);
  CodomainT operator()(const DomainT &_domain) const {
    return value_impl(_domain);
  }

  template <typename OtherDomainT>
  MapComposition<CodomainT, OtherDomainT>
  operator|(const Map<DomainT, OtherDomainT> &_domain);

  virtual CodomainT value_impl(const DomainT &_domain) const = 0;

  std::shared_ptr<Manifold>
  value_ptr(std::shared_ptr<const Manifold> &_in) const {
    std::shared_ptr<const DomainT> in =
        std::dynamic_pointer_cast<const DomainT>(_in);
    CodomainT buff = value_impl(*in);
    return std::make_shared<CodomainT>(std::move(buff));
  }
};

template <typename DomainT, typename CodomainT>
class MapComposition : public MapBase {

private:
  std::vector<std::unique_ptr<MapBase>> map_array_;

public:
  typedef DomainT Domain;
  typedef CodomainT Codomain;

  CodomainT operator()(const DomainT &_domain) const {

    std::shared_ptr<const Manifold> var = &_domain;
    for (const auto &map : map_array_) {
      var = map->value_ptr(var);
    }
    return *dynamic_cast<CodomainT *>(var);
  }

  Eigen::Matrix<double, CodomainT::dim, DomainT::dim>
  diff(const DomainT &_domain) const {

    Eigen::MatrixXd result = Eigen::MatrixXd::Identity();

    std::shared_ptr<const Manifold> var = &_domain;
    for (const auto &map : map_array_) {
      result = map->diff_ptr(var) * result;
      var = map->value_ptr(var);
    }
    return result;
  }
};
*/

template <typename T> class Parametrization;
template <typename T> class Chart {
public:
  virtual Eigen::Matrix<double, T::dim, 1> operator()(const T &x) const = 0;
  virtual Eigen::Matrix<double, T::dim, T::tangent_repr_dim>
  diff(const T &x) const = 0;

  // virtual Parametrization<T> inverse() const = 0;
};

template <typename T> class Parametrization {
  virtual T operator()(const Eigen::Matrix<double, T::dim, 1> &x) const = 0;
  virtual Eigen::Matrix<double, T::tangent_repr_dim, T::dim>
  diff(const Eigen::Matrix<double, T::dim, 1> &x) const = 0;

  //  virtual Chart<T> inverse() const = 0;
};

} // namespace manifolds
