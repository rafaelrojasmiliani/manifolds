#pragma once

#include <Manifolds/Manifold.hpp>
#include <Manifolds/Topology/Open.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <list>
#include <memory>

namespace manifolds {

class MapBaseComposition;

class MapBase {

public:
  std::unique_ptr<ManifoldBase> value_ptr(const ManifoldBase &_in) const {
    return std::unique_ptr<ManifoldBase>(value_ptr_impl(_in));
  }
  std::unique_ptr<MapBase> clone() const {
    return std::unique_ptr<MapBase>(clone_impl());
  }

  std::unique_ptr<MapBase> move_clone() {
    return std::unique_ptr<MapBase>(move_clone_impl());
  }

  // virtual Eigen::MatrixXd diff_ptr(const ManifoldBase &_in) const = 0;
  virtual ~MapBase() = default;
  MapBase(const MapBase &) = default;
  MapBase(MapBase &&) = default;
  MapBase() = default;

private:
  virtual ManifoldBase *value_ptr_impl(const ManifoldBase &) const = 0;

protected:
  virtual MapBase *clone_impl() const = 0;
  virtual MapBase *move_clone_impl() = 0;
};

template <typename Current> class MapInheritanceHelper : public MapBase {
public:
  MapInheritanceHelper() = default;
  MapInheritanceHelper(const MapInheritanceHelper &_that) = default;
  MapInheritanceHelper(MapInheritanceHelper &&_that) = default;
  virtual ~MapInheritanceHelper() = default;

protected:
  virtual MapBase *clone_impl() const override {

    return new Current(*static_cast<const Current *>(this));
  }

  virtual MapBase *move_clone_impl() override {
    return new Current(std::move(*static_cast<Current *>(this)));
  }
};

template <class T> auto SingletonVector(T &&x) {
  std::vector<std::decay_t<T>> ret;
  ret.push_back(std::forward<T>(x));
  return ret;
}

class MapBaseComposition : public MapInheritanceHelper<MapBaseComposition> {
public:
  MapBaseComposition() = delete;
  MapBaseComposition(const MapBaseComposition &_that) {
    for (const auto &map : _that.maps_)
      maps_.push_back(map->clone());
  }
  MapBaseComposition(MapBaseComposition &&_that) {
    for (const auto &map : _that.maps_)
      maps_.push_back(map->move_clone());
  }

  MapBaseComposition(const MapBase &_in) : MapInheritanceHelper(), maps_() {
    maps_.push_back(_in.clone());
  }
  MapBaseComposition(MapBase &&_in) : MapInheritanceHelper(), maps_() {
    maps_.push_back(_in.move_clone());
  }

  ManifoldBase *value_ptr_impl(const ManifoldBase &_in) const override {

    std::unique_ptr<ManifoldBase> res1 = maps_.front()->value_ptr(_in);
    std::unique_ptr<ManifoldBase> res2(nullptr);
    auto it = maps_.begin();
    std::advance(it, 1); // or, use `++it`

    while (it != maps_.end()) {
      res2 = (*it)->value_ptr(*res1);
      res1 = std::move(res2);
      ++it;
    }

    return res1.release();
  }

private:
  std::vector<std::unique_ptr<MapBase>> maps_;
};

template <typename DomainType, typename CoDomainType>
class Map : public MapInheritanceHelper<Map<DomainType, CoDomainType>> {
public:
  std::unique_ptr<CoDomainType> value_ptr() const {
    return std::unique_ptr<CoDomainType>(
        static_cast<CoDomainType *>(this->value_ptr_impl()));
  }
  CoDomainType operator()(const DomainType &_in) const {
    return *static_cast<CoDomainType *>(value_ptr_impl(_in));
  }
};
/*

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
