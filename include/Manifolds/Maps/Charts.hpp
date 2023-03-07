#pragma once

#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/Maps/Map.hpp>
namespace manifolds {

//------------------------------------------------------
/// Chart which map an element of the manifold into Rn
//------------------------------------------------------
template <typename Atlas, bool Faithfull>
class Manifold<Atlas, Faithfull>::Chart
    : public MapInheritanceHelper<
          Manifold<Atlas, Faithfull>::Chart,
          Map<Manifold<Atlas, Faithfull>, LinearManifold<Atlas::dimension>>> {

  friend Manifold<Atlas, Faithfull>::Parametrization;

private:
  Representation point1_;
  Representation point2_;

  using DomainRepr = Manifold<Atlas, Faithfull>::Representation;
  using CoDomainRepr =
      typename LinearManifold<Atlas::dimension>::Representation;
  using Element = Manifold<Atlas, Faithfull>;

public:
  Chart(const Element &p1, const Element &p2)
      : point1_(p1.crepr()), point2_(p2.crepr()) {}

  const Representation &point1() const { return point1_; }
  const Representation &point2() const { return point2_; }

private:
  virtual bool value_on_repr(const DomainRepr &_in,
                             CoDomainRepr &_result) const override {
    Atlas::chart(point1_, point2_, _in, _result);
    return true;
  }
  virtual bool
  diff_from_repr(const DomainRepr &_in,
                 Eigen::Ref<Eigen::MatrixXd> &_mat) const override {
    Atlas::chart_diff(point1_, point2_, _in, _mat);

    return true;
  }
};

//------------------------------------------------------
/// Change of coordinates
//------------------------------------------------------
template <typename Atlas, bool Faithfull>
class Manifold<Atlas, Faithfull>::ChangeOfCoordinates
    : public MapInheritanceHelper<
          Manifold<Atlas, Faithfull>::ChangeOfCoordinates,
          Map<LinearManifold<Atlas::dimension>,
              LinearManifold<Atlas::dimension>>> {
private:
  Representation point1A_;
  Representation point2A_;
  Representation point1B_;
  Representation point2B_;

  using DomainRepr = typename LinearManifold<Atlas::dimension>::Representation;
  using CoDomainRepr =
      typename LinearManifold<Atlas::dimension>::Representation;
  using Element = Manifold<Atlas, Faithfull>;

public:
  ChangeOfCoordinates(const Element &p1A, const Element &p2A,
                      const Element &p1B, const Element &p2B)
      : point1A_(p1A.crepr()), point2A_(p2A.crepr()), point1B_(p1B.crepr()),
        point2B_(p2B.crepr()) {}

private:
  virtual bool value_on_repr(const DomainRepr &_in,
                             CoDomainRepr &_result) const override {
    Atlas::change_of_coordinates(point1A_, point2A_, point1B_, point2B_, _in,
                                 _result);
    return true;
  }
  virtual bool
  diff_from_repr(const DomainRepr &_in,
                 Eigen::Ref<Eigen::MatrixXd> &_mat) const override {
    Atlas::change_of_coordinates_diff(point1A_, point2A_, point1B_, point2B_,
                                      _in, _mat);

    return true;
  }
};

//------------------------------------------------------
/// Parametrization: maps Rn into a manifold
//------------------------------------------------------
template <typename Atlas, bool Faithfull>
class Manifold<Atlas, Faithfull>::Parametrization
    : public MapInheritanceHelper<
          Manifold<Atlas, Faithfull>::Parametrization,
          Map<LinearManifold<Atlas::dimension>, Manifold<Atlas, Faithfull>>> {
private:
  Representation point1_;
  Representation point2_;

  using CoDomainRepr = Manifold<Atlas, Faithfull>::Representation;
  using DomainRepr = typename LinearManifold<Atlas::dimension>::Representation;
  using Element = Manifold<Atlas, Faithfull>;

public:
  Parametrization(const Element &p1, const Element &p2)
      : point1_(p1.crepr()), point2_(p2.crepr()) {}

  const Representation &point1() const { return point1_; }
  const Representation &point2() const { return point2_; }

  Manifold<Atlas, Faithfull>::ChangeOfCoordinates
  compose(const Manifold<Atlas, Faithfull>::Chart &chart) const {
    return ChangeOfCoordinates(point1_, point2_, chart.point1(),
                               chart.point2());
  }

private:
  virtual bool value_on_repr(const DomainRepr &_in,
                             CoDomainRepr &_result) const override {
    Atlas::param(point1_, point2_, _in, _result);
    return true;
  }
  virtual bool
  diff_from_repr(const DomainRepr &_in,
                 Eigen::Ref<Eigen::MatrixXd> &_mat) const override {
    Atlas::param_diff(point1_, point2_, _in, _mat);

    return true;
  }
};
} // namespace manifolds
