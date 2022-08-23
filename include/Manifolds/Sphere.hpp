#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Manifolds/Manifold.hpp>
#include <Manifolds/Map.hpp>
namespace manifolds {

class S2 : public Manifold<Eigen::Vector3d, 2, Eigen::Vector3d, 3> {
public:
  S2(const Eigen::Vector3d &_vec) : Manifold(_vec.normalized()) {}
};

class S2Chart : public Chart<S2> {
public:
  Eigen::Matrix<double, 2, 1> operator()(const S2 &x);
  const Eigen::Matrix<double, 2, 3> diff(const S2 &x) const;
};
} // namespace manifolds
