#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Manifolds/Manifold.hpp>
#include <Manifolds/Map.hpp>
namespace manifolds {

class S2 : public Manifold<Eigen::Vector3d, 2, Eigen::Vector3d, 3> {
public:
  S2(const Eigen::Vector3d &_vec);
  bool operator==(const S2 &_other) const;
};

class S2Chart : public Chart<S2> {
private:
  const Eigen::Vector3d x_;
  const Eigen::Vector3d y_;
  const Eigen::Vector3d z_;

public:
  S2Chart(const S2 &_p1, const S2 &_p2);
  Eigen::Matrix<double, 2, 1> operator()(const S2 &x) const override;
  Eigen::Matrix<double, 2, 3> diff(const S2 &x) const override;

  // Parametrization<S2> inverse() const override;
};

class S2Param : public Parametrization<S2> {
private:
  const Eigen::Vector3d x_;
  const Eigen::Vector3d y_;
  const Eigen::Vector3d z_;

public:
  S2Param(const S2 &_p1, const S2 &_p2);
  S2 operator()(const Eigen::Matrix<double, S2::dim, 1> &x) const;

  Eigen::Matrix<double, S2::tangent_repr_dim, S2::dim>
  diff(const Eigen::Matrix<double, S2::dim, 1> &x) const;

  // Chart<S2> inverse() const;
};
} // namespace manifolds
