#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Manifolds/Charts.hpp>
#include <Manifolds/Manifold.hpp>
namespace manifolds {

class S2
    : public ManifoldInheritanceHelper<S2, Manifold<Eigen::Vector3d, 2, 3>> {
public:
  S2();
  S2(const Eigen::Vector3d &_vec);
  bool operator==(const S2 &_other) const;
};

class S2Chart : public MapInheritanceHelper<S2Chart, Chart<S2>> {
private:
  const Eigen::Vector3d x_;
  const Eigen::Vector3d y_;
  const Eigen::Vector3d z_;

public:
  S2Chart(const S2 &_p1, const S2 &_p2);
  // Eigen::Matrix<double, 2, 1> value_impl(const S2 &_in) const override;
  bool value_on_repr(const Eigen::Vector3d &_p,
                     Eigen::Vector2d &_x) const override;

private:
  bool diff_from_repr(const Eigen::Vector3d &_in,
                      Eigen::MatrixXd &_mat) const override;
  // Parametrization<S2> inverse() const override;
};

class S2Param : public MapInheritanceHelper<S2Param, Parametrization<S2>> {
private:
  const Eigen::Vector3d x_;
  const Eigen::Vector3d y_;
  const Eigen::Vector3d z_;

public:
  S2Param(const S2 &_p1, const S2 &_p2);
  // S2 value_impl(const Eigen::Matrix<double, 2, 1> &_in) const override;

private:
  bool value_on_repr(const Eigen::Vector2d &_x,
                     Eigen::Vector3d &_p) const override;
  bool diff_from_repr(const Eigen::Vector2d &_in,
                      Eigen::MatrixXd &_mat) const override;

  // Chart<S2> inverse() const;
};
} // namespace manifolds
