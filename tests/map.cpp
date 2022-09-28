
#include <Eigen/Core>
#include <Manifolds/LinearManifolds.hpp>
#include <Manifolds/Map.hpp>
#include <gtest/gtest.h>

/* This example shows how to automatically generate a chart in the sphere from
 * two points.
 *
 * The charts are spherical coordinates with inclination and azimuth  with
 * respect to a frame of coordinates generated from two points.
 * Two points are used to generate charts and respective parametrizations.
 * The chart should cover both points.
 *
 *
 *
 *
 * */
using namespace manifolds;

template <typename Set>
class Identity : public MapInheritanceHelper<Identity<Set>, Map<Set, Set>> {
public:
  Identity() = default;
  Identity(const Identity &_that) = default;
  Identity(Identity &&_that) = default;

private:
  bool value_on_repr(const typename Set::Representation &_in,
                     typename Set::Representation &_result) const override {

    _result = _in;
    return true;
  }
  bool diff_from_repr(const typename Set::Representation &,
                      Eigen::MatrixXd &_mat) const override {
    _mat.noalias() = Eigen::MatrixXd::Identity(Set::dim, Set::dim);
    return true;
  }

  //  Eigen::MatrixXd diff(const ManifoldBase &_in) const override {
  //    return Eigen::Matrix<double, Set::dim, Set::dim>::Identity();
  //  }
};

TEST(Map, Identity) {

  Identity<Reals<1>> id;

  Reals<1> p(3);
  Reals<1> p2 = id(p);
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
