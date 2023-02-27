#include <Eigen/Core>
#include <Manifolds/LinearManifolds.hpp>
#include <Manifolds/Map.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <memory>

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

TEST(Map, Identity) {

  R3 p1(Eigen::Vector3d::Random());
  R3 p3(Eigen::Vector3d::Random());

  /// Test function
  Identity<R3> id;
  R3 p2 = id(p1);
  EXPECT_TRUE(p1.crepr().isApprox(p2.crepr()));
  id(p1, p3);
  EXPECT_TRUE(p1.crepr().isApprox(p3.crepr()));

  /// Test value as representation
  Eigen::Vector3d v = Eigen::Vector3d::Random();
  Eigen::Vector3d w;
  w = id(v); // repr (rerp)
  EXPECT_TRUE(w.isApprox(v));
  Eigen::Vector3d z;
  id(v, z); // (repr, repr)
  EXPECT_TRUE(z.isApprox(v));
  id(p1, v); // (man, rerp)
  EXPECT_TRUE(p1.crepr().isApprox(v));
  id(v, p2); // (rerp, man)
  EXPECT_TRUE(p2.crepr().isApprox(v));
  z = id(p1); // repr = (man)
  EXPECT_TRUE(p1.crepr().isApprox(z));
  R3 p4 = id(z); // man = (repr)
  EXPECT_TRUE(p4.crepr().isApprox(z));

  // Test buffer generator
  std::unique_ptr<ManifoldBase> codom_buff_ptr = id.codomain_buffer();
  *static_cast<R3 *>(codom_buff_ptr.get()) = p1;
  EXPECT_TRUE(
      static_cast<R3 *>(codom_buff_ptr.get())->crepr().isApprox(p1.crepr()));

  std::unique_ptr<ManifoldBase> dom_buff_ptr = id.codomain_buffer();
  *static_cast<R3 *>(dom_buff_ptr.get()) = p1;
  EXPECT_TRUE(
      static_cast<R3 *>(dom_buff_ptr.get())->crepr().isApprox(p1.crepr()));

  Eigen::MatrixXd m1 = id.linearization_buffer();
  EXPECT_EQ(m1.rows(), R3::tangent_repr_dimension);
  EXPECT_EQ(m1.cols(), R3::tangent_repr_dimension);

  // Test differential
  Eigen::Matrix<double, 3, 3> dm1;
  id.diff(p1, dm1);
  EXPECT_TRUE(dm1.isApprox(dm1.Identity()));
  Eigen::Matrix<double, 3, 3> dm = id.diff(p1);
  EXPECT_TRUE(dm.isApprox(dm.Identity()));

  // Test getters
  EXPECT_EQ(id.get_codom_dim(), R3::dimension);
  EXPECT_EQ(id.get_dom_dim(), R3::dimension);
  EXPECT_EQ(id.get_codom_tangent_repr_dim(), R3::tangent_repr_dimension);
  EXPECT_EQ(id.get_dom_tangent_repr_dim(), R3::tangent_repr_dimension);

  // Test MapBase constructor
  std::unique_ptr<ManifoldBase> p1_ptr = std::make_unique<R3>(p1);

  std::unique_ptr<MapBase> mb = std::make_unique<Identity<R3>>();
  std::unique_ptr<ManifoldBase> p2_ptr = mb->value(p1_ptr);
  EXPECT_TRUE(static_cast<R3 *>(p1_ptr.get())
                  ->crepr()
                  .isApprox(static_cast<R3 *>(p2_ptr.get())->crepr()));
  std::unique_ptr<ManifoldBase> p3_ptr = mb->codomain_buffer();
  mb->value(p1_ptr, p3_ptr);
  EXPECT_TRUE(static_cast<R3 *>(p1_ptr.get())
                  ->crepr()
                  .isApprox(static_cast<R3 *>(p3_ptr.get())->crepr()));
  // Test clone
  std::unique_ptr<MapBase> mb2 = mb->clone();
  p2_ptr = mb->value(p1_ptr);
  EXPECT_TRUE(static_cast<R3 *>(p1_ptr.get())
                  ->crepr()
                  .isApprox(static_cast<R3 *>(p2_ptr.get())->crepr()));

  EXPECT_EQ(mb->get_codom_dim(), R3::dimension);
  EXPECT_EQ(mb->get_dom_dim(), R3::dimension);
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
