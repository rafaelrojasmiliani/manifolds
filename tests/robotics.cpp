
#include <gtest/gtest.h>

#include <Manifolds/LinearManifolds/GLPolynomial.hpp>
#include <Manifolds/LinearManifolds/GLPolynomialOperators.hpp>
#include <Manifolds/PinocchioFK.hpp>

using namespace manifolds;

TEST(Robotics, DirectKinematics) {

  auto fk =
      ForwardKinematics<7>::from_urdf("./urdf/panda_arm.urdf", "panda_link0");
  auto q = ForwardKinematics<7>::domain_t::random();

  auto x = fk(q);
}

TEST(Robotics, Motion) {

  constexpr std::size_t intervals = 2;
  constexpr std::size_t nglp = 8;
  constexpr std::size_t dim = 7;

  auto fk =
      ForwardKinematics<dim>::from_urdf("./urdf/panda_arm.urdf", "panda_link0");

  IntervalPartition<intervals> interval(0.0, 2.0);
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;

  auto fk2 = Pol::Composition<DenseLinearManifold<3>>(fk);

  auto q = Pol::random(interval);

  auto x = fk2(q);
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
