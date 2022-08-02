#include <Manifolds/Algebra/VectorSpace.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

/* Test that we get the correct basis*/
using namespace manifolds;

class Rn : public algebra::VectorSpace<Rn, Eigen::VectorXd> {
public:
  using algebra::VectorSpace<Rn, Eigen::VectorXd>::VectorSpace;
};

TEST(Basis, Get_Basis) {}

int main(int argc, char **argv) {

  Rn vector(3);
  Rn vector2(3);

  vector.contains(vector2);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
