

#include <Manifolds/LinearManifolds/Polynomials.hpp>
#include <eigen3/unsupported/Eigen/Polynomials>
#include <gtest/gtest.h>
#include <numeric> // std::iota
#include <random>

using namespace manifolds;
TEST(Reals, Basic) {
  Eigen::Vector3d coeff(-0.2151830138973625, -0.3111717537041549,
                        0.708563939215852);

  CanonicPolynomial<3> pol = coeff;
  double result_ground_thruth = Eigen::poly_eval(coeff, 0.1);
  double result_tes = pol(0.1);
  EXPECT_NEAR(result_ground_thruth, result_tes, 1.0e-9);
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
