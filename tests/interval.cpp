#include <Manifolds/Interval.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <random>
#include <unordered_map>

using namespace manifolds;

/// -----------------------------------------------
/// Test points and weights with respect to a table
/// -----------------------------------------------
TEST(Interval, Construction) {

  CanonicInterval t;

  EXPECT_NO_THROW({ t = 1.0; });
  EXPECT_NO_THROW({ t = 0.0; });
  EXPECT_NO_THROW({ t = -1.0; });

  EXPECT_THROW({ t = 1.1; }, std::invalid_argument);

  // test cast to double
  auto f = [](const CanonicInterval &m) { return static_cast<double>(m); };
  double res = f(t);
  EXPECT_NEAR(t.crepr(), res, 1.0e-9);

  // test cast to double
  double m = 0.5;
  res = f(m);
  EXPECT_NEAR(m, res, 1.0e-9);

  // test cast to reals
  Reals r;
  r = t;
  EXPECT_NEAR(r.crepr(), t.crepr(), 1.0e-9);

  EXPECT_THROW({ CanonicInterval(Reals(4.5)); }, std::invalid_argument);

  EXPECT_THROW({ CanonicInterval(4.5); }, std::invalid_argument);

  EXPECT_THROW({ f(4.5); }, std::invalid_argument);

  auto f2 = []() -> CanonicInterval { return 1.0; };

  EXPECT_NO_THROW({ f2(); });

  auto f3 = []() -> CanonicInterval { return 5.0; };
  EXPECT_THROW({ f3(); }, std::invalid_argument);
}

TEST(Interval, Gauss_Lobatto_Weights) {}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
