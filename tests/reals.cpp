#include <Eigen/Core>
#include <Manifolds/LinearManifolds/Reals.hpp>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace manifolds;

TEST(Reals, Basic) {
  double gt = 9.0;
  Reals a = gt;
  EXPECT_NEAR(a.crepr(), gt, 1.0e-9);

  double b = a;

  EXPECT_NEAR(a.crepr(), b, 1.0e-9);

  auto f = [](const double &a) { return a; };

  EXPECT_NEAR(a.crepr(), f(a), 1.0e-9);

  auto f2 = [](double &a) { a = 8; };
  f2(a);
  EXPECT_NEAR(a.crepr(), 8, 1.0e-9);
}
TEST(Reals, Lifting) {

  Reals::Lifting sin(static_cast<double (*)(double)>(std::sin),
                     static_cast<double (*)(double)>(std::cos));

  Reals::Lifting cos(static_cast<double (*)(double)>(std::cos),
                     [](double in) { return -std::sin(in); });

  Identity<Reals> id;

  MapComposition m(id);

  for (int _ = 0; _ < 100; _++) {
    m = m.compose(sin);
  }
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
