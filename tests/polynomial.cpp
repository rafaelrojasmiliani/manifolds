
#include <gtest/gtest.h>

#include <Manifolds/LinearManifolds/GLPolynomial.hpp>
#include <Manifolds/LinearManifolds/GLPolynomialOperators.hpp>

using namespace manifolds;

TEST(GLPolynomial, LagrangePolynomial) {

  IntervalPartition<1> interval(-1, 1);
  using Pol = PWGLVPolynomial<12, 1, 1>;

  Pol p(interval);
  (void)p;
  p = Pol::zero(interval);

  for (int k = 0; k < 100; k++) {
    double t = p.get_domain().get_random();
    double val = 0.0;
    for (int i = 0; i < 12; i++) {
      p[i](0) = 1.0;
      val += p(t).crepr()(0);
      p[i](0) = 0.0;
    }
    EXPECT_LT(std::fabs(val - 1.0), 1.0e-12);
  }
}

TEST(GLPolynomial, DerivativeEvaluation) {

  using Pol = PWGLVPolynomial<12, 20, 9>;

  Pol p({0.0, 100.0});
  (void)p;
  Pol p2({0.0, 100.0});
  Pol::codomain_t val = Pol::codomain_t::random();
  p = Pol::constant({0.0, 100.0}, val);

  for (int k = 0; k < 100; k++) {
    double t = p.get_domain().get_random();

    Pol::codomain_t x = p(t);

    double d = p.diff(t)(0);
    EXPECT_LT(std::fabs(d - 0.0), 1.0e-12);
  }

  using ScalarPol = PWGLVPolynomial<12, 20, 1>;
  ScalarPol q = ScalarPol::identity({0.0, 100.0});
  for (int k = 0; k < 100; k++) {

    double t = q.get_domain().get_random();

    ScalarPol::codomain_t x = q(t);
    EXPECT_LT(std::fabs(x.crepr()(0) - t), 1.0e-10);

    double d = q.diff(t)(0);
    EXPECT_LT(std::fabs(d - 1.0), 1.0e-10);
  }
} /*

 TEST(GLPolynomial, DerivativeOperator) {
   using Pol = PWGLVPolynomial<10, 4, 7>;

   IntervalPartition<4> interval(0.0, 100.0);
   Pol pol = Pol::constant(interval, Eigen::Vector<double, 7>::Random());

   EXPECT_EQ(pol.get_dim(), 10 * 4 * 7);
   Pol::Derivative<1> deriv(interval);

   Pol deriv_pol(interval);
   deriv_pol = deriv(pol);

   EXPECT_TRUE(deriv_pol == Pol::zero(interval));
   Pol random_pol(interval);
   random_pol = *Pol::random();
   Pol random_pol_derivative(interval);
   random_pol_derivative = deriv(random_pol);
   for (int k = 0; k < 100; k++) {

     double t = random_pol.get_domain().get_random();

     Eigen::VectorXd x = random_pol.diff(t);
     Eigen::VectorXd y = random_pol_derivative(t).crepr();

     EXPECT_TRUE(y.isApprox(x));
   }
 }
 */
/*

TEST(GLPolynomial, ContinuityMatrix) {
  constexpr std::size_t intervals = 5;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 5;
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;

  IntervalPartition<intervals> intpart({0.0, 100.0});
  auto mat = Pol::continuity_matrix(2);

  Pol pol(intpart);
  Pol pold(intpart);
  Pol poldd(intpart);
  pol = Pol::random_projection();
  pold = Pol::Derivative<1>(intpart)(pol);
  poldd = Pol::Derivative<2>(intpart)(pol);

  auto vec = mat * pol.crepr();

  for (std::size_t k = 0; k < intervals - 1; k++)
    EXPECT_TRUE(vec.segment(k * dim, dim)
                    .isApprox(pol[(k + 1) * nglp - 1] - pol[(k + 1) * nglp]));

  for (std::size_t k = 0; k < intervals - 1; k++)
    EXPECT_TRUE((vec.segment((intervals - 1) * dim + k * dim, dim) * 2.0 /
                 intpart.subinterval_length(0))
                    .isApprox(pold[(k + 1) * nglp - 1] - pold[(k + 1) * nglp]));

  for (std::size_t k = 0; k < intervals - 1; k++) {
    Eigen::Vector<double, dim> dx =
        vec.segment(2 * (intervals - 1) * dim + k * dim, dim) *
        std::pow(2.0 / intpart.subinterval_length(0), 2.0);
    EXPECT_TRUE(dx.isApprox(poldd[(k + 1) * nglp - 1] - poldd[(k + 1) * nglp]));
  }
}

TEST(GLPolynomial, ImmersionMatrix) {
  constexpr std::size_t intervals = 3;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 2;
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;

  auto mat = Pol::continuity_matrix(2);
  auto basis = Pol::Continuous<2>::canonical_immersion();

  Eigen::SparseMatrix<double, Eigen::RowMajor> zero = mat * basis;

  EXPECT_LT(zero.coeffs().abs().maxCoeff(), 1.0e-9);
}

TEST(GLPolynomial, Immersion) {
  constexpr std::size_t intervals = 4;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 3;
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;
  IntervalPartition<intervals> intpart({0.0, 100.0});

  auto pol = Pol(intpart);
  auto pold = Pol(intpart);
  auto poldd = Pol(intpart);
  auto pc = Pol::Continuous<2>(intpart);

  pc = Pol::Continuous<2>::random_projection();

  auto im = Pol::Continuous<2>::Immersion(intpart);

  pol = im(pc);
  pold = Pol::Derivative<1>(intpart)(pol);
  poldd = Pol::Derivative<2>(intpart)(pol);

  for (std::size_t k = 0; k < intervals - 1; k++) {
    EXPECT_TRUE(pol[(k + 1) * nglp - 1].isApprox(pol[(k + 1) * nglp]));
    EXPECT_TRUE(pold[(k + 1) * nglp - 1].isApprox(pold[(k + 1) * nglp]));
    EXPECT_TRUE(poldd[(k + 1) * nglp - 1].isApprox(poldd[(k + 1) * nglp]));
  }
}

TEST(GLPolynomial, Projection) {
  constexpr std::size_t intervals = 4;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 3;
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;
  IntervalPartition<intervals> intpart({0.0, 100.0});
  Pol pol(intpart);
  Pol pold(intpart);
  Pol poldd(intpart);
  pol = Pol::random_projection();

  auto proj = Pol::Continuous<2>::euclidean_projector();
  pol = proj * pol.crepr();
  pold = Pol::Derivative<1>(intpart)(pol);
  poldd = Pol::Derivative<2>(intpart)(pol);
}
*/

/*
TEST(GLPolynomial, Lifting) {
  constexpr std::size_t intervals = 4;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 3;
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;
  Pol::Composition<DenseLinearManifold<1>>(
      [](const Eigen::Vector<double, 3> &, Eigen::Vector<double, 1> &) {
        return true;
      },
      [](const Eigen::Vector<double, 3> &, Eigen::Ref<Eigen::MatrixXd>) {
        return true;
      });
}*/

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
