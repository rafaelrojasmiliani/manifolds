#pragma once
#include <Manifolds/Detail.hpp>
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/problem.h>
#include <ifopt/variable_set.h>

namespace manifolds {

namespace optimizer {

template <std::size_t Dim>
DenseLinearManifold<Dim> get_initial_from_affine_ctr() {}

template <std::size_t Dim>
class ManifoldVariable : public ifopt::VariableSet,
                         public DenseLinearManifold<Dim> {
public:
  using manifold_t = DenseLinearManifold<Dim>;
  using manifold_t::operator=;
  ManifoldVariable() : ifopt::VariableSet(Dim, "variable"), manifold_t() {

    ifopt::Bounds default_bound(-ifopt::inf, ifopt::inf);
    bounds_ = ifopt::Component::VecBound(Dim, default_bound);
  }
  void SetVariables(const Eigen::VectorXd &_vec) override {
    this->repr() = _vec;
  }
  Eigen::VectorXd GetValues() const override {
    // std::cout << "var get values \n";
    // fflush(stdout);
    return this->crepr();
  }

  ifopt::Component::VecBound GetBounds() const override {

    // std::cout << "var get Bounds \n";
    // fflush(stdout);
    return bounds_;
  }

  std::unique_ptr<ManifoldVariable> clone() const & {
    return std::make_unique<ManifoldVariable>(this->clone());
  }

  std::unique_ptr<ManifoldVariable> clone() && {
    return std::make_unique<ManifoldVariable>(this->move_clone_impl());
  }

  std::unique_ptr<ManifoldVariable> move_clone() {
    return std::make_unique<ManifoldVariable>(this->move_clone_impl());
  }
  ~ManifoldVariable() override = default;

private:
  ifopt::Component::Component::VecBound bounds_;

  virtual ManifoldVariable *clone_impl() const override {
    return new ManifoldVariable(*this);
  }

  virtual ManifoldVariable *move_clone_impl() override {
    return new ManifoldVariable(std::move(*this));
  }
};

template <typename Map> class Cost : public ifopt::CostTerm, public Map {
  static_assert(std::is_same_v<typename std::decay_t<Map>::codomain_t, Reals>);
  static_assert(std::is_same_v<
                typename std::decay_t<Map>::domain_t,
                DenseLinearManifold<std::decay_t<Map>::domain_t::dimension>>);

public:
  using codomain_t = Reals;
  using domain_t = DenseLinearManifold<std::decay_t<Map>::domain_t::dimension>;

  Cost(const Map &_map) : CostTerm("cost"), Map(_map) {}
  Cost(Map &&_map) : CostTerm("cost"), Map(std::move(_map)) {}

  double GetCost() const override {
    // std::cout << "value cost\n";
    // fflush(stdout);
    Reals output;
    input_buff_.repr().noalias() =
        this->GetVariables()->GetComponent("variable")->GetValues();
    this->value(input_buff_, output);
    return output;
  }
  void FillJacobianBlock(std::string,
                         ifopt::CostTerm::Jacobian &jac) const override {

    // std::cout << "jac cost\n";
    // fflush(stdout);
    Reals output;
    input_buff_.repr().noalias() =
        this->GetVariables()->GetComponent("variable")->GetValues();

    // std::cout << "jac got values\n";
    // fflush(stdout);
    if constexpr (Map::differential_type_id == detail::MatrixTypeId::Sparse) {
      // std::cout << "jac eval sparse\n";
      // fflush(stdout);
      this->diff(input_buff_, output_buff_, jac);
    } else {
      // std::cout << "jac eval dense\n";
      // fflush(stdout);
      this->diff(input_buff_, output_buff_, jac_buff_);
      // std::cout << "going into loop\n";
      // fflush(stdout);
      for (long i = 0; i < jac_buff_.rows(); i++)
        for (long j = 0; j < jac_buff_.cols(); j++)
          jac.coeffRef(i, j) = jac_buff_(i, j);
    }
    // std::cout << "jac cost end\n";
    // fflush(stdout);
  }

  std::unique_ptr<Cost> clone() const & {
    return std::unique_ptr<Cost>(static_cast<Cost *>(this->clone_impl()));
  }
  std::unique_ptr<Cost> clone() && { return move_clone(); }

  std::unique_ptr<Cost> move_clone() {
    return std::unique_ptr<Cost>(static_cast<Cost *>(this->move_clone_impl()));
  }
  ~Cost() override = default;

private:
  mutable domain_t input_buff_;
  mutable codomain_t output_buff_;
  mutable Eigen::Matrix<double, 1, domain_t::dimension> jac_buff_;

  virtual Cost *clone_impl() const override { return new Cost(*this); }
  virtual Cost *move_clone_impl() override {
    return new Cost(std::move(*this));
  }
};

template <typename Map>
class Constraint : public Map, public ifopt::ConstraintSet {
public:
  static_assert(std::is_same_v<
                typename std::decay_t<Map>::codomain_t,
                DenseLinearManifold<std::decay_t<Map>::codomain_t::dimension>>);
  static_assert(std::is_same_v<
                typename std::decay_t<Map>::domain_t,
                DenseLinearManifold<std::decay_t<Map>::domain_t::dimension>>);

  using domain_t = DenseLinearManifold<std::decay_t<Map>::domain_t::dimension>;
  using codomain_t =
      DenseLinearManifold<std::decay_t<Map>::codomain_t::dimension>;

  Constraint(const Map &_map)
      : Map(_map), ifopt::ConstraintSet(codomain_t::dimension, gen_random(10)) {

    ifopt::Bounds default_bound(0.0, 0.0);
    bounds_ = ifopt::Component::VecBound(codomain_t::dimension, default_bound);
  }
  Constraint(Map &&_map)
      : Map(std::move(_map)),
        ifopt::ConstraintSet(codomain_t::dimension, gen_random(10)) {

    ifopt::Bounds default_bound(0.0, 0.0);
    bounds_ = ifopt::Component::VecBound(codomain_t::dimension, default_bound);
  }

  Eigen::VectorXd GetValues() const override {

    // std::cout << "value\n";
    // fflush(stdout);
    input_buff_.repr().noalias() =
        this->GetVariables()->GetComponent("variable")->GetValues();
    this->value(input_buff_, output_buff_);
    return output_buff_.crepr();
  }

  ifopt::Component::VecBound GetBounds() const override {

    // std::cout << "var constraint bounds\n";
    // fflush(stdout);
    return bounds_;
  }

  void FillJacobianBlock(std::string _var,
                         ifopt::ConstraintSet::Jacobian &jac) const override {

    if (_var != "variable")
      throw std::logic_error("");
    // std::cout << "constranit jac eval\n";
    // fflush(stdout);
    input_buff_.repr().noalias() =
        this->GetVariables()->GetComponent("variable")->GetValues();
    // std::cout << "constranit jac got variables\n";
    // fflush(stdout);
    if constexpr (Map::differential_type_id == detail::MatrixTypeId::Sparse) {
      // std::cout << "constranit jac getting sparse jackn\n";
      // fflush(stdout);
      this->diff(input_buff_, output_buff_, jac);
      // std::cout << "constranit jac got sparse jac\n";
      // fflush(stdout);
    } else {
      // std::cout << "constranit jac getting dense jack\n";
      // fflush(stdout);
      this->diff(input_buff_, output_buff_, jac_buff_);
      for (long i = 0; i < jac_buff_.rows(); i++)
        for (long j = 0; j < jac_buff_.cols(); j++)
          jac.coeffRef(i, j) = jac_buff_(i, j);
      // std::cout << "constranit jac got dense jack\n";
      // fflush(stdout);
    }
  }

  // ----

  std::unique_ptr<Constraint> clone() const & {
    return std::unique_ptr<Constraint>(this->clone_impl());
  }
  std::unique_ptr<Constraint> clone() && { return this->move_clone(); }

  std::unique_ptr<Constraint> move_clone() {
    return std::unique_ptr<Constraint>(this->move_clone_impl());
  }

private:
  mutable domain_t input_buff_;
  mutable codomain_t output_buff_;
  mutable Eigen::Matrix<double, 1, domain_t::dimension> jac_buff_;
  ifopt::Component::Component::VecBound bounds_;

  virtual Constraint *clone_impl() const override {
    return new Constraint(*this);
  }
  virtual Constraint *move_clone_impl() override {
    return new Constraint(std::move(*this));
  }
  inline static std::random_device rd;
  inline static std::mt19937 mt = std::mt19937(rd());

  static std::string gen_random(const int len) {
    static const char alphanum[] = "0123456789"
                                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                   "abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<std::size_t> uint_dist(0,
                                                         sizeof(alphanum) - 2);
    std::string tmp_s;
    tmp_s.reserve(len);

    for (int i = 0; i < len; ++i) {
      tmp_s += alphanum[uint_dist(mt)];
    }

    return tmp_s;
  }
};

class Problem {
public:
  template <typename CostMap> Problem(const Cost<CostMap> &_cost) {

    using cost_t = std::decay_t<CostMap>;
    auto cost_ptr = std::shared_ptr<ifopt::CostTerm>(_cost.clone().release());
    auto var_ptr =
        std::make_shared<ManifoldVariable<cost_t::domain_t::dimension>>();
    problem_.AddVariableSet(var_ptr);
    problem_.AddCostSet(cost_ptr);
  }

  template <typename CostMap> Problem(Cost<CostMap> &&_cost) {
    using cost_t = std::decay_t<CostMap>;
    auto cost_ptr =
        std::shared_ptr<ifopt::CostTerm>(_cost.move_clone().release());
    auto var_ptr =
        std::make_shared<ManifoldVariable<cost_t::domain_t::dimension>>();
    problem_.AddVariableSet(var_ptr);
    problem_.AddCostSet(cost_ptr);
  }

  template <typename M> Problem &operator&&(Constraint<M> &&_c) {

    auto c_ptr =
        std::shared_ptr<ifopt::ConstraintSet>(_c.move_clone().release());

    problem_.AddConstraintSet(c_ptr);

    return *this;
  }

  template <typename M> Problem &operator&&(const Constraint<M> &_c) {

    auto c_ptr = std::shared_ptr<ifopt::ConstraintSet>(_c.clone().release());

    problem_.AddConstraintSet(c_ptr);

    return *this;
  }

  Eigen::VectorXd solve_ipopt() {
    ifopt::IpoptSolver ipopt;
    // 3.1 Customize the solver
    ipopt.SetOption("linear_solver", "mumps");
    ipopt.SetOption("jacobian_approximation", "exact");
    // ipopt.SetOption("fast_step_computation", "yes");
    ipopt.SetOption("max_iter", 1000);
    ipopt.SetOption("derivative_test", "first-order");
    ipopt.SetOption("hessian_approximation", "limited-memory");
    ipopt.SetOption("print_level", 5);

    // 4. Ask the solver to solve the problem
    ipopt.Solve(problem_);
    return problem_.GetOptVariables()->GetValues();
  }

private:
  ifopt::Problem problem_;
};

} // namespace optimizer

} // namespace manifolds
