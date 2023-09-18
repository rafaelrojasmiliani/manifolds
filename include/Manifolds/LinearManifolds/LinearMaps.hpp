#pragma once
#include <Manifolds/Detail.hpp>
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/Maps/Map.hpp>
namespace manifolds {

template <typename Domain, typename Codomain> class SparseLinearMap;

template <typename Domain, typename Codomain>
class DenseLinearMap
    : public detail::Clonable<
          DenseLinearMap<Domain, Codomain>,
          DenseMatrixManifold<Codomain::dimension, Domain::dimension>,
          Map<Domain, Codomain>> {

public:
  using base_t = detail::Clonable<
      DenseLinearMap<Domain, Codomain>,
      DenseMatrixManifold<Codomain::dimension, Domain::dimension>,
      Map<Domain, Codomain>>;
  using manifold_t =
      DenseMatrixManifold<Codomain::dimension, Domain::dimension>;
  using map_t = Map<Domain, Codomain>;

  using base_t::base_t;
  using base_t::operator=;
  DenseLinearMap &operator=(const DenseLinearMap &) = default;
  DenseLinearMap() = default;
  DenseLinearMap(const DenseLinearMap &) = default;
  DenseLinearMap(DenseLinearMap &&) = default;

  /*
  template <typename OtherDomain>
  DenseLinearMap<Codomain, OtherDomain>
  compose(const DenseLinearMap<OtherDomain, Codomain> &_in) const {
    return DenseLinearMap<Codomain, OtherDomain>(base_t::crepr() * _in.crepr());
  }

  template <typename OtherDomain>
  DenseLinearMap<OtherDomain, Codomain>
  compose(const SparseLinearMap<OtherDomain, Domain> &_in) const;
*/
  virtual bool
  value_on_repr(const typename base_t::domain_facade_t &_in,
                typename base_t::codomain_facade_t &_out) const override {
    const typename base_t::domain_t::Representation *input_ptr;
    typename base_t::codomain_t::Representation *output_ptr;
    if constexpr (base_t::domain_t::is_faithful)
      input_ptr = &_in;
    else
      input_ptr = &(_in.crepr());
    if constexpr (base_t::codomain_t::is_faithful)
      output_ptr = &_out;
    else
      output_ptr = &(this->get_repr(_out));
    *output_ptr = this->crepr() * (*input_ptr);
    return true;
  }

  virtual bool diff_from_repr(const typename base_t::domain_facade_t &_in,
                              typename base_t::codomain_facade_t &_out,
                              detail::dense_matrix_ref_t _mat) const override {
    this->value_on_repr(_in, _out);
    _mat = this->crepr();
    return true;
  }
};
template <typename Domain, typename Codomain>
class SparseLinearMap
    : public detail::Clonable<
          SparseLinearMap<Domain, Codomain>,
          SparseMatrixManifold<Codomain::dimension, Domain::dimension>,
          Map<Domain, Codomain, detail::MatrixTypeId::Sparse>> {

public:
  using map_t = Map<Domain, Codomain, detail::MatrixTypeId::Sparse>;
  using manifold_t =
      SparseMatrixManifold<Codomain::dimension, Domain::dimension>;
  using base_t = detail::Clonable<
      SparseLinearMap<Domain, Codomain>,
      SparseMatrixManifold<Codomain::dimension, Domain::dimension>,
      Map<Domain, Codomain, detail::MatrixTypeId::Sparse>>;

  using base_t::base_t;
  using base_t::operator=;
  SparseLinearMap() = default;
  SparseLinearMap(const SparseLinearMap &) = default;
  SparseLinearMap(SparseLinearMap &&) = default;
  /*
    template <typename OtherDomain>
    DenseLinearMap<Codomain, OtherDomain>
    compose(const DenseLinearMap<OtherDomain, Domain> &_in) const {
      return DenseLinearMap<OtherDomain, Codomain>(base_t::crepr() *
    _in.crepr());
    }

    template <typename OtherDomain>
    SparseLinearMap<OtherDomain, Codomain>
    compose(const SparseLinearMap<OtherDomain, Domain> &_in) const {
      return DenseLinearMap<OtherDomain, Codomain>(base_t::crepr() *
    _in.crepr());
    }
  */
  virtual bool
  value_on_repr(const typename base_t::domain_facade_t &_in,
                typename base_t::codomain_facade_t &_out) const override {
    const typename base_t::domain_t::Representation *input_ptr;
    typename base_t::codomain_t::Representation *output_ptr;
    if constexpr (base_t::domain_t::is_faithful)
      input_ptr = &_in;
    else
      input_ptr = &(_in.crepr());
    if constexpr (base_t::codomain_t::is_faithful)
      output_ptr = &_out;
    else
      output_ptr = &(this->get_repr(_out));
    *output_ptr = this->crepr() * (*input_ptr);
    return true;
  }

  virtual bool diff_from_repr(const typename base_t::domain_facade_t &_in,
                              typename base_t::codomain_facade_t &_out,
                              detail::sparse_matrix_ref_t _mat) const override {
    this->value_on_repr(_in, _out);
    _mat.get() = this->crepr();
    _mat.get().makeCompressed();
    return true;
  }
};

template <typename Domain>
class AlgebraicMap
    : public Map<Domain, Domain, detail::decide_matrix_type<Domain, Domain>()> {

private:
  Domain value_;
  AlgebraicMap(const Domain &_value) : value_(_value) {}
  AlgebraicMap(Domain &&_value) : value_(std::move(_value)) {}
  const Domain &get_value() const { return value_; }
};

template <typename Domain>
class Plus : public detail::Clonable<Plus<Domain>, AlgebraicMap<Domain>> {

public:
  using base_t = detail::Clonable<Plus<Domain>, AlgebraicMap<Domain>>;

  Plus(const Domain &_value) : base_t(_value) {}

  virtual bool
  value_on_repr(const typename base_t::domain_facade_t &_in,
                typename base_t::codomain_facade_t &_out) const override {

    const typename base_t::domain_t::Representation *input_ptr;
    typename base_t::codomain_t::Representation *output_ptr;

    if constexpr (base_t::domain_t::is_faithful)
      input_ptr = &_in;
    else
      input_ptr = &(_in.crepr());
    if constexpr (base_t::codomain_t::is_faithful)
      output_ptr = &_out;
    else
      output_ptr = &(this->get_repr(_out));
    *output_ptr = this->get_value().crepr() + (*input_ptr);
    return true;
  }

  virtual bool
  diff_from_repr(const typename base_t::domain_facade_t &_in,
                 typename base_t::codomain_facade_t &_out,
                 typename base_t::differential_ref_t _mat) const override {
    this->value_on_repr(_in, _out);

    if constexpr (base_t::differential_type_id ==
                  detail::MatrixTypeId::Sparse) {
      for (long i = 0; i < (long)base_t::domain_t::dimension; i++)
        _mat.get().coeffRef(i, i) = 1.0;
      _mat.get().makeCompressed();
    } else {
      for (long i = 0; i < (long)base_t::domain_t::dimension; i++)
        _mat(i, i) = 1.0;
    }
    return true;
  }
};

template <typename Domain, typename Codomain>
class SparseAffineMap
    : public detail::Clonable<
          Map<Domain, Codomain, detail::MatrixTypeId::Sparse>> {

private:
  detail::sparse_matrix_t mat_;
  Eigen::Matrix<double, Codomain::dimension, 1> vec_;

public:
  using map_t = Map<Domain, Codomain, detail::MatrixTypeId::Sparse>;

  using base_t =
      detail::Clonable<Map<Domain, Codomain, detail::MatrixTypeId::Sparse>>;

  virtual bool
  value_on_repr(const typename base_t::domain_facade_t &_in,
                typename base_t::codomain_facade_t &_out) const override {
    const typename base_t::domain_t::Representation *input_ptr;
    typename base_t::codomain_t::Representation *output_ptr;
    if constexpr (base_t::domain_t::is_faithful)
      input_ptr = &_in;
    else
      input_ptr = &(_in.crepr());
    if constexpr (base_t::codomain_t::is_faithful)
      output_ptr = &_out;
    else
      output_ptr = &(this->get_repr(_out));
    *output_ptr = mat_ * (*input_ptr) + vec_;
    return true;
  }

  virtual bool diff_from_repr(const typename base_t::domain_facade_t &_in,
                              typename base_t::codomain_facade_t &_out,
                              detail::sparse_matrix_ref_t _mat) const override {
    this->value_on_repr(_in, _out);
    _mat.get() = mat_;
    _mat.get().makeCompressed();
    return true;
  }
};
/*
template <typename Domain, typename Codomain>
template <typename OtherDomain>
DenseLinearMap<OtherDomain, Codomain> DenseLinearMap<Domain, Codomain>::compose(
    const SparseLinearMap<OtherDomain, Domain> &_in) const {
  return DenseLinearMap<OtherDomain, Codomain>(base_t::crepr() * _in.crepr());
}
*/
} // namespace manifolds
