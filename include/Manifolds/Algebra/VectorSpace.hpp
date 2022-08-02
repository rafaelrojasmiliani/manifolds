#pragma once

#include <Manifolds/Topology/Open.hpp>
#include <functional>
#include <utility>
#include <vector>

namespace manifolds {
namespace algebra {

template <typename Current> class LinearCombination;

template <typename Current, typename RepresentationT>
class VectorSpace
    : public topology::Open<VectorSpace<Current, RepresentationT>> {

protected:
  RepresentationT element_;

public:
  template <typename... Args>
  VectorSpace(Args... _in)
      : topology::Open<VectorSpace<Current, RepresentationT>>(
            [](const auto &) { return true; }),
        element_(_in...) {}
  /*
    VectorSpace &operator=(const RepresentationT &_in) { element_ = _in; }

    FieldT inner_product(const Current &_lhs) = 0;

    FieldT norm() = 0;

    Current &operator=(const LinearCombination<Current> &_in) {
      return _in.eval();
    }

    LinearCombination<Current> operator*(const FieldT &_lhs) const {
      LinearCombination<Current> result;
      result.array_.push_back({_lhs, *this});
    }*/
};

/*
template <typename Current> class LinearCombination {
private:
  std::vector<
      std::pair<std::reference_wrapper<const Current>,
                std::reference_wrapper<const typename Current::field_t>>>
      array_;

  LinearCombination() = delete;

public:
  LinearCombination<Current> operator+(const Current &_lhs) {
    array_.emplace_back({Current::field_t::FieldUnity, _lhs});
  }
};
*/
} // namespace algebra

} // namespace manifolds
