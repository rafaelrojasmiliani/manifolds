#pragma once
#include <Eigen/Core>
#include <functional>
#include <utility>
namespace manifolds {

namespace topology {

template <typename ManifoldT> class Open {
private:
  std::function<bool(const ManifoldT &)> contains_;

public:
  Open(std::function<bool(const ManifoldT &)> _contains)
      : contains_(_contains) {}
  typedef ManifoldT Manifold;
  bool contains(const ManifoldT &_in) const { return contains_(_in); }
};

template <long Dimension> class CartensianInterval {
  std::vector<std::pair<double, double>> intervals_;

public:
  const static long dim = Dimension;

  CartensianInterval(const std::vector<std::pair<double, double>> &_intervals)
      : intervals_(_intervals) {}

  bool contains(const Eigen::Matrix<double, Dimension, 1> &_in) const {

    if ((long)intervals_.size() != _in.size())
      return false;
    for (long i = 0; i < _in.size(); i++) {
      if (_in(i) < intervals_.at(i).first or intervals_.at(i).second < _in(i))
        return false;
    }
    return true;
  }
};
} // namespace topology
} // namespace manifolds
