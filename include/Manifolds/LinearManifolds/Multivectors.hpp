#pragma once
#include "gcem.hpp"
#include "gcem_incl/binomial_coef.hpp"
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
namespace manifolds {

template <std::size_t K, std::size_t N>
class Multivector
    : public ManifoldInheritanceHelper<
          Multivector<K, N>, LinearManifold<gcem::binomial_coef(N, K)>> {

  template <std::size_t K2>
  Multivector<K + K2, N> wetge(const Multivector<K2, N> _other);

  constexpr static const std::array<
      std::array<std::size_t, gcem::binomial_coef(N, K)>>
      canonic_base_indices std::vector<std::vector<std::size_t>> comb(int K,
                                                                      int N) {

    std::vector<std::vector<std::size_t>> result;
    if (K > N)
      return result;
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0);      // N-K trailing 0's

    // print integers and permute bitmask
    do {
      std::vector<std::size_t> multi_index(K);
      int k = 0;
      for (int i = 0; i < N; ++i) // [0..N-1] integers
      {
        if (bitmask[i]) {
          multi_index[k] = i;
          k++;
        }
      }
      result.push_back(std::move(multi_index));
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    return result;
  }
};

}; // namespace manifolds
