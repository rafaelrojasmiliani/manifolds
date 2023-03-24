#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <gcem.hpp>
#include <iostream>
#include <iterator>
#include <ostream>
#include <tuple>
#include <utility>
namespace manifolds::collocation::detail {

template <std::size_t N>
constexpr std::tuple<double, double, double> q_and_evaluation(double _t) {

  /*
   *  David A. Kopriva
   *  Implementing Spectral
   *  Methods for Partial
   *  Differential Equations
   *  Algorithm 24: qAndLEvaluation
   * */
  double l_n_minus_2 = 1.0;
  double l_n_minus_1 = _t;
  double lprime_n_minus_2 = 0.0;
  double lprime_n_minus_1 = 1.0;
  double l_n = 0;
  double lprime_n = 0;
  double lprime_n_plus_1 = 0;
  double l_n_plus_1 = 0;
  double q = 0;
  double qprime = 0;

  std::size_t uick = 0;

  for (uick = 2; uick < N + 1; uick++) {

    l_n = (2.0 * (double)uick - 1.0) / ((double)uick) * _t * l_n_minus_1 -
          ((double)uick - 1.0) / ((double)uick) * l_n_minus_2;

    lprime_n = lprime_n_minus_2 + (2.0 * (double)uick - 1.0) * l_n_minus_1;

    l_n_minus_2 = l_n_minus_1;
    l_n_minus_1 = l_n;

    lprime_n_minus_2 = lprime_n_minus_1;
    lprime_n_minus_1 = lprime_n;
  }

  l_n_plus_1 = (2.0 * (double)uick - 1.0) / ((double)uick) * _t * l_n -
               ((double)uick - 1.0) / ((double)uick) * l_n_minus_2;

  lprime_n_plus_1 = lprime_n_minus_2 + (2.0 * (double)uick - 1.0) * l_n;

  q = l_n_plus_1 - l_n_minus_2;

  qprime = lprime_n_plus_1 - lprime_n_minus_2;

  return std::tuple<double, double, double>(q, qprime, l_n);
}

template <typename Function, std::size_t... Indices>
constexpr auto make_array_helper(Function f, std::index_sequence<Indices...>)
    -> std::array<typename std::result_of<Function(int)>::type,
                  sizeof...(Indices)> {
  return {f(Indices)...};
}

template <int N, typename Function>
constexpr auto make_array(Function f)
    -> std::array<typename std::result_of<Function(std::size_t)>::type, N> {
  return make_array_helper(f, std::make_index_sequence<N>{});
}

template <std::size_t book_n>
constexpr double refine_glp(double point, std::size_t n) {

  if (n == 0)
    return point;

  auto m = q_and_evaluation<book_n>(point);
  double q = std::get<0>(m);
  double q_prime = std::get<1>(m);

  double delta = -q / q_prime;

  double result = point + delta;

  if (gcem::abs(delta) < 1.0e-12 * gcem::abs(point))
    return result;

  return refine_glp<book_n>(result, n - 1);
}

template <std::size_t book_n>
constexpr std::pair<double, double> compute_single_glp(int uicj) {

  /// Elementary cases
  if (uicj == 0 and book_n == 1)
    return {-1, 1.0};
  if (uicj == 0)
    return {-1, 2.0 / ((double)book_n * ((double)book_n + 1.0))};

  // Compute GL point
  double arg =
      (((double)uicj + 0.25) * M_PI / (double)book_n) -
      (3.0 / (8.0 * (double)book_n * M_PI)) * (1.0 / ((double)uicj + 0.25));

  double point = refine_glp<book_n>(-gcem::cos(arg), 100);
  std::pair<double, double> result;
  result.first = point;

  // Compute GL weight
  auto m = q_and_evaluation<book_n>(result.first);

  result.second = 2.0 / ((double)book_n * ((double)book_n + 1.0) *
                         std::get<2>(m) * std::get<2>(m));
  return result;
}

template <typename T, std::size_t NumPoints>
constexpr std::array<T, NumPoints>
concat_pairs(std::array<T, static_cast<int>(((double)(NumPoints)) / 2.0)> lhs) {
  std::array<T, NumPoints> result{};
  std::size_t index = 0;

  // constexpr std::size_t book_n = NumPoints - 1;
  for (auto &el : lhs) {
    result[index].first = el.first;
    result[index].second = el.second;
    ++index;
  }
  constexpr std::size_t book_n = NumPoints - 1;

  std::size_t index2 = index - 1;
  if ((NumPoints - 1) % 2u == 0) {
    result[(NumPoints - 1) / 2].first = 0.0;
    auto m = q_and_evaluation<NumPoints - 1>(result[(NumPoints - 1) / 2].first);
    result[(NumPoints - 1) / 2].second =
        2.0 / ((double)book_n * ((double)book_n + 1.0) *
               gcem::pow(std::get<2>(m), 2.0));
    index++;
  }
  while (index < NumPoints) {
    result[index].first = -result[index2].first;
    result[index].second = result[index2].second;
    index++;
    index2--;
  }
  /* for (auto &el : rhs) { */
  /*   result[index].first = el.first; */
  /*   result[index].second = el.second; */
  /*   ++index; */
  /* } */

  /* for (std::size_t i = a + 1; i < NumPoints; i++) { */
  /*   result[i].first = -result[NumPoints - i - 1].first; */
  /*   result[i].second = result[NumPoints - i - 1].second; */
  /* } */
  return result;
}
template <std::size_t NumPoints>
constexpr std::array<double, NumPoints>
get_first(std::array<std::pair<double, double>,
                     static_cast<int>(((double)(NumPoints)) / 2.0)>
              lhs) {
  std::array<double, NumPoints> result{};
  std::size_t index = 0;

  // constexpr std::size_t book_n = NumPoints - 1;
  for (auto &el : lhs) {
    result[index] = el.first;
    ++index;
  }
  constexpr std::size_t book_n = NumPoints - 1;

  std::size_t index2 = index - 1;
  if ((NumPoints - 1) % 2u == 0) {
    result[(NumPoints - 1) / 2] = 0.0;

    index++;
  }
  while (index < NumPoints) {
    result[index] = -result[index2];
    index++;
    index2--;
  }
  return result;
}
template <std::size_t NumPoints>
constexpr std::array<double, NumPoints>
get_second(std::array<std::pair<double, double>,
                      static_cast<int>(((double)(NumPoints)) / 2.0)>
               lhs) {
  std::array<double, NumPoints> result{};
  std::size_t index = 0;

  // constexpr std::size_t book_n = NumPoints - 1;
  for (auto &el : lhs) {
    result[index] = el.second;
    ++index;
  }
  constexpr std::size_t book_n = NumPoints - 1;

  std::size_t index2 = index - 1;
  if ((NumPoints - 1) % 2u == 0) {
    auto m = q_and_evaluation<NumPoints - 1>(lhs[(NumPoints - 1) / 2].first);
    result[(NumPoints - 1) / 2] =
        2.0 / ((double)book_n * ((double)book_n + 1.0) *
               gcem::pow(std::get<2>(m), 2.0));
    index++;
  }
  while (index < NumPoints) {
    result[index] = result[index2];
    index++;
    index2--;
  }
  return result;
}

template <std::size_t NumPoints>
constexpr std::array<std::pair<double, double>, NumPoints> compute_glp_aux() {

  constexpr std::size_t a = static_cast<int>(((double)(NumPoints)) / 2.0) - 1;

  constexpr auto half = make_array<a + 1>(compute_single_glp<NumPoints - 1>);

  constexpr auto result =
      concat_pairs<std::pair<double, double>, NumPoints>(half);

  return result;
}

template <std::size_t NumPoints>
constexpr std::array<double, NumPoints> compute_glp() {

  constexpr std::size_t a = static_cast<int>(((double)(NumPoints)) / 2.0) - 1;

  constexpr auto half = make_array<a + 1>(compute_single_glp<NumPoints - 1>);

  constexpr auto result = get_first<NumPoints>(half);

  return result;
}
template <std::size_t NumPoints>
constexpr std::array<double, NumPoints> compute_glw() {

  constexpr std::size_t a = static_cast<int>(((double)(NumPoints)) / 2.0) - 1;

  constexpr auto half = make_array<a + 1>(compute_single_glp<NumPoints - 1>);

  constexpr auto result = get_second<NumPoints>(half);

  return result;
}
template <>
constexpr std::array<std::pair<double, double>, 2> compute_glp_aux() {
  return {std::make_pair(-1.0, 1.0), std::make_pair(1.0, 1.0)};
}
template <> constexpr std::array<double, 2> compute_glp() {
  return {-1.0, 1.0};
}
template <> constexpr std::array<double, 2> compute_glw() { return {1.0, 1.0}; }
} // namespace manifolds::collocation::detail
