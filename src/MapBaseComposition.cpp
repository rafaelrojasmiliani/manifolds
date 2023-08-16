#include <Manifolds/Maps/MapBaseComposition.hpp>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
namespace manifolds {

detail::mixed_matrix_t get_diff_comp_buffer(const MapBase &_first,
                                            const MapBase &_second) {

  if (_first.differential_type() == detail::MatrixTypeId::Sparse and
      _second.differential_type() == detail::MatrixTypeId::Sparse)
    return detail::sparse_matrix_t(_second.get_codom_tangent_repr_dim(),
                                   _first.get_dom_tangent_repr_dim());

  return Eigen::MatrixXd(_second.get_codom_tangent_repr_dim(),
                         _first.get_dom_tangent_repr_dim());
}

void MapBaseComposition::add_matrix_to_result_buffers() {

  std::size_t last_index = matrix_buffers_.size() - 1;
  std::size_t penultimate_index = last_index - 1;
  std::visit(
      [this, last_index, penultimate_index](auto &&arg, auto &&arg2) {
        using T1 = std::decay_t<decltype(arg)>;
        using T2 = std::decay_t<decltype(arg2)>;
        if constexpr (std::is_same_v<T1, Eigen::SparseMatrix<double>> &&
                      std::is_same_v<T2, Eigen::SparseMatrix<double>>) {
          // this matrix stores the result of diff_penultimat * diff_last

          this->matrix_result_buffers_.emplace_back(detail::sparse_matrix_t(
              maps_[penultimate_index]->get_codom_tangent_repr_dim(),
              maps_[last_index]->get_dom_tangent_repr_dim()));
        } else {
          this->matrix_result_buffers_.emplace_back(Eigen::MatrixXd(
              maps_[penultimate_index]->get_codom_tangent_repr_dim(),
              maps_[last_index]->get_dom_tangent_repr_dim()));
        }
      },
      matrix_buffers_[last_index], matrix_buffers_[penultimate_index]);
}
void MapBaseComposition::fill_matrix_result_buffers() {

  matrix_result_buffers_.clear();
  if (maps_.size() <= 1)
    return;

  add_matrix_to_result_buffers();

  if (maps_.size() <= 2)
    return;
  std::size_t last_index = matrix_buffers_.size() - 1;

  for (int i = matrix_buffers_.size() - 3; i >= 0; i--) {
    std::visit(
        [this, i, last_index](auto &&arg, auto &&arg2) {
          using T1 = std::decay_t<decltype(arg)>;
          using T2 = std::decay_t<decltype(arg2)>;
          if constexpr (std::is_same_v<T1, Eigen::SparseMatrix<double>> &&
                        std::is_same_v<T2, Eigen::SparseMatrix<double>>) {
            matrix_result_buffers_.emplace(
                matrix_result_buffers_.begin(),
                Eigen::SparseMatrix<double>(maps_[i]->get_codom_dim(),
                                            maps_[last_index]->get_dom_dim()));
          } else {
            matrix_result_buffers_.emplace(
                matrix_result_buffers_.begin(),
                Eigen::MatrixXd(maps_[i]->get_codom_dim(),
                                maps_[last_index]->get_dom_dim()));
          }
        },
        matrix_result_buffers_.front(), matrix_buffers_[i]);
  }
  if (matrix_result_buffers_.size() != matrix_buffers_.size() - 1) {
    throw std::logic_error("here");
  }
}
// -------------------------------------------
// -------- Live cycle -----------------------
// -------------------------------------------
MapBaseComposition::MapBaseComposition(const MapBaseComposition &_that)
    : MapBase(), maps_(), codomain_buffers_() {

  for (const auto &map : _that.maps_) {
    append(*map);
  }
}

MapBaseComposition::MapBaseComposition(MapBaseComposition &&_that)
    : MapBase(), maps_(std::move(_that.maps_)),
      codomain_buffers_(std::move(_that.codomain_buffers_)) {}

MapBaseComposition::MapBaseComposition(const MapBase &_in)
    : MapBase(), maps_() {
  append(_in);
}

MapBaseComposition::MapBaseComposition(MapBase &&_other) : MapBase(), maps_() {
  append(std::move(_other));
}

MapBaseComposition::MapBaseComposition(
    const std::vector<std::reference_wrapper<const MapBase>> &_in)
    : MapBase(), maps_(), codomain_buffers_() {
  for (const auto &map : _in) {
    append(map.get());
  }
}

MapBaseComposition &
MapBaseComposition::operator=(const MapBaseComposition &that) {
  maps_.clear();
  codomain_buffers_.clear();
  matrix_buffers_.clear();

  std::transform(that.maps_.begin(), that.maps_.end(),
                 std::back_inserter(maps_),
                 [](const auto &in) { return in->clone(); });

  std::transform(that.codomain_buffers_.begin(), that.codomain_buffers_.end(),
                 std::back_inserter(codomain_buffers_),
                 [](const auto &in) { return in->clone(); });

  // Here, change to Variant of dense and sparse matrix
  std::transform(that.matrix_buffers_.begin(), that.matrix_buffers_.end(),
                 std::back_inserter(matrix_buffers_),
                 [](const auto &in) { return in; });

  fill_matrix_result_buffers();
  return *this;
}

MapBaseComposition &MapBaseComposition::operator=(MapBaseComposition &&that) {

  maps_.clear();
  codomain_buffers_.clear();
  matrix_buffers_.clear();

  maps_ = std::move(that.maps_);
  codomain_buffers_ = std::move(that.codomain_buffers_);
  // Here, change to Variant of dense and sparse matrix
  matrix_buffers_ = std::move(that.matrix_buffers_);

  return *this;
}
// ---------------------------------------------------------
// -------- Modifiers  -------------------------------------
// ---------------------------------------------------------

// --------   Append  --------------------------------------
//    f_1      f_2      f_3     f_4    | Buffer of functions
// ---------------------------------------------------------
//  ib  ob                             | ib: input buffer
//  ib  1b    b1 ob                    | ob: output buffer
//  ib  1b    1b  2b   2b ob           | ib: buff  codom of f_i
//  ib  1b    1b  2b   2b 3b   3b ob   |

void MapBaseComposition::append(const MapBase &_in) {

  if (maps_.size() >= 1) {
    codomain_buffers_.push_back(maps_.back()->codomain_buffer());
    matrix_buffers_.push_back(maps_.back()->linearization_buffer());
  }

  maps_.push_back(_in.clone());

  aux_codomain_buffers_ptr_.push_back(nullptr);
}

void MapBaseComposition::append(MapBase &&_in) {

  if (maps_.size() >= 1) {
    codomain_buffers_.push_back(maps_.back()->codomain_buffer());
    matrix_buffers_.push_back(maps_.back()->linearization_buffer());
  }

  maps_.push_back(_in.move_clone());

  aux_codomain_buffers_ptr_.push_back(nullptr);
}
// -------------------------------------------
// -------- Interface  -----------------------
// -------------------------------------------
// --------   Evaluation  ----------------------------------
//    f_1      f_2      f_3     f_4    | Buffer of functions
// ---------------------------------------------------------
//  ib  ob                             | ib: input buffer
//  ib  1b    b1 ob                    | ob: output buffer
//  ib  1b    1b  2b   2b ob           | ib: buff  codom of f_i
//  ib  1b    1b  2b   2b 3b   3b ob   |

bool MapBaseComposition::value_impl(const ManifoldBase *_in,
                                    ManifoldBase *_out) const {

  if (maps_.size() == 1) {
    return maps_.front()->value_impl(_in, _out);
  }

  maps_.front()->value_impl(_in, codomain_buffers_.front().get());

  std::vector<ManifoldBase *> buffers;

  for (std::size_t i = 1; i < maps_.size() - 1; i++) {
    bool res = maps_[i]->value_impl(codomain_buffers_[i - 1].get(),
                                    codomain_buffers_[i].get());
    if (not res) {
      return res;
    }
  }

  return maps_.back()->value_impl(codomain_buffers_.back().get(), _out);
}
// diff_{n-1} diff_{n-2} diff_{n-3} ...    diff_3 diff_2 diff_1 diff_0
//
// diff_{n-1} diff_{n-2} diff_{n-3} ...    diff_3 diff_2 diff_1 diff_0
//
// diff_{n-1} diff_{n-2} diff_{n-3} ...    diff_3 diff_2 diff_1 diff_0
//
bool MapBaseComposition::diff_impl(const ManifoldBase *_in,
                                   detail::mixed_matrix_ref_t _mat) const {
  auto map_it = maps_.end();
  auto codomain_it = codomain_buffers_.end();
  auto matrix_it = matrix_buffers_.end();
  auto matrix_it_prev = matrix_buffers_.end();
  auto matrix_result_it = matrix_result_buffers_.end();
  map_it = std::prev(map_it, 1);
  codomain_it = std::prev(codomain_it, 1);
  matrix_it = std::prev(matrix_it, 1);
  matrix_it_prev = std::prev(matrix_it_prev, 1);
  matrix_result_it = std::prev(matrix_result_it, 1);

  (*map_it)->value_impl(_in, codomain_it->get());

  std::visit(
      [&](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, Eigen::MatrixXd>)
          (*map_it)->diff_impl(_in, std::get<0>(*matrix_it));
        else if constexpr (std::is_same_v<T, Eigen::SparseMatrix<double>>) {
          (*map_it)->diff_impl(_in, std::get<1>(*matrix_it));
        } else {
          throw std::logic_error("cannot be here");
        }
      },
      *matrix_it);

  if (maps_.size() == 1) {
    std::get<0>(_mat) = std::get<0>(*matrix_it);
    return true;
  }

  while (map_it != maps_.begin()) {
    auto domain_it = codomain_it;
    --map_it;
    --codomain_it;
    --matrix_it;
    (*map_it)->value(*domain_it, *codomain_it);

    // Here we load the buffer for the differential of the function
    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, Eigen::MatrixXd>)
            (*map_it)->diff(**domain_it, std::get<0>(*matrix_it));
          else if constexpr (std::is_same_v<T, Eigen::SparseMatrix<double>>) {
            (*map_it)->diff(**domain_it, std::get<1>(*matrix_it));
          } else {
            throw std::logic_error("cannot be here");
          }
        },
        *matrix_it);

    // Here we accumulate in the accumulation buffer
    if (matrix_result_it == std::prev(matrix_result_buffers_.end(), 1)) {
      std::visit(
          [&](auto &&m1, auto &&m2) {
            using T1 = std::decay_t<decltype(m1)>;
            using T2 = std::decay_t<decltype(m2)>;
            if constexpr (std::is_same_v<T1, Eigen::MatrixXd> ||
                          std::is_same_v<T2, Eigen::MatrixXd>)
              std::get<0>(*matrix_result_it) = m1 * m2;
            else
              std::get<1>(*matrix_result_it) = m1 * m2;
          },
          *matrix_it, *matrix_it_prev);
    } else {
      std::visit(
          [&](auto &&m1, auto &&m2) {
            using T1 = std::decay_t<decltype(m1)>;
            using T2 = std::decay_t<decltype(m2)>;
            if constexpr (std::is_same_v<T1, Eigen::MatrixXd> ||
                          std::is_same_v<T2, Eigen::MatrixXd>)
              std::get<0>(*matrix_result_it) = m1 * m2;
            else
              std::get<1>(*matrix_result_it) = m1 * m2;
          },
          *matrix_it, *(matrix_result_it + 1));
    }
    --matrix_it_prev;
    --matrix_result_it;
  }

  std::get<0>(_mat) = std::get<0>(matrix_result_buffers_.front());
  return true;
}

ManifoldBase *MapBaseComposition::domain_buffer_impl() const {
  return maps_.back()->domain_buffer().release();
}

ManifoldBase *MapBaseComposition::codomain_buffer_impl() const {
  return maps_.front()->codomain_buffer().release();
}

// -------------------------------------------
// -------- Getters    -----------------------
// -------------------------------------------
std::size_t MapBaseComposition::get_dom_dim() const {
  return maps_.back()->get_dom_dim();
}
std::size_t MapBaseComposition::get_codom_dim() const {
  return maps_.front()->get_codom_dim();
}
std::size_t MapBaseComposition::get_dom_tangent_repr_dim() const {
  return maps_.back()->get_dom_dim();
}
std::size_t MapBaseComposition::get_codom_tangent_repr_dim() const {
  return maps_.front()->get_codom_dim();
}

} // namespace manifolds
