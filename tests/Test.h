#pragma once
#include <Manifolds/Manifold.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <gtest/gtest.h>
namespace manifolds {

template <typename T> struct TestManifold {

  TestManifold() {
    T empty_constructed;

    T move_constructed(T::random_projection());

    // assignemtn
    empty_constructed = move_constructed;

    // comparisoon

    EXPECT_TRUE(empty_constructed == move_constructed);

    manifold_base(move_constructed);
  }

  /// Here we test that we can manipulate the value of the
  /// representation from the ManifoldBase clase
  void manifold_base(const ManifoldBase &_base) {

    // Tess that we can cat the ManifoldBase to T
    EXPECT_TRUE(_base.is_same<T>());
    EXPECT_TRUE(dynamic_cast<const T *>(&_base) != nullptr);

    const T &original = *dynamic_cast<const T *>(&_base);

    std::unique_ptr<ManifoldBase> cloned_base = _base.clone();

    // Test getters, dimension and tangent representation dimension
    EXPECT_TRUE(cloned_base->get_dim() == T::dimension);
    EXPECT_TRUE(cloned_base->get_tanget_repr_dim() ==
                T::tangent_repr_dimension);

    // Test that we can access to the represent from the base pointer
    T &ref = *dynamic_cast<T *>(cloned_base.get());

    EXPECT_TRUE(original == ref);

    std::unique_ptr<ManifoldBase> moved_base = cloned_base->move_clone();

    EXPECT_FALSE(cloned_base->has_value());

    T &ref_to_moved_base = *dynamic_cast<T *>(moved_base.get());

    // Test that the move operator also the reference
    EXPECT_TRUE(original == ref_to_moved_base);

    T other(T::random_projection());

    auto other_base = static_cast<ManifoldBase &>(other).clone();

    moved_base->assign(other_base);

    EXPECT_TRUE(other == ref_to_moved_base);

    EXPECT_TRUE(moved_base->is_equal(other_base));
  }
};

template <typename T> struct TestManifoldFaithful : public TestManifold<T> {

  TestManifoldFaithful() : TestManifold<T>() {

    T manifold = T::random_projection();

    typename T::Representation representation;

    representation = manifold;

    EXPECT_TRUE(T::atlas::comparison(representation, manifold.crepr()));

    representation = T::atlas::random_projection();

    manifold = representation;
    EXPECT_TRUE(T::atlas::comparison(representation, manifold.crepr()));

    // copy constructor from representation

    T copy_constructed(representation);

    // Test cast to const reference
    auto fun_takes_const_reference =
        [&representation](const typename T::Representation &v) {
          EXPECT_TRUE(T::atlas::comparison(representation, v));
        };

    fun_takes_const_reference(copy_constructed);

    // Test cast to reference
    auto fun_takes_reference =
        [&representation](typename T::Representation &v) {
          v = representation;
        };
    T empty_constructed;
    fun_takes_reference(empty_constructed);
    EXPECT_TRUE(empty_constructed == T::CRef(representation));
  }
};

template <typename T> struct TestMap {
  using domain_t = typename T::domain_t;
  using codomain_t = typename T::codomain_t;
  TestMap() {
    domain_t domain_value = domain_t::random_projection();
    codomain_t codomain_value_a = codomain_t::random_projection();
    codomain_t codomain_value_b = codomain_t::random_projection();

    T current_map;

    current_map(domain_value, codomain_value_a);
    codomain_value_b = current_map(domain_value);

    EXPECT_TRUE(codomain_value_a == codomain_value_b);

    EXPECT_TRUE(codomain_value_a == codomain_value_b);

    map_base(current_map);
  }
  /// Here we test that we can oapply the function with
  /// the MapBase clase
  void map_base(const MapBase &_base) {

    EXPECT_TRUE(dynamic_cast<const T *>(&_base) != nullptr);

    const T &original = *dynamic_cast<const T *>(&_base);

    std::unique_ptr<MapBase> cloned_base = _base.clone();

    // Test getters, dimension and tangent representation dimension
    EXPECT_TRUE(cloned_base->get_codom_dim() == T::codomain_t::dimension);
    EXPECT_TRUE(cloned_base->get_codom_tangent_repr_dim() ==
                T::codomain_t::tangent_repr_dimension);
    EXPECT_TRUE(cloned_base->get_dom_dim() == T::domain_t::dimension);
    EXPECT_TRUE(cloned_base->get_dom_tangent_repr_dim() ==
                T::domain_t::tangent_repr_dimension);

    typename T::domain_t domain_value = T::domain_t::random_projection();
    typename T::codomain_t codomain_value_a =
        T::codomain_t::random_projection();

    original(domain_value, codomain_value_a);
    auto codomain_value_b = cloned_base->value(domain_value.clone());
  }
};

} // namespace manifolds
