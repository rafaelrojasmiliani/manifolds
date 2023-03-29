#pragma once
#include <Manifolds/Maps/MapBase.hpp>
namespace manifolds {

/// Base clase to implement type agnostic composition of maps.
///
/// The observable state of this class is given by an array of unique pointers
/// to type agnostic maps.
/// In addition this class contains an array of buffers to memory allocation
/// when evaluation and differentiation.
class MapBaseComposition : virtual public MapBase {
public:
  // -------------------------------------------
  // -------- Live cycle -----------------------
  // -------------------------------------------
  MapBaseComposition() = delete;

  /// Copy constructor
  MapBaseComposition(const MapBaseComposition &_that);

  /// Move constructor
  MapBaseComposition(MapBaseComposition &&_that);

  /// Cast constructor from a type agnostic map
  MapBaseComposition(const MapBase &_in);

  /// Move-cast constructor from a type agnostic map
  MapBaseComposition(MapBase &&_in);

  /// Constructor from a pointer to a type agnostic map
  explicit MapBaseComposition(const std::unique_ptr<MapBase> &_in);

  /// Move-cast constructor from any type agnostic map
  explicit MapBaseComposition(std::unique_ptr<MapBase> &&_in);

  /// Construct from a vector of unique pointers to type agnostic maps
  MapBaseComposition(const std::vector<std::unique_ptr<MapBase>> &_in);

  /// Move-construct from a vector of unique pointers to type agnostic maps
  MapBaseComposition(std::vector<std::unique_ptr<MapBase>> &&_in);

  /// Default destructor.
  virtual ~MapBaseComposition() = default;

  /// Assigment
  MapBaseComposition &operator=(const MapBaseComposition &that);

  /// Move-assigment
  MapBaseComposition &operator=(MapBaseComposition &&that);

  std::unique_ptr<MapBaseComposition> clone() const {
    return std::unique_ptr<MapBaseComposition>(
        reinterpret_cast<MapBaseComposition *>(clone_impl()));
  }

  std::unique_ptr<MapBaseComposition> move_clone() {
    return std::unique_ptr<MapBaseComposition>(move_clone_impl());
  }

  // -------------------------------------------
  // -------- Modifiers  -----------------------
  // -------------------------------------------
  void append(const MapBase &_in);
  void append(MapBase &&_in);

  // -------------------------------------------
  // -------- getters    -----------------------
  // -------------------------------------------
  std::size_t get_dom_dim() const override;
  std::size_t get_codom_dim() const override;

  std::size_t get_dom_tangent_repr_dim() const override;
  std::size_t get_codom_tangent_repr_dim() const override;

protected:
  std::vector<std::unique_ptr<MapBase>> maps_;
  mutable std::vector<std::unique_ptr<ManifoldBase>> codomain_buffers_;

  /// Stores the matrices result from the different maps in the composition
  /// diff_0 diff_1 ... diff_{n-1}
  mutable std::vector<DifferentialReprType> matrix_buffers_;

  /// Stores the result of the multiplication sequcen of matrices
  /// diff_0 diff_1 diff_2 ... diff_{n-3} diff_{n-2} diff_{n-1}
  //                                      \----------v-------/
  //                                        matrix_result_buffers_[n-2]
  //                           \----------v----------/
  //                            matrix_result_buffers_[n-3]
  mutable std::vector<DifferentialReprType> matrix_result_buffers_;

  bool value_impl(const ManifoldBase *_in, ManifoldBase *_other) const override;

  // Here, change to Variant of dense and sparse matrix
  bool diff_impl(const ManifoldBase *_in,
                 DifferentialReprRefType _mat) const override;

  ManifoldBase *domain_buffer_impl() const override;
  ManifoldBase *codomain_buffer_impl() const override;

  virtual MapBaseComposition *clone_impl() const override {
    return new MapBaseComposition(*this);
  }

  virtual MapBaseComposition *move_clone_impl() override {
    return new MapBaseComposition(std::move(*(this)));
  }

  void fill_matrix_result_buffers();
  void add_matrix_to_result_buffers();
};

} // namespace manifolds
