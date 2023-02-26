#include<Eigen/Core>
class SphereAtlas 
{
public:
	using Representation = Eigen::Vector3d;
	
	using Tangent = Eigen::Matrix<double, 3, 1>;
	
	static void chart(const Eigen::Vector3d &point1, const Eigen::Vector3d &point2, const Eigen::Vector3d &element, Eigen::Matrix<double, 2, 1> &result);
	
	static void chart_diff(const Eigen::Vector3d &point1, const Eigen::Vector3d &point2, const Eigen::Vector3d &element, Eigen::Matrix<double, 2, 3> &result);
	
	static void param(const Eigen::Vector3d &point1, const Eigen::Vector3d &point2, const Eigen::Matrix<double, 2, 1> &coordinates, Eigen::Vector3d &result);
	
	static void param_diff(const Eigen::Vector3d &point1, const Eigen::Vector3d &point2, const Eigen::Matrix<double, 2, 1> &coordinates, Eigen::Matrix<double, 3, 2> &result);
	
	static Eigen::Vector3d random_projection();
	
	static void change_of_coordinates(const Eigen::Vector3d &point1A, const Eigen::Vector3d &point2A, const Eigen::Vector3d &point1B, const Eigen::Vector3d &point2B, const Eigen::Matrix<double, 2, 1> &coordinates, Eigen::Matrix<double, 2, 1> &result);
	
	static void change_of_coordinates_diff(const Eigen::Vector3d &point1A, const Eigen::Vector3d &point2A, const Eigen::Vector3d &point1B, const Eigen::Vector3d &point2B, const Eigen::Matrix<double, 2, 1> &coordinates, Eigen::Matrix<double, 2, 2> &result);
	
	static const std::size_t dimension;
	
	static const std::size_t tanget_repr_dimension;
	
	
private:
};
