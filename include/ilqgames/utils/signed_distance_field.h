#pragma once
#include <ilqgames/utils/types.h>

namespace ilqgames {

class SignedDistanceField {
 public:
  SignedDistanceField(const Eigen::Vector3d& origin,
                      const Eigen::Vector3d& resolution,
                      const Eigen::Vector3i& dim,
                      std::vector<double> data)
      : origin_(origin), resolution_(resolution), dim_(dim), data_(std::move(data)) {
    CHECK_EQ(data_.size(), dim_.x() * dim_.y() * dim_.z()) 
        << "SDF Data size does not match dimensions!";
  }

  static SignedDistanceField CreateFloor(const Eigen::Vector3d& origin,
                                         const Eigen::Vector3d& resolution,
                                         const Eigen::Vector3i& dim,
                                         double z_height) {
    return GenerateGrid(origin, resolution, dim, [&](const Eigen::Vector3d& p) {
      return p.z() - z_height; 
    });
  }

  static SignedDistanceField CreateBox(const Eigen::Vector3d& origin,
                                       const Eigen::Vector3d& resolution,
                                       const Eigen::Vector3i& dim,
                                       const Eigen::Vector3d& center,
                                       const Eigen::Vector3d& size) {
    Eigen::Vector3d half_size = size / 2.0;
    return GenerateGrid(origin, resolution, dim, [&](const Eigen::Vector3d& p) {
      // Precise Box SDF Math
      Eigen::Vector3d d = (p - center).cwiseAbs() - half_size;
      double outside_dist = d.cwiseMax(0.0).norm();
      double inside_dist = std::min(std::max(d.x(), std::max(d.y(), d.z())), 0.0);
      return outside_dist + inside_dist;
    });
  }

  static SignedDistanceField CreateVerticalWall(const Eigen::Vector3d& origin,
                                                const Eigen::Vector3d& resolution,
                                                const Eigen::Vector3i& dim,
                                                double x_pos, 
                                                double thickness) {
    double half_thick = thickness / 2.0;
    return GenerateGrid(origin, resolution, dim, [&](const Eigen::Vector3d& p) {
      return std::abs(p.x() - x_pos) - half_thick;
    });
  }

  inline double GetDistance(const Eigen::Vector3d& position) const;
  inline Eigen::Vector3d GetGradient(const Eigen::Vector3d& position) const;
  
  bool IsInBounds(const Eigen::Vector3d& position) const {
    Eigen::Vector3d diff = position - origin_;
    return (diff.x() >= 0 && diff.x() < dim_.x() * resolution_.x() &&
            diff.y() >= 0 && diff.y() < dim_.y() * resolution_.y() &&
            diff.z() >= 0 && diff.z() < dim_.z() * resolution_.z());
  }

  const Eigen::Vector3d& GetOrigin() const { return origin_; }
  const Eigen::Vector3d& GetResolution() const { return resolution_; }
  const Eigen::Vector3i& GetDim() const { return dim_; }

 private:
  static SignedDistanceField GenerateGrid(const Eigen::Vector3d& origin,
                                          const Eigen::Vector3d& resolution,
                                          const Eigen::Vector3i& dim,
                                          std::function<double(const Eigen::Vector3d&)> func) {
    std::vector<double> data(dim.x() * dim.y() * dim.z());
    int idx = 0;
    // Iterate z, y, x
    for (int z = 0; z < dim.z(); ++z) {
      for (int y = 0; y < dim.y(); ++y) {
        for (int x = 0; x < dim.x(); ++x) {
          double px = origin.x() + x * resolution.x();
          double py = origin.y() + y * resolution.y();
          double pz = origin.z() + z * resolution.z();
          data[idx++] = func(Eigen::Vector3d(px, py, pz));
        }
      }
    }
    return SignedDistanceField(origin, resolution, dim, data);
  }

  inline double GetRaw(int x, int y, int z) const;

  Eigen::Vector3d origin_;
  Eigen::Vector3d resolution_;
  Eigen::Vector3i dim_;
  std::vector<double> data_;
};

inline double SignedDistanceField::GetRaw(int x, int y, int z) const {
  x = std::max(0, std::min(x, dim_.x() - 1));
  y = std::max(0, std::min(y, dim_.y() - 1));
  z = std::max(0, std::min(z, dim_.z() - 1));
  return data_[x + y * dim_.x() + z * dim_.x() * dim_.y()];
}

inline double SignedDistanceField::GetDistance(const Eigen::Vector3d& position) const {
  Eigen::Vector3d p_grid = (position - origin_).cwiseQuotient(resolution_);

  int x0 = static_cast<int>(std::floor(p_grid.x()));
  int y0 = static_cast<int>(std::floor(p_grid.y()));
  int z0 = static_cast<int>(std::floor(p_grid.z()));

  double xd = p_grid.x() - x0;
  double yd = p_grid.y() - y0;
  double zd = p_grid.z() - z0;

  double c00 = GetRaw(x0, y0, z0) * (1.0 - xd) + GetRaw(x0 + 1, y0, z0) * xd;
  double c10 = GetRaw(x0, y0 + 1, z0) * (1.0 - xd) + GetRaw(x0 + 1, y0 + 1, z0) * xd;
  double c01 = GetRaw(x0, y0, z0 + 1) * (1.0 - xd) + GetRaw(x0 + 1, y0, z0 + 1) * xd;
  double c11 = GetRaw(x0, y0 + 1, z0 + 1) * (1.0 - xd) + GetRaw(x0 + 1, y0 + 1, z0 + 1) * xd;

  double c0 = c00 * (1.0 - yd) + c10 * yd;
  double c1 = c01 * (1.0 - yd) + c11 * yd;

  return c0 * (1.0 - zd) + c1 * zd;
}

inline Eigen::Vector3d SignedDistanceField::GetGradient(const Eigen::Vector3d& position) const {
  const double h = 1e-3; 
  const double inv_2h = 1.0 / (2.0 * h);
  Eigen::Vector3d dx(h, 0, 0), dy(0, h, 0), dz(0, 0, h);
  
  return Eigen::Vector3d(
      (GetDistance(position + dx) - GetDistance(position - dx)) * inv_2h,
      (GetDistance(position + dy) - GetDistance(position - dy)) * inv_2h,
      (GetDistance(position + dz) - GetDistance(position - dz)) * inv_2h
  );
}

}  // namespace ilqgames