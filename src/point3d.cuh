#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>

struct Track {
    uint32_t _image_ID;
    uint32_t _max_num_2D_points;
};

class Point3D {
public:
    Point3D(uint64_t id) : _point_ID(id){};
    uint64_t GetPoint3D_ID() const { return _point_ID; }

public:
    Eigen::Vector3d _xyz;
    std::array<uint8_t, 3> _rgb;
    double _error;
    std::vector<Track> _tracks;

private:
    uint64_t _point_ID;
};