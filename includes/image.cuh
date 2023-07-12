#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Geometry/Quaternion.h>
#include <string>
#include <vector>

// This is basically a hack to speed up the reading of the points2D togther with its ID
struct ImagePoint {
    double _x;
    double _y;
    uint64_t _point_id;
};
class Image {
public:
    Image(uint32_t image_ID) : _image_ID(image_ID) {}
    uint32_t GetImageID() const { return _image_ID; }

public:
    uint32_t _camera_id;
    std::string _name;
    Eigen::Quaterniond _qvec;
    Eigen::Vector3d _tvec;
    std::vector<ImagePoint> _points2D_ID;

private:
    uint32_t _image_ID;
};
