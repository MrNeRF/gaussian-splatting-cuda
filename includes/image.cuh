#pragma once

#pragma diag_suppress code_of_warning
#include <Eigen/Dense>
#include <Eigen/src/Geometry/Quaternion.h>
#pragma diag_default code_of_warning
#include <string>
#include <vector>

// This is basically a hack to speed up the reading of the points2D togther with its ID
// TODO: Remove this class finally
class Image {
public:
    Image() = default;
    Image(uint32_t image_ID) : _image_ID(image_ID) {}
    uint32_t GetImageID() const { return _image_ID; }

public:
    uint32_t _camera_id;
    std::string _name;
    Eigen::Quaternionf _qvec;
    Eigen::Vector3f _tvec;

private:
    uint32_t _image_ID;
};
