#pragma once

#include <vector>

struct Point {
    float x;
    float y;
    float z;
};

struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct PointCloud {
    std::vector<Point> _points;
    std::vector<Color> _colors;
};