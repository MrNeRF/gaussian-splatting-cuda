#include "camera.cuh"
#include "image.cuh"
#include "point_cloud.cuh"
#include "read_utils.cuh"
#include "utils.cuh"
#include <algorithm>
#include <exception>
#include <execution>
#include <fstream>
#include <iostream>
#include <memory>
#include <tinyply.h>
#include <unordered_map>
#include <vector>

// Reads and preloads a binary file into a string stream
// file_path: path to the file
// returns: a unique pointer to a string stream
std::unique_ptr<std::istream> read_binary(std::filesystem::path file_path) {
    std::ifstream file(file_path, std::ios::binary);
    std::unique_ptr<std::istream> file_stream;
    if (file.fail()) {
        throw std::runtime_error("Failed to open file: " + file_path.string());
    }
    // preload
    std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(file), {});
    file_stream = std::make_unique<std::stringstream>(std::string(buffer.begin(), buffer.end()));
    return file_stream;
}

// Returns the file size of a given ifstream in MB
float file_in_mb(std::istream* file_stream) {
    file_stream->seekg(0, std::ios::end);
    const float size_mb = file_stream->tellg() * 1e-6f;
    file_stream->seekg(0, std::ios::beg);
    return size_mb;
}

// Reads ply file and prints header
PointCloud read_ply_file(std::filesystem::path file_path) {
    auto ply_stream_buffer = read_binary(file_path);

    tinyply::PlyFile ply_file;
    ply_file.parse_header(*ply_stream_buffer);

    PointCloud point_cloud;

    try {
        std::shared_ptr<tinyply::PlyData> vertices = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
        const size_t byte_count = vertices->buffer.size_bytes();
        point_cloud._points.resize(vertices->count);
        std::memcpy(point_cloud._points.data(), vertices->buffer.get(), byte_count);
    } catch (const std::exception& e) {
        std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
        std::shared_ptr<tinyply::PlyData> normals = ply_file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
        const size_t byte_count = normals->buffer.size_bytes();
        point_cloud._normals.resize(normals->count);
        std::memcpy(point_cloud._normals.data(), normals->buffer.get(), byte_count);
    } catch (const std::exception& e) {
        std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
        std::shared_ptr<tinyply::PlyData> colors = ply_file.request_properties_from_element("vertex", {"red", "green", "blue"});
        const size_t byte_count = colors->buffer.size_bytes();
        point_cloud._colors.resize(colors->count);
        std::memcpy(point_cloud._colors.data(), colors->buffer.get(), byte_count);
    } catch (const std::exception& e) {
        std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    return point_cloud;
}

void write_ply_file(std::filesystem::path file_path, const PointCloud& point_cloud) {

    std::filebuf fb_binary;
    fb_binary.open(file_path.c_str(), std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) {
        throw std::runtime_error("failed to open " + file_path.string());
    } else if (point_cloud._points.empty()) {
        throw std::runtime_error("point cloud is empty");
    }

    tinyply::PlyFile binary_point3D_file;

    if (!point_cloud._points.empty()) {
        binary_point3D_file.add_properties_to_element("vertex", {"x", "y", "z"},
                                                      tinyply::Type::FLOAT32, point_cloud._points.size(), const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(point_cloud._points.data())), tinyply::Type::INVALID, 0);
    }

    if (!point_cloud._normals.empty()) {
        binary_point3D_file.add_properties_to_element("vertex", {"nx", "ny", "nz"},
                                                      tinyply::Type::FLOAT32, point_cloud._normals.size(), const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(point_cloud._normals.data())), tinyply::Type::INVALID, 0);
    }

    if (!point_cloud._colors.empty()) {

        binary_point3D_file.add_properties_to_element("vertex", {"red", "green", "blue"},
                                                      tinyply::Type::UINT8, point_cloud._colors.size(), const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(point_cloud._colors.data())), tinyply::Type::INVALID, 0);
    }
    binary_point3D_file.write(outstream_binary, true);
}

template <typename T>
T read_binary_value(std::istream& file) {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

// TODO: Do something with the images vector
// adapted from https://github.com/colmap/colmap/blob/dev/src/colmap/base/reconstruction.cc
std::vector<Image> read_images_binary(std::filesystem::path file_path) {
    auto image_stream_buffer = read_binary(file_path);
    const size_t image_count = read_binary_value<uint64_t>(*image_stream_buffer);

    std::vector<Image> images;
    images.reserve(image_count);

    for (size_t i = 0; i < image_count; ++i) {
        const auto image_ID = read_binary_value<uint32_t>(*image_stream_buffer);
        auto& img = images.emplace_back(image_ID);
        img._qvec.x() = read_binary_value<double>(*image_stream_buffer);
        img._qvec.y() = read_binary_value<double>(*image_stream_buffer);
        img._qvec.z() = read_binary_value<double>(*image_stream_buffer);
        img._qvec.w() = read_binary_value<double>(*image_stream_buffer);
        img._qvec.normalize();

        img._tvec.x() = read_binary_value<double>(*image_stream_buffer);
        img._tvec.y() = read_binary_value<double>(*image_stream_buffer);
        img._tvec.z() = read_binary_value<double>(*image_stream_buffer);

        img._camera_id = read_binary_value<uint32_t>(*image_stream_buffer);

        char character;
        do {
            image_stream_buffer->read(&character, 1);
            if (character != '\0') {
                img._name += character;
            }
        } while (character != '\0');

        const size_t number_points = read_binary_value<uint64_t>(*image_stream_buffer);
        // Calculate total size needed for point data

        // Read all the point data at once
        img._points2D_ID.resize(number_points);
        image_stream_buffer->read(reinterpret_cast<char*>(img._points2D_ID.data()), number_points * sizeof(ImagePoint));
    }

    return images;
}

// TODO: Do something with the cameras vector
// adapted from https://github.com/colmap/colmap/blob/dev/src/colmap/base/reconstruction.cc
std::unordered_map<uint32_t, Camera> read_cameras_binary(std::filesystem::path file_path) {
    auto camera_stream_buffer = read_binary(file_path);
    const size_t camera_count = read_binary_value<uint64_t>(*camera_stream_buffer);

    std::unordered_map<uint32_t, Camera> cameras;
    cameras.reserve(camera_count);

    for (size_t i = 0; i < camera_count; ++i) {
        auto camera_ID = read_binary_value<uint32_t>(*camera_stream_buffer);
        auto model_id = read_binary_value<int>(*camera_stream_buffer);
        auto cam = Camera(model_id);
        cam._camera_ID = camera_ID;
        cam._width = read_binary_value<uint64_t>(*camera_stream_buffer);
        cam._height = read_binary_value<uint64_t>(*camera_stream_buffer);

        camera_stream_buffer->read(reinterpret_cast<char*>(cam._params.data()), cam._params.size() * sizeof(double));
        cameras.emplace(camera_ID, cam);
    }

    return cameras;
}

// adapted from https://github.com/colmap/colmap/blob/dev/src/colmap/base/reconstruction.cc
// TODO: There should be points3D data returned
void read_point3D_binary(std::filesystem::path file_path) {
    auto point3D_stream_buffer = read_binary(file_path);
    const size_t point3D_count = read_binary_value<uint64_t>(*point3D_stream_buffer);

    struct Track {
        uint32_t _image_ID;
        uint32_t _max_num_2D_points;
    };

    PointCloud point_cloud;
    point_cloud._points = std::vector<Point>(point3D_count);
    point_cloud._colors = std::vector<Color>(point3D_count);
    //  point_cloud._normals.reserve(point3D_count); <- no normals saved. Just ignore.
    for (size_t i = 0; i < point3D_count; ++i) {
        // just ignore the point3D_ID
        read_binary_value<uint64_t>(*point3D_stream_buffer);
        // vertices
        point_cloud._points[i].x = read_binary_value<double>(*point3D_stream_buffer);
        point_cloud._points[i].y = read_binary_value<double>(*point3D_stream_buffer);
        point_cloud._points[i].z = read_binary_value<double>(*point3D_stream_buffer);

        // colors
        point_cloud._colors[i].r = read_binary_value<uint8_t>(*point3D_stream_buffer);
        point_cloud._colors[i].g = read_binary_value<uint8_t>(*point3D_stream_buffer);
        point_cloud._colors[i].b = read_binary_value<uint8_t>(*point3D_stream_buffer);

        // the rest can be ignored.
        read_binary_value<double>(*point3D_stream_buffer); // ignore

        const size_t track_length = read_binary_value<uint64_t>(*point3D_stream_buffer);
        std::vector<Track> tracks;
        tracks.resize(track_length);
        point3D_stream_buffer->read(reinterpret_cast<char*>(tracks.data()), track_length * sizeof(Track));
    }

    write_ply_file(file_path.parent_path() / "testply.ply", point_cloud);
}

std::vector<CameraInfo> read_colmap_cameras(const std::filesystem::path file_path, const std::unordered_map<uint32_t, Camera>& cameras, const std::vector<Image>& images) {

    std::vector<CameraInfo> camera_infos(images.size());

    // Create a vector with all the keys from the images map
    std::vector<uint32_t> keys(images.size());
    std::generate(keys.begin(), keys.end(), [n = 0]() mutable { return n++; });

    std::for_each(std::execution::par, keys.begin(), keys.end(), [&](uint32_t image_ID) {
        // Make a copy of the image object to avoid accessing the shared resource
        Image image = images[image_ID];
        auto it = cameras.find(image._camera_id);
        auto& camera = it->second; // This should never fail
        const uint64_t channels = 3;
        unsigned char* img = read_image(file_path / image._name, camera._width, camera._height, channels);

        camera_infos[image_ID].SetImage(img, camera._width, camera._height, channels);
        camera_infos[image_ID]._camera_ID = image._camera_id;
        camera_infos[image_ID]._R = qvec2rotmat(image._qvec).transpose();
        camera_infos[image_ID]._T = image._tvec;

        if (camera._camera_model == CAMERA_MODEL::SIMPLE_PINHOLE) {
            double focal_length_x = camera._params[0];
            camera_infos[image_ID]._fov_x = focal2fov(focal_length_x, camera_infos[image_ID].GetImageWidth());
            camera_infos[image_ID]._fov_y = focal2fov(focal_length_x, camera_infos[image_ID].GetImageHeight());
        } else if (camera._camera_model == CAMERA_MODEL::PINHOLE) {
            double focal_length_x = camera._params[0];
            double focal_length_y = camera._params[1];
            camera_infos[image_ID]._fov_x = focal2fov(focal_length_x, camera_infos[image_ID].GetImageWidth());
            camera_infos[image_ID]._fov_y = focal2fov(focal_length_y, camera_infos[image_ID].GetImageHeight());
        } else {
            throw std::runtime_error("Camera model not supported");
        }
    });

    return camera_infos;
}

std::pair<Eigen::Vector3d, double> get_center_and_diag(std::vector<Eigen::Vector3d>& cam_centers) {
    Eigen::Vector3d avg_cam_center = Eigen::Vector3d::Zero();
    for (const auto& center : cam_centers) {
        avg_cam_center += center;
    }
    avg_cam_center /= static_cast<double>(cam_centers.size());

    double max_dist = 0;
    for (const auto& center : cam_centers) {
        max_dist = std::max(max_dist, (center - avg_cam_center).norm());
    }

    return {avg_cam_center, max_dist};
}

std::pair<Eigen::Vector3d, double> getNerfppNorm(std::vector<CameraInfo>& cam_info) {
    std::vector<Eigen::Vector3d> cam_centers;

    for (CameraInfo& cam : cam_info) {
        Eigen::Matrix4d W2C = getWorld2View2(cam._R, cam._T);
        Eigen::Matrix4d C2W = W2C.inverse();
        cam_centers.push_back(C2W.block<3, 1>(0, 3));
    }

    auto [center, diagonal] = get_center_and_diag(cam_centers);

    double radius = diagonal * 1.1;
    Eigen::Vector3d translate = -center;

    return {translate, radius};
}

// TODO: There should be data returned
void read_colmap_scene_info(std::filesystem::path file_path) {
    auto cameras = read_cameras_binary(file_path / "sparse/0/cameras.bin");
    auto images = read_images_binary(file_path / "sparse/0/images.bin");

    // if (filesystem::exists(file_path / "sparse/0/points3D.bin")) {
    read_point3D_binary(file_path / "sparse/0/points3D.bin");
    //}
    auto camera_infos = read_colmap_cameras(file_path / "images", cameras, images);
    auto [translate, radius] = getNerfppNorm(camera_infos);
}
