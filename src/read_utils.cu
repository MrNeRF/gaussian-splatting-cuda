#include "camera.cuh"
#include "image.cuh"
#include "point3d.cuh"
#include "read_utils.cuh"
#include "utils.cuh"
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
void read_ply_file(std::filesystem::path filepath) {
    auto ply_stream_buffer = read_binary(filepath);

    tinyply::PlyFile ply_file;
    ply_file.parse_header(*ply_stream_buffer);

    std::cout << "\t[ply_header] Type: " << (ply_file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto& c : ply_file.get_comments())
        std::cout << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto& c : ply_file.get_info())
        std::cout << "\t[ply_header] Info: " << c << std::endl;

    for (const auto& e : ply_file.get_elements()) {
        std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
        for (const auto& p : e.properties) {
            std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
            if (p.isList)
                std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
            std::cout << std::endl;
        }
    }
}

template <typename T>
T read_binary_value(std::istream& file) {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

// TODO: Do something with the images vector
// adapted from https://github.com/colmap/colmap/blob/dev/src/colmap/base/reconstruction.cc
std::unordered_map<uint32_t, Image> read_images_binary(std::filesystem::path file_path) {
    auto image_stream_buffer = read_binary(file_path);
    const size_t image_count = read_binary_value<uint64_t>(*image_stream_buffer);

    std::unordered_map<uint32_t, Image> images;
    images.reserve(image_count);

    for (size_t i = 0; i < image_count; ++i) {
        const auto image_ID = read_binary_value<uint32_t>(*image_stream_buffer);
        auto img = Image(image_ID);
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

        images.emplace(image_ID, img);
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
        cam._width = read_binary_value<uint64_t>(*camera_stream_buffer);
        cam._height = read_binary_value<uint64_t>(*camera_stream_buffer);

        camera_stream_buffer->read(reinterpret_cast<char*>(cam._params.data()), cam._params.size() * sizeof(double));
        cameras.emplace(camera_ID, cam);
    }

    return cameras;
}

// adapted from https://github.com/colmap/colmap/blob/dev/src/colmap/base/reconstruction.cc
// TODO: There should be points3D data returned
std::vector<Point3D> read_point3D_binary(std::filesystem::path file_path) {
    auto point3D_stream_buffer = read_binary(file_path);
    const size_t point3D_count = read_binary_value<uint64_t>(*point3D_stream_buffer);

    std::vector<Point3D> points3D;
    points3D.reserve(point3D_count);
    for (size_t i = 0; i < point3D_count; ++i) {
        const auto point3D_ID = read_binary_value<uint64_t>(*point3D_stream_buffer);
        auto& point = points3D.emplace_back(point3D_ID);
        point._xyz.x() = read_binary_value<double>(*point3D_stream_buffer);
        point._xyz.y() = read_binary_value<double>(*point3D_stream_buffer);
        point._xyz.z() = read_binary_value<double>(*point3D_stream_buffer);

        point._rgb[0] = read_binary_value<uint8_t>(*point3D_stream_buffer);
        point._rgb[1] = read_binary_value<uint8_t>(*point3D_stream_buffer);
        point._rgb[2] = read_binary_value<uint8_t>(*point3D_stream_buffer);

        point._error = read_binary_value<double>(*point3D_stream_buffer);

        const size_t track_length = read_binary_value<uint64_t>(*point3D_stream_buffer);
        point._tracks.resize(track_length);
        point3D_stream_buffer->read(reinterpret_cast<char*>(point._tracks.data()), track_length * sizeof(Track));
    }

    return points3D;
}

std::vector<CameraInfo> read_colmap_cameras(std::filesystem::path file_path, std::unordered_map<uint32_t, Camera>& cameras, std::unordered_map<uint32_t, Image>& images) {

    std::vector<CameraInfo> camera_infos;
    camera_infos.reserve(images.size());
    for (const auto& [image_ID, image] : images) {
        auto it = cameras.find(image._camera_id);
        auto& camera = it->second; // This should never fail

        auto& camInfo = camera_infos.emplace_back();
        const uint64_t channels = 3;
        unsigned char* img = read_image(file_path / image._name, camera._width, camera._height, channels);
        camInfo.SetImage(img, camera._width, camera._height, channels);
        camInfo._camera_ID = image._camera_id;
        camInfo._R = qvec2rotmat(image._qvec).transpose();
        camInfo._T = image._tvec;

        if (camera._camera_model == CAMERA_MODEL::SIMPLE_PINHOLE) {
            double focal_length_x = camera._params[0];
            camInfo._fov_x = focal2fov(focal_length_x, camInfo.GetImageWidth());
            camInfo._fov_y = focal2fov(focal_length_x, camInfo.GetImageHeight());
        } else if (camera._camera_model == CAMERA_MODEL::PINHOLE) {
            double focal_length_x = camera._params[0];
            double focal_length_y = camera._params[1];
            camInfo._fov_x = focal2fov(focal_length_x, camInfo.GetImageWidth());
            camInfo._fov_y = focal2fov(focal_length_y, camInfo.GetImageHeight());
        } else {
            throw std::runtime_error("Camera model not supported");
        }
    }

    return camera_infos;
}

// TODO: There should be data returned
void read_colmap_scene_info(std::filesystem::path file_path) {
    auto cameras = read_cameras_binary(file_path / "sparse/0/cameras.bin");
    auto images = read_images_binary(file_path / "sparse/0/images.bin");
    auto points3D = read_point3D_binary(file_path / "sparse/0/points3D.bin");
    read_colmap_cameras(file_path / "images", cameras, images);
}
