#include "camera_info.cuh"
#include "camera_utils.cuh"
#include "image.cuh"
#include "point_cloud.cuh"
#include "read_utils.cuh"
#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <tinyply.h>
#include <unordered_map>
#include <vector>

std::unordered_map<int, std::pair<CAMERA_MODEL, uint32_t>> camera_model_ids = {
    {0, {CAMERA_MODEL::SIMPLE_PINHOLE, 3}},
    {1, {CAMERA_MODEL::PINHOLE, 4}},
    {2, {CAMERA_MODEL::SIMPLE_RADIAL, 4}},
    {3, {CAMERA_MODEL::RADIAL, 5}},
    {4, {CAMERA_MODEL::OPENCV, 8}},
    {5, {CAMERA_MODEL::OPENCV_FISHEYE, 8}},
    {6, {CAMERA_MODEL::FULL_OPENCV, 12}},
    {7, {CAMERA_MODEL::FOV, 5}},
    {8, {CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE, 4}},
    {9, {CAMERA_MODEL::RADIAL_FISHEYE, 5}},
    {10, {CAMERA_MODEL::THIN_PRISM_FISHEYE, 12}},
    {11, {CAMERA_MODEL::UNDEFINED, -1}}};

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
    tinyply::PlyFile file;
    std::shared_ptr<tinyply::PlyData> vertices, normals, colors;
    file.parse_header(*ply_stream_buffer);
    // The header information can be used to programmatically extract properties on elements
    // known to exist in the header prior to reading the data. For brevity of this sample, properties
    // like vertex position are hard-coded:
    try {
        vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try {
        normals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
    } catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try {
        colors = file.request_properties_from_element("vertex", {"red", "green", "blue"});
    } catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    file.read(*ply_stream_buffer);

    PointCloud point_cloud;
    if (vertices) {
        std::cout << "\tRead " << vertices->count << " total vertices " << std::endl;
        try {
            point_cloud._points.resize(vertices->count);
            std::memcpy(point_cloud._points.data(), vertices->buffer.get(), vertices->buffer.size_bytes());

            int counter = 0;
            for (const auto& v : point_cloud._points) {
                std::cout << "\tRead Vertex: " << v.x << " " << v.y << " " << v.z << std::endl;
                if (counter++ > 9) {
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "tinyply exception: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Error: vertices not found" << std::endl;
        exit(0);
    }

    if (normals) {
        std::cout << "\tRead " << normals->count << " total vertex normals " << std::endl;
        try {
            point_cloud._normals.resize(normals->count);
            std::memcpy(point_cloud._normals.data(), normals->buffer.get(), normals->buffer.size_bytes());

            int counter = 0;
            for (const auto& n : point_cloud._normals) {
                std::cout << "\tRead Colors: " << static_cast<int>(n.x) << " " << static_cast<int>(n.y) << " " << static_cast<int>(n.z) << std::endl;
                if (counter++ > 9) {
                    break;
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "tinyply exception: " << e.what() << std::endl;
        }
    }

    if (colors) {
        std::cout << "\tRead " << colors->count << " total vertex colors " << std::endl;
        try {
            point_cloud._colors.resize(colors->count);
            std::memcpy(point_cloud._colors.data(), colors->buffer.get(), colors->buffer.size_bytes());

            int counter = 0;
            for (const auto& c : point_cloud._colors) {
                std::cout << "\tRead Colors: " << static_cast<int>(c.r) << " " << static_cast<int>(c.g) << " " << static_cast<int>(c.b) << std::endl;
                if (counter++ > 9) {
                    break;
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "tinyply exception: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Error: colors not found" << std::endl;
        exit(0);
    }

    return point_cloud;
}

void write_ply_file(const std::filesystem::path& file_path, const PointCloud& point_cloud) {

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
                                                      tinyply::Type::FLOAT32, point_cloud._points.size(),
                                                      const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(point_cloud._points.data())),
                                                      tinyply::Type::INVALID,
                                                      0);
    }

    if (!point_cloud._normals.empty()) {
        binary_point3D_file.add_properties_to_element("vertex", {"nx", "ny", "nz"},
                                                      tinyply::Type::FLOAT32,
                                                      point_cloud._normals.size(),
                                                      const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(point_cloud._normals.data())),
                                                      tinyply::Type::INVALID,
                                                      0);
    }

    if (!point_cloud._colors.empty()) {

        binary_point3D_file.add_properties_to_element("vertex", {"red", "green", "blue"},
                                                      tinyply::Type::UINT8, point_cloud._colors.size(),
                                                      const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(point_cloud._colors.data())),
                                                      tinyply::Type::INVALID,
                                                      0);
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
    const auto image_count = read_binary_value<uint64_t>(*image_stream_buffer);

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

        const auto number_points = read_binary_value<uint64_t>(*image_stream_buffer);

        // Read all the point data at once
        img._points2D_ID.resize(number_points);
        image_stream_buffer->read(reinterpret_cast<char*>(img._points2D_ID.data()), number_points * sizeof(ImagePoint));
    }

    return images;
}

// TODO: Do something with the cameras vector
// adapted from https://github.com/colmap/colmap/blob/dev/src/colmap/base/reconstruction.cc
std::unordered_map<uint32_t, CameraInfo> read_cameras_binary(std::filesystem::path file_path) {
    auto camera_stream_buffer = read_binary(file_path);
    const auto camera_count = read_binary_value<uint64_t>(*camera_stream_buffer);

    std::unordered_map<uint32_t, CameraInfo> cameras;
    cameras.reserve(camera_count);

    for (size_t i = 0; i < camera_count; ++i) {
        auto cam = CameraInfo();
        cam._camera_ID = read_binary_value<uint32_t>(*camera_stream_buffer);
        auto model_id = read_binary_value<int>(*camera_stream_buffer);
        cam._width = read_binary_value<uint64_t>(*camera_stream_buffer);
        cam._height = read_binary_value<uint64_t>(*camera_stream_buffer);
        cam._camera_model = std::get<0>(camera_model_ids[model_id]);
        auto camera_param_count = std::get<1>(camera_model_ids[model_id]);
        cam._params.resize(camera_param_count);
        camera_stream_buffer->read(reinterpret_cast<char*>(cam._params.data()), cam._params.size() * sizeof(double));
        cameras.emplace(cam._camera_ID, cam);
    }

    return cameras;
}

// adapted from https://github.com/colmap/colmap/blob/dev/src/colmap/base/reconstruction.cc
// TODO: There should be points3D data returned
PointCloud read_point3D_binary(std::filesystem::path file_path) {
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

        const auto track_length = read_binary_value<uint64_t>(*point3D_stream_buffer);
        std::vector<Track> tracks;
        tracks.resize(track_length);
        point3D_stream_buffer->read(reinterpret_cast<char*>(tracks.data()), track_length * sizeof(Track));
    }

    write_ply_file(file_path.parent_path() / "points3D.ply", point_cloud);
    return point_cloud;
}

std::vector<CameraInfo> read_colmap_cameras(const std::filesystem::path file_path,
                                            const std::unordered_map<uint32_t, CameraInfo>& cameras,
                                            const std::vector<Image>& images) {
    std::vector<CameraInfo> camera_infos(images.size());

    for (size_t image_ID = 0; image_ID < images.size(); ++image_ID) {
        // Make a copy of the image object to avoid accessing the shared resource
        Image image = images[image_ID];
        auto it = cameras.find(image._camera_id);
        camera_infos[image_ID] = it->second; // Make a copy

        //        std::cout << "Image widht: " << camera_infos[image_ID]._im  << std::endl;
        //        std::cout << "Image height: " << camera_infos[image_ID]._image_height << std::endl;
        auto [img_data, width, height, channels] = read_image(file_path / image._name);
        camera_infos[image_ID]._img_w = width;
        camera_infos[image_ID]._img_h = height;
        camera_infos[image_ID]._channels = channels;
        camera_infos[image_ID]._img_data = img_data;

        camera_infos[image_ID]._R = qvec2rotmat(image._qvec).transpose();
        camera_infos[image_ID]._T = image._tvec;

        if (camera_infos[image_ID]._camera_model == CAMERA_MODEL::SIMPLE_PINHOLE) {
            double focal_length_x = camera_infos[image_ID]._params[0];
            camera_infos[image_ID]._fov_x = focal2fov(focal_length_x, camera_infos[image_ID]._width);
            camera_infos[image_ID]._fov_y = focal2fov(focal_length_x, camera_infos[image_ID]._height);
        } else if (camera_infos[image_ID]._camera_model == CAMERA_MODEL::PINHOLE) {
            double focal_length_x = camera_infos[image_ID]._params[0];
            double focal_length_y = camera_infos[image_ID]._params[1];
            camera_infos[image_ID]._fov_x = focal2fov(focal_length_x, camera_infos[image_ID]._width);
            camera_infos[image_ID]._fov_y = focal2fov(focal_length_y, camera_infos[image_ID]._height);
        } else {
            throw std::runtime_error("Camera model not supported");
        }

        camera_infos[image_ID]._image_name = image._name;
        camera_infos[image_ID]._image_path = file_path / image._name;
    }
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
        Eigen::Matrix4d W2C = getWorld2View2Eigen(cam._R, cam._T);
        Eigen::Matrix4d C2W = W2C.inverse();
        cam_centers.emplace_back(C2W.block<3, 1>(0, 3));
    }

    auto [center, diagonal] = get_center_and_diag(cam_centers);

    double radius = diagonal * 1.1;
    Eigen::Vector3d translate = -center;

    return {translate, radius};
}

std::unique_ptr<SceneInfo> read_colmap_scene_info(std::filesystem::path file_path) {
    auto cameras = read_cameras_binary(file_path / "sparse/0/cameras.bin");
    auto images = read_images_binary(file_path / "sparse/0/images.bin");

    auto sceneInfos = std::make_unique<SceneInfo>();
    if (!std::filesystem::exists(file_path / "sparse/0/points3D.ply")) {
        sceneInfos->_point_cloud = read_point3D_binary(file_path / "sparse/0/points3D.bin");
    } else {
        sceneInfos->_point_cloud = read_ply_file(file_path / "sparse/0/points3D.ply");
    }
    sceneInfos->_ply_path = file_path / "sparse/0/points3D.ply";
    sceneInfos->_cameras = read_colmap_cameras(file_path / "images", cameras, images);
    auto [translate, radius] = getNerfppNorm(sceneInfos->_cameras);
    sceneInfos->_nerf_norm_radius = radius;
    sceneInfos->_nerf_norm_translation = translate;
    return sceneInfos;
}
