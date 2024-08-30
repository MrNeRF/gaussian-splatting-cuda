#include "camera_info.cuh"
#include "camera_utils.cuh"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "future"
#include "image.cuh"
#include "point_cloud.cuh"
#include "read_utils.cuh"
#include <algorithm>
#include <exception>
#include <thread>
#include <filesystem>
#include <cstring>
#include <sstream>
#include <iomanip>
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


PointCloud read_ply_file(std::filesystem::path file_path) {
    auto ply_stream_buffer = read_binary(file_path);
    tinyply::PlyFile file;
    std::shared_ptr<tinyply::PlyData> vertices, normals, colors;
    file.parse_header(*ply_stream_buffer);

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

    // Lambda function to handle copying data in parallel
    auto copy_data = [](std::shared_ptr<tinyply::PlyData> ply_data, auto& destination, const std::string& name) {
        if (ply_data) {
            std::cout << "\tRead " << ply_data->count << " total " << name << std::endl;
            try {
                destination.resize(ply_data->count);
                std::memcpy(destination.data(), ply_data->buffer.get(), ply_data->buffer.size_bytes());
            } catch (const std::exception& e) {
                std::cerr << "tinyply exception: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Error: " << name << " not found" << std::endl;
        }
    };

    // Launch parallel tasks

    auto vertices_future = std::async(std::launch::async,
                                      [copy_data, vertices, &capture0 = point_cloud._points] { return copy_data(vertices, capture0, "vertices"); });

    auto normals_future = std::async(std::launch::async,
                                     [copy_data, normals, &capture0 = point_cloud._normals] { return copy_data(normals, capture0, "vertex normals"); });

    auto colors_future = std::async(std::launch::async,
                                    [copy_data, colors, &capture0 = point_cloud._colors] { return copy_data(colors, capture0, "vertex colors"); });

    // Wait for all tasks to complete
    vertices_future.get();
    normals_future.get();
    colors_future.get();

    return point_cloud;
}

void Write_output_ply(const std::filesystem::path& file_path, const std::vector<torch::Tensor>& tensors, const std::vector<std::string>& attribute_names) {
    tinyply::PlyFile plyFile;

    size_t attribute_offset = 0; // An offset to track the attribute names

    for (size_t i = 0; i < tensors.size(); ++i) {
        // Calculate the number of columns in the tensor.
        size_t columns = tensors[i].size(1);

        std::vector<std::string> current_attributes;
        for (size_t j = 0; j < columns; ++j) {
            current_attributes.push_back(attribute_names[attribute_offset + j]);
        }

        plyFile.add_properties_to_element(
            "vertex",
            current_attributes,
            tinyply::Type::FLOAT32,
            tensors[i].size(0),
            reinterpret_cast<uint8_t*>(tensors[i].data_ptr<float>()),
            tinyply::Type::INVALID,
            0);

        attribute_offset += columns; // Increase the offset for the next tensor.
    }

    std::filebuf fb;
    fb.open(file_path, std::ios::out | std::ios::binary);
    std::ostream outputStream(&fb);
    plyFile.write(outputStream, true); // 'true' for binary format
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

// adapted from https://github.com/colmap/colmap/blob/dev/src/colmap/base/reconstruction.cc
std::vector<Image> read_images_binary(std::filesystem::path file_path) {
    struct ImagePoint { // we dont need this later
        double _x;
        double _y;
        uint64_t _point_id;
    };

    auto image_stream_buffer = read_binary(file_path);
    const auto image_count = read_binary_value<uint64_t>(*image_stream_buffer);

    std::vector<Image> images;
    images.reserve(image_count);

    for (size_t i = 0; i < image_count; ++i) {
        const auto image_ID = read_binary_value<uint32_t>(*image_stream_buffer);
        auto& img = images.emplace_back(image_ID);
        img._qvec.x() = static_cast<float>(read_binary_value<double>(*image_stream_buffer));
        img._qvec.y() = static_cast<float>(read_binary_value<double>(*image_stream_buffer));
        img._qvec.z() = static_cast<float>(read_binary_value<double>(*image_stream_buffer));
        img._qvec.w() = static_cast<float>(read_binary_value<double>(*image_stream_buffer));
        img._qvec.normalize();

        img._tvec.x() = static_cast<float>(read_binary_value<double>(*image_stream_buffer));
        img._tvec.y() = static_cast<float>(read_binary_value<double>(*image_stream_buffer));
        img._tvec.z() = static_cast<float>(read_binary_value<double>(*image_stream_buffer));

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
        std::vector<ImagePoint> points(number_points); // we throw this away
        image_stream_buffer->read(reinterpret_cast<char*>(points.data()), number_points * sizeof(ImagePoint));
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
    auto point3D_stream = read_binary(file_path);

    // Read point count
    uint64_t point3D_count;
    point3D_stream->read(reinterpret_cast<char*>(&point3D_count), sizeof(uint64_t));

    struct Track {
        uint32_t _image_ID;
        uint32_t _max_num_2D_points;
    };

    PointCloud point_cloud;
    point_cloud._points.resize(point3D_count);
    point_cloud._colors.resize(point3D_count);

    // Read the entire file into a vector
    std::vector<char> buffer(std::istreambuf_iterator<char>(*point3D_stream), {});

    // Process the data in parallel
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = point3D_count / num_threads;

    std::vector<std::future<void>> futures;

    auto process_chunk = [&](size_t start, size_t end) {
        const char* data = buffer.data();
        for (size_t i = start; i < end; ++i) {
            // Skip point3D_ID
            data += sizeof(uint64_t);

            // Read vertices
            point_cloud._points[i].x = static_cast<float>(*reinterpret_cast<const double*>(data));
            data += sizeof(double);
            point_cloud._points[i].y = static_cast<float>(*reinterpret_cast<const double*>(data));
            data += sizeof(double);
            point_cloud._points[i].z = static_cast<float>(*reinterpret_cast<const double*>(data));
            data += sizeof(double);

            // Read colors
            point_cloud._colors[i].r = static_cast<uint8_t>(*data++);
            point_cloud._colors[i].g = static_cast<uint8_t>(*data++);
            point_cloud._colors[i].b = static_cast<uint8_t>(*data++);

            // Skip error
            data += sizeof(double);

            // Skip track data
            uint64_t track_length;
            std::memcpy(&track_length, data, sizeof(uint64_t));
            data += sizeof(uint64_t);
            data += track_length * sizeof(Track);
        }
    };

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? point3D_count : (t + 1) * chunk_size;
        futures.emplace_back(std::async(std::launch::async, process_chunk, start, end));
    }

    for (auto& future : futures) {
        future.wait();
    }

    // We will make this optional for debugging
    // write_ply_file(file_path.parent_path() / "points3D.ply", point_cloud);
    return point_cloud;
}


std::vector<CameraInfo> read_colmap_cameras(const std::filesystem::path& file_path,
                                            const std::unordered_map<uint32_t, CameraInfo>& cameras,
                                            const std::vector<Image>& images,
                                            int resolution) {
    std::vector<CameraInfo> camera_infos(images.size());

    // Use TBB's parallel_for to process images in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, images.size()),
                      [&](const tbb::blocked_range<size_t>& range) {
                          for (size_t i = range.begin(); i < range.end(); ++i) {
                              const Image& image = images[i];
                              auto it = cameras.find(image._camera_id);
                              if (it == cameras.end()) {
                                  throw std::runtime_error("Camera ID " + std::to_string(image._camera_id) + " not found");
                              }

                              CameraInfo& camera_info = camera_infos[i];
                              camera_info = it->second;  // Copy the CameraInfo

                              // Load and process the image
                              auto [img_data, width, height, channels] = read_image(file_path / image._name, resolution);
                              camera_info._img_w = width;
                              camera_info._img_h = height;
                              camera_info._channels = channels;
                              camera_info._img_data = img_data;

                              camera_info._R = qvec2rotmat(image._qvec).transpose();
                              camera_info._T = image._tvec;

                              camera_info._image_name = image._name;
                              camera_info._image_path = file_path / image._name;

                              switch (camera_info._camera_model) {
                              case CAMERA_MODEL::SIMPLE_PINHOLE: {
                                  const float focal_length_x = camera_info._params[0];
                                  camera_info._fov_x = focal2fov(focal_length_x, camera_info._width);
                                  camera_info._fov_y = focal2fov(focal_length_x, camera_info._height);
                              } break;
                              case CAMERA_MODEL::PINHOLE: {
                                  const float focal_length_x = camera_info._params[0];
                                  const float focal_length_y = camera_info._params[1];
                                  camera_info._fov_x = focal2fov(focal_length_x, camera_info._width);
                                  camera_info._fov_y = focal2fov(focal_length_y, camera_info._height);
                              } break;
                              case CAMERA_MODEL::SIMPLE_RADIAL:
                                  throw std::runtime_error("Camera model SIMPLE_RADIAL not supported");
                              case CAMERA_MODEL::RADIAL:
                                  throw std::runtime_error("Camera model RADIAL not supported");
                              case CAMERA_MODEL::OPENCV:
                                  throw std::runtime_error("Camera model OPENCV not supported");
                              case CAMERA_MODEL::OPENCV_FISHEYE:
                                  throw std::runtime_error("Camera model OPENCV_FISHEYE not supported");
                              case CAMERA_MODEL::FULL_OPENCV:
                                  throw std::runtime_error("Camera model FULL_OPENCV not supported");
                              case CAMERA_MODEL::FOV:
                                  throw std::runtime_error("Camera model FOV not supported");
                              case CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE:
                                  throw std::runtime_error("Camera model SIMPLE_RADIAL_FISHEYE not supported");
                              case CAMERA_MODEL::RADIAL_FISHEYE:
                                  throw std::runtime_error("Camera model RADIAL_FISHEYE not supported");
                              case CAMERA_MODEL::THIN_PRISM_FISHEYE:
                                  throw std::runtime_error("Camera model THIN_PRISM_FISHEYE not supported");
                              case CAMERA_MODEL::UNDEFINED:
                                  throw std::runtime_error("Camera model UNDEFINED (and thus not supported)");
                              default:
                                  throw std::runtime_error("Camera model not supported");
                              }
                          }
                      });

    return camera_infos;
}

std::pair<Eigen::Vector3f, float> get_center_and_diag(std::vector<Eigen::Vector3f>& cam_centers) {
    Eigen::Vector3f avg_cam_center = Eigen::Vector3f::Zero();
    for (const auto& center : cam_centers) {
        avg_cam_center += center;
    }
    avg_cam_center /= static_cast<float>(cam_centers.size());

    float max_dist = 0;
    for (const auto& center : cam_centers) {
        max_dist = std::max(max_dist, (center - avg_cam_center).norm());
    }

    return {avg_cam_center, max_dist};
}

void getNerfppNorm(SceneInfo* scene_info) {
    std::vector<Eigen::Vector3f> cam_centers;
    for (CameraInfo& cam : scene_info->_cameras) {
        Eigen::Matrix4f W2C = getWorld2View2Eigen(cam._R, cam._T);
        Eigen::Matrix4f C2W = W2C.inverse();
        cam_centers.emplace_back(C2W.block<3, 1>(0, 3));
    }

    auto [center, diagonal] = get_center_and_diag(cam_centers);

    scene_info->_nerf_norm_radius = diagonal * 1.1f;
    scene_info->_nerf_norm_translation = -center;
}

std::unique_ptr<SceneInfo> read_colmap_scene_info(std::filesystem::path file_path, int resolution) {
    auto cameras = read_cameras_binary(file_path / "sparse/0/cameras.bin");
    auto images = read_images_binary(file_path / "sparse/0/images.bin");

    auto sceneInfos = std::make_unique<SceneInfo>();
    sceneInfos->_point_cloud = read_point3D_binary(file_path / "sparse/0/points3D.bin");

    sceneInfos->_ply_path = file_path / "sparse/0/points3D.ply";
    sceneInfos->_cameras = read_colmap_cameras(file_path / "images", cameras, images, resolution);

    getNerfppNorm(sceneInfos.get());

    const auto print_stats = [&]() {
        const auto& cam_0 = sceneInfos->_cameras[0];
        const auto n_cams = sceneInfos->_cameras.size();

        const float image_mpixels = cam_0._img_w * cam_0._img_h / 1'000'000.0f;
        const std::string resized = resolution == 2 || resolution == 4 || resolution == 8 ? " (resized) " : "";
        std::cout << "Training with " << n_cams << " images of "
                  << cam_0._img_w << " x " << cam_0._img_h << resized + " pixels ("
                  << std::fixed << std::setprecision(3) << image_mpixels << " Mpixel per image, "
                  << std::fixed << std::setprecision(1) << image_mpixels * n_cams << " Mpixel total)" << std::endl;
    };

    print_stats();
    return sceneInfos;
}
