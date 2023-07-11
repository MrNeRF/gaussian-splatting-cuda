#include "image.cuh"
#include "read_utils.cuh"
#include <fstream>
#include <iostream>
#include <memory>
#include <tinyply.h>
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

// copied from https://github.com/colmap/colmap/blob/dev/src/colmap/base/reconstruction.cc
void read_images_binary(std::filesystem::path file_path) {
    auto image_stream_buffer = read_binary(file_path / "images.bin");
    const size_t number_images = read_binary_value<uint64_t>(*image_stream_buffer);

    for (size_t i = 0; i < number_images; ++i) {
        Image img;
        img._id = read_binary_value<uint32_t>(*image_stream_buffer);
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
}

void read_colmap_scene_info(std::filesystem::path file_path) {
    read_images_binary(file_path);
}