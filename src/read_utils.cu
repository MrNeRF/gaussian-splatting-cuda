#include "read_utils.cuh"
#include <vector>
#include <fstream>
#include <memory>
#include <iostream>


std::unique_ptr<std::istream> read_binary(std::filesystem::path filepath){
    std::ifstream file(filepath, std::ios::binary);
    std::unique_ptr<std::istream> file_stream;
    if (file.fail()) {
        throw std::runtime_error("Failed to open file: " + filepath.string());
    }
    // preload
    std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(file), {});
    file_stream = std::make_unique<std::stringstream>(std::string(buffer.begin(), buffer.end()));
    return file_stream;
}

void read_ply_file(std::filesystem::path filepath){
    auto ply_stream_buffer = read_binary(filepath);
    ply_stream_buffer->seekg(0, std::ios::end);
    const float size_mb = ply_stream_buffer->tellg() * 1e-6f;
    ply_stream_buffer->seekg(0, std::ios::beg);
    
    tinyply::PlyFile ply_file;
    ply_file.parse_header(*ply_stream_buffer);

    std::cout << "\t[ply_header] Type: " << (ply_file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto & c : ply_file.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto & c : ply_file.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl;

    for (const auto & e : ply_file.get_elements())
    {
        std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
        for (const auto & p : e.properties)
        {
            std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
            if (p.isList) std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
            std::cout << std::endl;
        }
    }
}