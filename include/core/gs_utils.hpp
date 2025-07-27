#pragma once

#include <string>
#include <sstream>

namespace gs {
inline std::string get_tensor_key(const void* ptr) {
    std::stringstream ss;
    ss << ptr;
    return ss.str();
}

}