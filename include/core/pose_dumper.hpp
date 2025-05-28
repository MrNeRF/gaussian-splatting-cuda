/**
* pose_dumper_rt.hpp  ─  Dump only R (3×3) and T (3×1) for every camera.
*
* File layout (little-endian, float32):
*   uint32  N                      ─ number of cameras
*   N × 36B R00 R01 … R22          ─ row-major rotation matrix
*   N × 12B Tx Ty Tz               ─ translation
* Total per camera: 48 bytes.
*
* Works with:
*   • CameraInfo stored directly in a std::vector
*   • CameraInfo stored as the .second of a {key, value} pair in
*     std::map / std::unordered_map
* and supports Eigen or libtorch types for _R and _T.
*/

#pragma once
#include <array>
#include <vector>
#include <fstream>
#include <cstring>
#include <type_traits>
#include <stdexcept>

#ifdef __has_include
#  if __has_include(<torch/torch.h>)
#    include <torch/torch.h>
#    define PD_RT_HAS_TORCH 1
#  endif
#  if __has_include(<Eigen/Dense>)
#    include <Eigen/Dense>
#    define PD_RT_HAS_EIGEN 1
#  endif
#endif

namespace posedump_rt {

   /* ------------------------------------------------------------------ */
   /* helpers                                                            */
   /* ------------------------------------------------------------------ */
   template<typename T> struct always_false : std::false_type {};

/* ----- Eigen overloads -------------------------------------------- */
#if PD_RT_HAS_EIGEN
   template<class Derived>
   auto to_array_R(const Eigen::MatrixBase<Derived>& M)
       -> std::enable_if_t<(Derived::RowsAtCompileTime == 3 &&
                            Derived::ColsAtCompileTime == 3),
                           std::array<float,9>>
   {
       std::array<float,9> out;
       Eigen::Matrix<float,3,3,Eigen::RowMajor> tmp = M.template cast<float>();
       std::memcpy(out.data(), tmp.data(), 9 * sizeof(float));
       return out;
   }

   template<class Derived>
   auto to_array_T(const Eigen::MatrixBase<Derived>& t)
       -> std::enable_if_t<(Derived::RowsAtCompileTime == 3 &&
                            Derived::ColsAtCompileTime == 1),
                           std::array<float,3>>
   {
       return {static_cast<float>(t(0)),
               static_cast<float>(t(1)),
               static_cast<float>(t(2))};
   }
#endif  /* PD_RT_HAS_EIGEN */

/* ----- libtorch overloads ----------------------------------------- */
#if PD_RT_HAS_TORCH
   inline std::array<float,9> to_array_R(const torch::Tensor& M)
   {
       auto r = M.to(torch::kCPU).contiguous().view({9});
       return {r[0].item<float>(), r[1].item<float>(), r[2].item<float>(),
               r[3].item<float>(), r[4].item<float>(), r[5].item<float>(),
               r[6].item<float>(), r[7].item<float>(), r[8].item<float>()};
   }

   inline std::array<float,3> to_array_T(const torch::Tensor& t)
   {
       auto v = t.to(torch::kCPU).contiguous().view({3});
       return {v[0].item<float>(), v[1].item<float>(), v[2].item<float>()};
   }
#endif  /* PD_RT_HAS_TORCH */

   /* ----- fall-backs that trigger a clear error ---------------------- */
   template<typename X>
   auto to_array_R(const X&) -> std::array<float,9>
   { static_assert(always_false<X>::value, "Unsupported _R type"); }

   template<typename X>
   auto to_array_T(const X&) -> std::array<float,3>
   { static_assert(always_false<X>::value, "Unsupported _T type"); }

   /* ------------------------------------------------------------------ */
   /* dump()                                                             */
   /* ------------------------------------------------------------------ */
   template<typename CameraContainer>
   void dump(const CameraContainer& cams, const std::string& path)
   {
       std::ofstream ofs(path, std::ios::binary);
       if (!ofs)
           throw std::runtime_error("pose_dumper_rt: cannot open " + path);

       const uint32_t N = static_cast<uint32_t>(cams.size());
       ofs.write(reinterpret_cast<const char*>(&N), sizeof(N));

       for (const auto& c : cams)
       {
           /* Accept either a bare CameraInfo or a map pair<key, CameraInfo> */
           using C = std::remove_cvref_t<decltype(c)>;
           const auto& cam = [&]() -> const auto& {
               if constexpr (std::is_class_v<C> &&
                             std::is_member_pointer_v<decltype(&C::second)>)
                   return c.second;      // map / unordered_map value
               else
                   return c;             // plain CameraInfo
           }();

           auto R = to_array_R(cam._R);
           auto T = to_array_T(cam._T);

           ofs.write(reinterpret_cast<const char*>(R.data()),
                     R.size() * sizeof(float));
           ofs.write(reinterpret_cast<const char*>(T.data()),
                     T.size() * sizeof(float));
       }
   }

} // namespace posedump_rt
