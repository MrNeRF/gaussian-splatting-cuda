//
// Created by paja on 8/30/24.
//

#ifndef GAUSSIAN_SPLATTING_CUDA_NANO_KDTREE_CUH
#define GAUSSIAN_SPLATTING_CUDA_NANO_KDTREE_CUH

#include <nanoflann.hpp>
#include <torch/torch.h>

class Nano_kdtree {
public:
    using KdTreeTensor = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, Nano_kdtree>,
        Nano_kdtree, 3, size_t>;

    Nano_kdtree(const torch::Tensor& xyz)
        : _xyz(xyz),
          _accessor(_xyz.accessor<float, 2>()),
          _kdTree(nullptr)
    {}

    ~Nano_kdtree() {
        clearKdTree<KdTreeTensor>();
    }

    [[nodiscard]] torch::Tensor compute_scales() const;

    template <typename T>
    T* getKdTree() const {
        return reinterpret_cast<T*>(_kdTree);
    }

    template <typename T>
    T* ensureKdTree() {
        return _kdTree ? reinterpret_cast<T*>(_kdTree) : createKdTree<T>();
    }

    inline size_t kdtree_get_point_count() const { return _xyz.size(0); }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX & bounding_box) const {
        return false;
    }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return _accessor[idx][dim];
    }

    template <typename T>
    T* createKdTree() {
        if (!_kdTree) {
            _kdTree = new T(3, *this, { 10 });
        }
        return reinterpret_cast<T*>(_kdTree);
    }

    template <typename T>
    void clearKdTree() {
        if (_kdTree) {
            delete reinterpret_cast<T*>(_kdTree);
            _kdTree = nullptr;
        }
    }

private:
    torch::Tensor _xyz;
    torch::TensorAccessor<float, 2> _accessor;
    void* _kdTree;
};

#endif // GAUSSIAN_SPLATTING_CUDA_NANO_KDTREE_CUH
