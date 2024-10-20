#include <cstddef>
#include <cuda.h>

template <int Size, typename T>
__global__ void __scalar_assign(T * __restrict__ data, T scalar)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < Size; i += stride)
    {
        data[i] = scalar;
    }
}

template <int Size, typename T>
__global__ void __vector_assign(T * __restrict__ lhs, T * __restrict__ rhs)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < Size; i += stride)
    {
        lhs[i] = rhs[i];
    }
}
template <int Size, typename T>
__global__ void __scalar_add(T * __restrict__ data, T scalar)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < Size; i += stride)
    {
        data[i] += scalar;
    }
}

template <int Size, typename T>
__global__ void __vector_add(T * __restrict__ lhs, T * __restrict__ rhs)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < Size; i += stride)
    {
        lhs[i] += rhs[i];
    }
}

template <int Size, typename T>
class InstantiatedShape
{
private:
    T* data;

public:
    InstantiatedShape()
    {
        cudaMalloc((void**)&this->data, sizeof(T) * Size);
    }

    InstantiatedShape& operator=(T scalar)
    {
        __scalar_assign<Size, T><<<128, 128>>>(this->data, scalar);
        return *this;
    }

    InstantiatedShape& operator=(const InstantiatedShape<Size, T>& rhs)
    {
        __vector_assign<Size, T><<<128, 128>>>(this->data, rhs.data);
        return *this;
    }

    InstantiatedShape& operator+=(T scalar)
    {
        __scalar_add<Size, T><<<128, 128>>>(this->data, scalar);
        return *this;
    }

    InstantiatedShape& operator+=(const InstantiatedShape<Size, T>& rhs)
    {
        __vector_add<Size, T><<<128, 128>>>(this->data, rhs.data);
        return *this;
    }

    T operator[](int idx)
    {
        T result;
        cudaMemcpy(&result, this->data + idx, sizeof(T), cudaMemcpyDeviceToHost);
        return result;
    }

    ~InstantiatedShape()
    {
        cudaFree(this->data);
    }
};

template <int Size>
struct Shape
{
    template <typename T>
    using shape = InstantiatedShape<Size, T>;
};