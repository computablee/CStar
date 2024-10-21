#ifndef __CSTAR
#define __CSTAR

#include <cstddef>
#include <cuda.h>
#include <iostream>

namespace CStar
{

template <typename T>
__global__ void __scalar_assign(T * __restrict__ data, T scalar, size_t size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size; i += stride)
    {
        data[i] = scalar;
    }
}

template <typename T>
__global__ void __vector_assign(T * __restrict__ lhs, T * __restrict__ rhs, size_t size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size; i += stride)
    {
        lhs[i] = rhs[i];
    }
}

template <typename T>
__global__ void __scalar_add(T * __restrict__ data, T scalar, size_t size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size; i += stride)
    {
        data[i] += scalar;
    }
}

template <typename T>
__global__ void __vector_add(T * __restrict__ lhs, T * __restrict__ rhs, size_t size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size; i += stride)
    {
        lhs[i] += rhs[i];
    }
}

template <typename T>
__global__ void __scalar_mult(T * __restrict__ data, T scalar, size_t size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size; i += stride)
    {
        data[i] *= scalar;
    }
}

template <typename T>
__global__ void __vector_mult(T * __restrict__ lhs, T * __restrict__ rhs, size_t size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size; i += stride)
    {
        lhs[i] *= rhs[i];
    }
}

// This reduction code is provided courtesy of Nvidia's slides
// See: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <typename T, unsigned int BlockSize>
__device__ void __warpReduce_add(volatile T * __restrict__ sdata, unsigned int tid)
{
    if constexpr (BlockSize >= 64) sdata[tid] += sdata[tid + 32];
    if constexpr (BlockSize >= 32) sdata[tid] += sdata[tid + 16];
    if constexpr (BlockSize >= 16) sdata[tid] += sdata[tid + 8];
    if constexpr (BlockSize >=  8) sdata[tid] += sdata[tid + 4];
    if constexpr (BlockSize >=  4) sdata[tid] += sdata[tid + 2];
    if constexpr (BlockSize >=  2) sdata[tid] += sdata[tid + 1];
}

template <typename T, unsigned int BlockSize>
__global__ void __reduce_add(T * __restrict__ idata, T * __restrict__ odata, size_t size)
{
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (BlockSize * 2) + tid;
    unsigned int gridSize = BlockSize * 2 * gridDim.x;
    sdata[tid] = static_cast<T>(0);

    while (i + BlockSize < size)
    {
        sdata[tid] += idata[i] + idata[i + BlockSize];
        i += gridSize;
    }

    __syncthreads();

    while (i < size)
    {
        sdata[tid] += idata[i];
        i += gridSize;
    }

    __syncthreads();

    if constexpr (BlockSize >= 512)
    {
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];

        __syncthreads();
    }

    if constexpr (BlockSize >= 256)
    {
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];

        __syncthreads();
    }

    if constexpr (BlockSize >= 128)
    {
        if (tid < 64)
            sdata[tid] += sdata[tid + 64];

        __syncthreads();
    }

    if (tid < 32) __warpReduce_add<T, BlockSize>(sdata, tid);
    if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template <typename T, unsigned int BlockSize>
__device__ void __warpReduce_mult(volatile T * __restrict__ sdata, unsigned int tid)
{
    if constexpr (BlockSize >= 64) sdata[tid] *= sdata[tid + 32];
    if constexpr (BlockSize >= 32) sdata[tid] *= sdata[tid + 16];
    if constexpr (BlockSize >= 16) sdata[tid] *= sdata[tid + 8];
    if constexpr (BlockSize >=  8) sdata[tid] *= sdata[tid + 4];
    if constexpr (BlockSize >=  4) sdata[tid] *= sdata[tid + 2];
    if constexpr (BlockSize >=  2) sdata[tid] *= sdata[tid + 1];
}

template <typename T, unsigned int BlockSize>
__global__ void __reduce_mult(T * __restrict__ idata, T * __restrict__ odata, size_t size)
{
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (BlockSize * 2) + tid;
    unsigned int gridSize = BlockSize * 2 * gridDim.x;
    sdata[tid] = static_cast<T>(1);

    while (i + BlockSize < size)
    {
        sdata[tid] *= idata[i] * idata[i + BlockSize];
        i += gridSize;
    }

    __syncthreads();

    while (i < size)
    {
        sdata[tid] *= idata[i];
        i += gridSize;
    }

    __syncthreads();

    if constexpr (BlockSize >= 512)
    {
        if (tid < 256)
            sdata[tid] *= sdata[tid + 256];

        __syncthreads();
    }

    if constexpr (BlockSize >= 256)
    {
        if (tid < 128)
            sdata[tid] *= sdata[tid + 128];

        __syncthreads();
    }

    if constexpr (BlockSize >= 128)
    {
        if (tid < 64)
            sdata[tid] *= sdata[tid + 64];

        __syncthreads();
    }

    if (tid < 32) __warpReduce_mult<T, BlockSize>(sdata, tid);
    if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template <typename T, int ... Size>
class InstantiatedShape;

template <typename T, int ... Size>
class InstantiatedShape final
{
private:
    T* data;
    size_t length;
    
    template <typename... Idx>
    size_t compute_index(Idx... idxs) const
    {
        static_assert(sizeof...(Size) == sizeof...(idxs), "Number of indices must match the number of dimensions.");
        static_assert((... && std::is_same<Idx, int>::value), "Indices must be of type int.");
        size_t indices[] = { static_cast<size_t>(idxs)... };
        size_t sizes[] = { static_cast<size_t>(Size)... };

        size_t index = 0;
        size_t multiplier = 1;
        for (int i = sizeof...(Size) - 1; i >= 0; --i)
        {
            index += indices[i] * multiplier;
            multiplier *= sizes[i];
        }
        return index;
    }

    class Reference
    {
        T* data;
        size_t index;

        friend class InstantiatedShape<T, Size ...>;

        Reference(T* data, size_t index) : data(data), index(index) { }

    public:
        operator T()
        {
            T result;
            cudaMemcpy(&result, this->data + index, sizeof(T), cudaMemcpyDeviceToHost);
            return result;
        }

        Reference& operator=(const T& rhs)
        {
            cudaMemcpy(this->data + index, &rhs, sizeof(T), cudaMemcpyHostToDevice);
            return *this;
        }

        Reference& operator=(const Reference& rhs)
        {
            cudaMemcpy(this->data + this->index, rhs.data + rhs.index, sizeof(T), cudaMemcpyDeviceToDevice);
            return *this;
        }
    };

public:
    InstantiatedShape() : length((... * Size))
    {
        cudaMalloc((void**)&this->data, sizeof(T) * length);
    }

    InstantiatedShape(T init) : length((... * Size))
    {
        cudaMalloc((void**)&this->data, sizeof(T) * length);
        __scalar_assign<T><<<this->length / 128, 128>>>(this->data, init, this->length);
        cudaDeviceSynchronize();
    }

    InstantiatedShape(const InstantiatedShape<T, Size ...>& init) : length((... * Size))
    {
        cudaMalloc((void**)&this->data, sizeof(T) * length);
        __vector_assign<T><<<this->length / 128, 128>>>(this->data, init.data, this->length);
        cudaDeviceSynchronize();
    }

    InstantiatedShape(InstantiatedShape<T, Size ...>&& init) : length((... * Size))
    {
        this->data = init.data;
        init.data = 0;
        init.length = 0;
    }

    ~InstantiatedShape()
    {
        if (this->data)
            cudaFree(this->data);
    }

    InstantiatedShape& operator=(T scalar)
    {
        __scalar_assign<T><<<this->length / 128, 128>>>(this->data, scalar, this->length);
        cudaDeviceSynchronize();
        return *this;
    }

    InstantiatedShape& operator=(const InstantiatedShape<T, Size ...>& rhs)
    {
        __vector_assign<T><<<this->length / 128, 128>>>(this->data, rhs.data, this->length);
        cudaDeviceSynchronize();
        return *this;
    }

    InstantiatedShape& operator+=(T scalar)
    {
        __scalar_add<T><<<this->length / 128, 128>>>(this->data, scalar, this->length);
        cudaDeviceSynchronize();
        return *this;
    }

    InstantiatedShape& operator+=(const InstantiatedShape<T, Size ...>& rhs)
    {
        __vector_add<T><<<this->length / 128, 128>>>(this->data, rhs.data, this->length);
        cudaDeviceSynchronize();
        return *this;
    }

    InstantiatedShape<T, Size ...> operator+(T scalar)
    {
        InstantiatedShape<T, Size ...> temp;
        cudaMemcpy(temp.data, this->data, sizeof(T) * this->length, cudaMemcpyDeviceToDevice);
        temp += scalar;
        return temp;
    }

    InstantiatedShape<T, Size ...> operator+(const InstantiatedShape<T, Size ...>& rhs)
    {
        InstantiatedShape<T, Size ...> temp;
        cudaMemcpy(temp.data, this->data, sizeof(T) * this->length, cudaMemcpyDeviceToDevice);
        temp += rhs;
        return temp;
    }

    InstantiatedShape& operator*=(T scalar)
    {
        __scalar_mult<T><<<this->length / 128, 128>>>(this->data, scalar, this->length);
        cudaDeviceSynchronize();
        return *this;
    }

    InstantiatedShape& operator*=(const InstantiatedShape<T, Size ...>& rhs)
    {
        __vector_mult<T><<<this->length / 128, 128>>>(this->data, rhs.data, this->length);
        cudaDeviceSynchronize();
        return *this;
    }

    InstantiatedShape<T, Size ...> operator*(T scalar)
    {
        InstantiatedShape<T, Size ...> temp;
        cudaMemcpy(temp.data, this->data, sizeof(T) * this->length, cudaMemcpyDeviceToDevice);
        temp *= scalar;
        return temp;
    }

    InstantiatedShape<T, Size ...> operator*(const InstantiatedShape<T, Size ...>& rhs)
    {
        InstantiatedShape<T, Size ...> temp;
        cudaMemcpy(temp.data, this->data, sizeof(T) * this->length, cudaMemcpyDeviceToDevice);
        temp *= rhs;
        return temp;
    }

    template <typename ... Idx,
        typename = std::enable_if_t<sizeof...(Idx) == sizeof...(Size)>,
        typename = std::enable_if_t<(... && std::is_same<Idx, int>::value)>>
    Reference operator()(Idx ... idxs)
    {
        return Reference(this->data, compute_index(idxs...));
    }

    template <typename ... Idx,
        typename = std::enable_if_t<sizeof...(Idx) == sizeof...(Size)>,
        typename = std::enable_if_t<(... && std::is_same<Idx, int>::value)>>
    const Reference operator()(Idx ... idxs) const
    {
        return Reference(this->data, compute_index(idxs...));
    }

    template <typename U, int ... S>
    friend U& operator+=(U& lhs, const InstantiatedShape<U, S ...>& rhs);

    template <typename U, int ... S>
    friend U& operator*=(U& lhs, const InstantiatedShape<U, S ...>& rhs);
};

template <typename T, int ... Size>
T& operator+=(T& lhs, const InstantiatedShape<T, Size ...>& rhs)
{
    constexpr int blockSize = 128;
    const int gridSize = rhs.length / blockSize;

    T* reductionArray;
    cudaMalloc((void**)&reductionArray, sizeof(T) * gridSize);
    __reduce_add<T, blockSize><<<gridSize, blockSize>>>(rhs.data, reductionArray, rhs.length);
    cudaDeviceSynchronize();

    T hostArray[gridSize];
    cudaMemcpy(hostArray, reductionArray, sizeof(T) * gridSize, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < gridSize; i++)
        lhs += hostArray[i];

    cudaFree(reductionArray);

    return lhs;
}

template <typename T, int ... Size>
T& operator*=(T& lhs, const InstantiatedShape<T, Size ...>& rhs)
{
    constexpr int blockSize = 128;
    const int gridSize = rhs.length / blockSize;

    T* reductionArray;
    cudaMalloc((void**)&reductionArray, sizeof(T) * gridSize);
    __reduce_mult<T, blockSize><<<gridSize, blockSize>>>(rhs.data, reductionArray, rhs.length);
    cudaDeviceSynchronize();

    T hostArray[gridSize];
    cudaMemcpy(hostArray, reductionArray, sizeof(T) * gridSize, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < gridSize; i++)
        lhs *= hostArray[i];

    cudaFree(reductionArray);

    return lhs;
}

template <int ... Size>
struct Shape
{
    template <typename T>
    using shape = InstantiatedShape<T, Size ...>;
};

template <typename>
struct is_shape : std::false_type {};

template <int ... Size>
struct is_shape<Shape<Size ...>> : std::true_type {};

template <typename ShapeInstance, typename = std::enable_if_t<is_shape<ShapeInstance>::value>>
struct rankof;

template <int ... Size>
struct rankof<Shape<Size ...>> { static const int value = sizeof...(Size); };

template <typename ShapeInstance, typename = std::enable_if_t<is_shape<ShapeInstance>::value>>
constexpr int rankof_t = 0;

template <int ... Size>
constexpr int rankof_t<Shape<Size ...>> = rankof<Shape<Size ...>>::value;

template <typename ShapeInstance, typename = std::enable_if_t<is_shape<ShapeInstance>::value>>
struct positionsof;

template <int ... Size>
struct positionsof<Shape<Size ...>> { static const int value = (... * Size); };

template <typename ShapeInstance, typename = std::enable_if_t<is_shape<ShapeInstance>::value>>
constexpr int positionsof_t = 0;

template <int ... Size>
constexpr int positionsof_t<Shape<Size ...>> = positionsof<Shape<Size ...>>::value;

}

#endif