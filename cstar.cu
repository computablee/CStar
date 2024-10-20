#include <cstddef>
#include <cuda.h>
#include <iostream>

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

// This reduction code is provided courtesy of Nvidia's slides
// See: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <typename T, unsigned int BlockSize>
__device__ void __warpReduce(volatile T * __restrict__ sdata, unsigned int tid)
{
    if constexpr (BlockSize >= 64) sdata[tid] += sdata[tid + 32];
    if constexpr (BlockSize >= 32) sdata[tid] += sdata[tid + 16];
    if constexpr (BlockSize >= 16) sdata[tid] += sdata[tid + 8];
    if constexpr (BlockSize >=  8) sdata[tid] += sdata[tid + 4];
    if constexpr (BlockSize >=  4) sdata[tid] += sdata[tid + 2];
    if constexpr (BlockSize >=  2) sdata[tid] += sdata[tid + 1];
}

template <int Size, typename T, unsigned int BlockSize>
__global__ void __reduce(T * __restrict__ idata, T * __restrict__ odata)
{
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (BlockSize * 2) + tid;
    unsigned int gridSize = BlockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i + BlockSize < Size)
    {
        sdata[tid] += idata[i] + idata[i + BlockSize];
        i += gridSize;
    }

    while (i < Size)
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

    if (tid < 32) __warpReduce<T, BlockSize>(sdata, tid);
    if (tid == 0) odata[blockIdx.x] = sdata[0];
}

template <typename T, int ... Size>
class InstantiatedShape;

template <typename T, int ... Size>
T& operator+=(T& lhs, InstantiatedShape<T, Size ...>& rhs);

template <typename T, int ... Size>
class InstantiatedShape
{
private:
    T* data;
    
    template <typename... Idx>
    size_t compute_index(Idx... idxs) const
    {
        static_assert(sizeof...(Size) == sizeof...(idxs), "Number of indices must match the number of dimensions.");
        static_assert((... && std::is_same<Idx, int>::value));
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
    };

public:
    InstantiatedShape()
    {
        cudaMalloc((void**)&this->data, sizeof(T) * (... * Size));
    }

    ~InstantiatedShape()
    {
        cudaFree(this->data);
    }

    InstantiatedShape& operator=(T scalar)
    {
        __scalar_assign<(... * Size), T><<<128, 128>>>(this->data, scalar);
        cudaDeviceSynchronize();
        return *this;
    }

    InstantiatedShape& operator=(const InstantiatedShape<T, Size ...>& rhs)
    {
        __vector_assign<(... * Size), T><<<128, 128>>>(this->data, rhs.data);
        cudaDeviceSynchronize();
        return *this;
    }

    InstantiatedShape& operator+=(T scalar)
    {
        __scalar_add<(... * Size), T><<<128, 128>>>(this->data, scalar);
        cudaDeviceSynchronize();
        return *this;
    }

    InstantiatedShape& operator+=(const InstantiatedShape<T, Size ...>& rhs)
    {
        __vector_add<(... * Size), T><<<128, 128>>>(this->data, rhs.data);
        cudaDeviceSynchronize();
        return *this;
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
    friend U& operator+=(U& lhs, InstantiatedShape<U, S ...>& rhs);
};

template <typename T, int ... Size>
T& operator+=(T& lhs, InstantiatedShape<T, Size ...>& rhs)
{
    constexpr int blockSize = 128;

    T* reductionArray;
    cudaMalloc((void**)&reductionArray, sizeof(T) * blockSize);
    __reduce<(... * Size), T, blockSize><<<128, blockSize>>>(rhs.data, reductionArray);
    cudaDeviceSynchronize();

    T hostArray[blockSize];
    cudaMemcpy(hostArray, reductionArray, sizeof(T) * blockSize, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < blockSize; i++)
        lhs += hostArray[i];

    return lhs;
}

template <int ... Size>
struct Shape
{
    template <typename T>
    using shape = InstantiatedShape<T, Size ...>;
};