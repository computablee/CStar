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
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (BlockSize * 2) + tid;
    unsigned int gridSize = BlockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < Size)
    {
        sdata[tid] += idata[i] + idata[i + BlockSize];
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

template <int Size, typename T>
class InstantiatedShape;

template <int Size, typename T>
T& operator+=(T& lhs, InstantiatedShape<Size, T>& rhs);

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

    template <int S, typename U>
    friend U& operator+=(U& lhs, InstantiatedShape<S, U>& rhs);
};

template <int Size, typename T>
T& operator+=(T& lhs, InstantiatedShape<Size, T>& rhs)
{
    constexpr int blockSize = 128;

    T* reductionArray;
    cudaMalloc((void**)&reductionArray, sizeof(T) * blockSize);
    __reduce<Size, T, blockSize><<<128, blockSize>>>(rhs.data, reductionArray);
    cudaDeviceSynchronize();

    T hostArray[blockSize];
    cudaMemcpy(hostArray, reductionArray, sizeof(T) * blockSize, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < blockSize; i++)
        lhs += hostArray[i];

    return lhs;
}

template <int Size>
struct Shape
{
    template <typename T>
    using shape = InstantiatedShape<Size, T>;
};