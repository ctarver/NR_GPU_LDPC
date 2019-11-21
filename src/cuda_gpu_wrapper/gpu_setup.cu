/**
    gpu_setup.cpp
    Purpose: General, intuitive wrapper of anything I man need in CUDA

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#include <cuda_runtime.h>
#include "utils/helper_cuda.h"
#include <string>
#include <sstream>
#include <cuda_fp16.h>


cudaStream_t* setup_streams(const unsigned int n_streams) {
    cudaStream_t* streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));
    for (int i = 0; i < n_streams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }
    return streams;
}

template <typename T>
T** allocate_cuda_placeholder(const unsigned int n_streams, const unsigned int n_bytes) {
    // Allocates placeholders for host arrays before copying to device. This will be page locked and
    // be 2D where the 1st dim represents streams.
    T** new_array = new T*[n_streams];
    for (int i = 0; i < n_streams; i++) {
        new_array[i] = (T*)malloc(n_bytes);
        checkCudaErrors(cudaHostAlloc((void**)&new_array[i], n_bytes, cudaHostAllocDefault));
    }
    return new_array;
}

template char** allocate_cuda_placeholder<char>(unsigned int n_streams,
                                                unsigned int n_bytes);
template float** allocate_cuda_placeholder<float>(unsigned int n_streams,
                                                  unsigned int n_bytes);
template int** allocate_cuda_placeholder<int>(unsigned int n_streams,
                                              unsigned int n_bytes);
template half** allocate_cuda_placeholder<half>(unsigned int n_streams, unsigned int n_bytes);

template <typename T>
void copy_to_cuda_staged_memory(T* host_array, T** cuda_array, const int n_bytes, const int offset, const int n_streams) {
    for (auto i_stream = 0; i_stream < n_streams; i_stream++) {
        memcpy(cuda_array[i_stream] + offset, host_array, n_bytes);
    }
}

template void copy_to_cuda_staged_memory<int>(int*, int**, int, int, int);

template void copy_to_cuda_staged_memory<char>(char*, char**, int, int, int);

template void copy_to_cuda_staged_memory<half>(half*, half**, int, int, int);

template void copy_to_cuda_staged_memory<float>(float*, float**, int, int, int);

template <typename T>
T** allocate_device_array(const unsigned int n_bytes_per_stream, const unsigned int n_streams) {
    T** new_array = new T*[n_streams];
    for (auto i = 0; i < n_streams; i++) {
        checkCudaErrors(cudaMalloc((void**)&new_array[i], n_bytes_per_stream));
    }
    return new_array;
}

template char** allocate_device_array<char>(unsigned int n_bytes_per_stream,
                                            unsigned int n_streams);
template half** allocate_device_array<half>(unsigned int n_bytes_per_stream,
                                            unsigned int n_streams);
template float** allocate_device_array<float>(unsigned int n_bytes_per_stream,
                                            unsigned int n_streams);

template <typename T>
void copy_to_gpu(T** host_array,
                 T** device_array,
                 const unsigned int n_bytes,
                 cudaStream_t* streams,
                 const unsigned int stream_index) {
    checkCudaErrors(cudaMemcpyAsync(device_array[stream_index], host_array[stream_index], n_bytes,
                                    cudaMemcpyHostToDevice, streams[stream_index]));
}

template void copy_to_gpu<char>(char**, char**, unsigned int, cudaStream_t*, unsigned int);
template void copy_to_gpu<half>(half**, half**, unsigned int, cudaStream_t*, unsigned int);
template void copy_to_gpu<float>(float**, float**, unsigned int, cudaStream_t*, unsigned int);

template <typename T>
void copy_from_gpu(T** host_array,
                   T** device_array,
                   const unsigned int n_bytes,
                   cudaStream_t* streams,
                   const unsigned int stream_index) {
    checkCudaErrors(cudaMemcpyAsync(host_array[stream_index], device_array[stream_index], n_bytes,
                                    cudaMemcpyDeviceToHost, streams[stream_index]));
}
template void copy_from_gpu<char>(char** host_array,
                                  char** device_array,
                                  unsigned int n_bytes,
                                  cudaStream_t* streams,
                                  unsigned int stream_index);
template void copy_from_gpu<half>(half** host_array,
                                  half** device_array,
                                  unsigned int n_bytes,
                                  cudaStream_t* streams,
                                  unsigned int stream_index);
template void copy_from_gpu<float>(float** host_array,
                                   float** device_array,
                                  unsigned int n_bytes,
                                  cudaStream_t* streams,
                                  unsigned int stream_index);

std::string print_devices() {
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf(" No devices supporting CUDA.");
    }

    cudaDeviceProp deviceProperty;
    const int currentDeviceID = 0;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProperty, currentDeviceID));

    std::stringstream buffer;
    buffer << "Device Name:" << deviceProperty.name << "\n";
    buffer << "    Device Major           :" << deviceProperty.major << "\n";
    buffer << "    Device Minor           :" << deviceProperty.minor << "\n";
    buffer << "    Clock Rate             :" << deviceProperty.clockRate << "\n";
    buffer << "    SM Count               :" << deviceProperty.multiProcessorCount << "\n";
    buffer << "    Total Global Memory    :" << deviceProperty.totalGlobalMem << "\n";
    buffer << "    Total Constant Memory  :" << deviceProperty.totalConstMem << "\n";
    buffer << "    Shared Memory per Block:" << deviceProperty.sharedMemPerBlock << "\n";
    buffer << "    Max Threads per Block  :" << deviceProperty.maxThreadsPerBlock << "\n";
    buffer << "    Registers per Block    :" << deviceProperty.regsPerBlock << "\n";
    buffer << "    Warp Size              :" << deviceProperty.warpSize << "\n";
    buffer << "    Memory Pitch           :" << deviceProperty.memPitch << "\n";
    buffer << "    maxGridSize[0],[1],[2]  :" << deviceProperty.maxGridSize[0] << ","
           << deviceProperty.maxGridSize[1] << "," << deviceProperty.maxGridSize[2] << "\n";
    buffer << "    maxThreadsDim[0],[1],[2]:" << deviceProperty.maxThreadsDim[0] << ","
           << deviceProperty.maxThreadsDim[1] << "," << deviceProperty.maxThreadsDim[2] << "\n";
    buffer << "    textureAlignment      :" << deviceProperty.textureAlignment << "\n";
    buffer << "    deviceOverlap         :" << deviceProperty.deviceOverlap << "\n";
    buffer << "    zero-copy data transfers:" << deviceProperty.canMapHostMemory << "\n";

    std::string to_return;
    to_return = buffer.str();
    return to_return;
}
