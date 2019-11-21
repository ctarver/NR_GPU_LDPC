/**
    gpu_setup.h
    Purpose: Header for the general, intuitive wrapper of anything I man need in CUDA

    @author Chance Tarver
    @version 0.1 11/18/19
*/
#ifndef GPU_SETUP
#define GPU_SETUP

#include <string>

cudaStream_t* setup_streams(unsigned int n_streams);

template <typename T>
T** allocate_cuda_placeholder(unsigned int n_streams, unsigned int n_bytes);

template <typename T>
void copy_to_cuda_staged_memory(T* host_array,
                                T** cuda_array,
                                int n_byes,
                                int offset,
                                int n_streams);

template <typename T>
T** allocate_device_array(unsigned int n_bytes_per_stream, unsigned int n_streams);

template <typename T>
void copy_to_gpu(T** host_array,
                 T** device_array,
                 unsigned int n_bytes,
                 cudaStream_t* streams,
                 unsigned int stream_index);

template <typename T>
void copy_from_gpu(T** host_array,
                   T** device_array,
                   unsigned int n_bytes,
                   cudaStream_t* streams,
                   unsigned int stream_index);

std::string print_devices();

#endif  // !GPU_SETUP