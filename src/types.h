/**
    types.h
    Purpose: The various types we use in the project to make returns easier and the overall code
   easier.

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#ifndef TYPES_H
#define TYPES_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

struct error_result {
    int bit_error;
    int n_bits;
    int frame_error;
    int n_frames;
};
inline error_result operator+(const error_result& a, const error_result& b) {
    const int bit_error = a.bit_error + b.bit_error;
    const int n_bits = a.n_bits + b.n_bits;
    const int frame_error = a.frame_error + b.frame_error;
    const int n_frames = a.n_frames + b.n_frames;
    const error_result answer = {bit_error, n_bits, frame_error, n_frames};
    return answer;
}

struct h_element {
    char x;
    char y;
    short value;
    char valid;
};

struct ldpc_params {
    int A;                          // payload size before CRC
    float rate;                     // Code rate
    int bg_index;                   // index of the NR basegraph
    int Z;                          // Lifting factor
    int K;                          // Number of bits per LDPC code block to be encoded
    int N;                          // Number of encoded bits per LDPC code block.
    int H_rows;                     // Number of rows in the base graph
    int H_cols;                     // Number of cols in the base graph.
    int* H;                         // Pointer to the parity check matrix, H.
    int n_nz_in_row;                // Max number of nonzero entries in a row of H
    int n_nz_in_col;                // Max number of nonzero entries in a col of H
    char* h_element_count_per_row;  // An array with all the number of nonzero entires in each row
                                    // of H
    char* h_element_count_per_col;
    int n_iterations;
    int N_before_puncture;  // Adds 2*Z to N
};

struct params {
    float snr_min;
    float snr_max;
    float snr_step;
    int n_codewords;
    int n_info_bits;
    float code_rate;
    int n_cw_per_mcw;
    int n_mcw;
    int n_streams;
};

template <typename T>
struct host_mem {
    // Non streamed host arrays.
    int* info_bin;
    unsigned long mem_size_infobits;
    int* cwds;
    float* modulated_cwds;
    unsigned long mem_size_cw;
    float* llr;
    unsigned long mem_size_llr;
    float* recv;
    T* llr_quan;

    // Streamed host arrays. Staged for transfer to GPU
    int** info_bin_cuda;
    T** llr_cuda;
    int mem_size_llr_cuda;
    char** hd_cuda;
    int mem_size_hd_cuda;
    char** hd_packed_cuda;
    int mem_size_hd_cuda_packed;
};

template <typename T>
struct dev_mem {
    cudaStream_t* streams;
    T** llr;
    T** dt;
    T** r;
    char** hard_decision;
    char** hard_decision_packed;
};

struct gpu_result {
    float best_latency;
    float cpu_run_time;
    int n_bits_total;
    int n_bit_errors;
    int n_frames;
    int n_frame_errors;
};
inline gpu_result operator+(const gpu_result& a, const gpu_result& b) {
    const float best_latency = (a.best_latency < b.best_latency) ? a.best_latency : b.best_latency;
    const float new_run_time = a.cpu_run_time + b.cpu_run_time;
    const float n_bits = a.n_bits_total + b.n_bits_total;
    const int n_bit_errors = a.n_bit_errors + b.n_bit_errors;
    const int n_frames = a.n_frames + b.n_frames;
    const int n_frame_errors = a.n_frame_errors + b.n_frame_errors;
    const gpu_result answer = {best_latency, new_run_time, n_bits,
                               n_bit_errors, n_frames,     n_frame_errors};
    return answer;
}

struct cuda_grid {
    dim3 cnp_blocks;
    dim3 cnp_threads;
    dim3 vnp_blocks;
    dim3 vnp_threads;
    dim3 pack_blocks;
    dim3 pack_threads;
    dim3 early_blocks;
    dim3 early_threads;
    unsigned long shared_r_cache_size;
    unsigned int cnp_threads_per_block;
};

struct kernel_params {
    unsigned int z;  // Lifting factor
    unsigned int n_cw_per_mcw;
    unsigned int n_total_vn;
    unsigned int n_total_cn;
    unsigned int n_blk_col;
    unsigned int threads_per_block;
};

#endif