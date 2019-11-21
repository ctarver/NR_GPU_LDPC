/**
    setup_exp.cu
    Purpose: Setup host and device memory systems

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#include <stdlib.h>
#include "types.h"
#include "cuda_gpu_wrapper/gpu_setup.h"
#include "host_helpers.h"
#include "encoding_tools.h"
#include "ldpc_kernels.h"
#include "globals.h"
#include <cassert>

template <typename T>
T* host_alloc(const unsigned int mem_size_llr) {
    const auto scaling_factor = 4 / sizeof(T);
    return (T*)malloc(mem_size_llr / scaling_factor);
}

// Default will be to do nothing. Char and half are overloads
template <typename T>
void reduce_precision(const float* full_prc_array, T* quant_array, const int n_elements) {
    for (auto i = 0; i < n_elements; i++) {
        quant_array[i] = full_prc_array[i];
    }
}

template <>
void reduce_precision<half>(const float* full_prc_array, half* quant_array, const int n_elements) {
    for (auto i = 0; i < n_elements; i++) {
        quant_array[i] = __float2half_rn(full_prc_array[i]);
    }
};

template <>
void reduce_precision<char>(const float* full_prc_array, char* quant_array, const int n_elements) {
    for (auto i = 0; i < n_elements; i++) {
        // TODO. Don't hardcode the sigma in this quantize fx
        quant_array[i] = quantize(0.5, full_prc_array[i], 8);
    }
}

template <typename T>
host_mem<T> setup_host(const params* p,
                       const unsigned int n_info_bits,
                       const unsigned int n_cw_bits) {
    // Calculate size of data to be encoded, coded data, and their log-liklihood ratios
    const auto mem_size_infobits = n_info_bits * sizeof(int);
    const auto mem_size_cw = n_cw_bits * sizeof(int);
    const auto mem_size_llr = n_cw_bits * sizeof(float);

    // Allocate HOST MEMORY for each.
    int* info_bin = (int*)malloc(mem_size_infobits);  // Uncoded information bits
    int* cwds = (int*)malloc(mem_size_cw);            // Encoded bits
    float* modulated_cwds = (float*)malloc(mem_size_llr);
    float* llr = (float*)malloc(mem_size_llr);
    float* recv = (float*)malloc(mem_size_llr);
    T* llr_quantized = host_alloc<T>(mem_size_llr);

    // HOST ARRAYS THAT ARE STAGING BEFORE GOING ONTO GPU STREAMS
    // Using Pinned Memory to setup staging array for transfers to and from device.
    const int mem_size_infobits_cuda = p->n_mcw * p->n_cw_per_mcw * n_info_bits * sizeof(int);
    const int mem_size_llr_cuda = p->n_mcw * p->n_cw_per_mcw * n_cw_bits * sizeof(T);
    const int mem_size_hd_cuda = p->n_mcw * p->n_cw_per_mcw * n_cw_bits * sizeof(char);

    // TODO: Change hard_decision to only pass final bits!
    const int mem_size_hd_packed_cuda = mem_size_hd_cuda / 8;

    int** info_bin_cuda = allocate_cuda_placeholder<int>(p->n_streams, mem_size_infobits_cuda);
    T** llr_cuda = allocate_cuda_placeholder<T>(p->n_streams, mem_size_llr_cuda);
    char** hd_cuda = allocate_cuda_placeholder<char>(p->n_streams, mem_size_hd_cuda);
    char** hd_packed_cuda = allocate_cuda_placeholder<char>(p->n_streams, mem_size_hd_packed_cuda);

    const host_mem<T> h = {info_bin,       mem_size_infobits,
                           cwds,           modulated_cwds,
                           mem_size_cw,    llr,
                           mem_size_llr,   recv,
                           llr_quantized,  info_bin_cuda,
                           llr_cuda,       mem_size_llr_cuda,
                           hd_cuda,        mem_size_hd_cuda,
                           hd_packed_cuda, mem_size_hd_packed_cuda};
    return h;
}

template host_mem<char> setup_host<char>(const params* p,
                                         unsigned int n_info_bits,
                                         unsigned int n_cw_bits);

template host_mem<half> setup_host<half>(const params* p,
                                         unsigned int n_info_bits,
                                         unsigned int n_cw_bits);

template host_mem<float> setup_host<float>(const params* p,
                                           unsigned int n_info_bits,
                                           unsigned int n_cw_bits);

template <typename T>
dev_mem<T> setup_dev_memory(const params* p, const ldpc_params* ldpc, const host_mem<T>* host) {
    cudaStream_t* streams = setup_streams(p->n_streams);

    // Create Pointers to Device Memory
    const int mem_size_dt_cuda =
        p->n_mcw * p->n_cw_per_mcw * ldpc->H_rows * ldpc->Z * ldpc->H_cols * sizeof(T);
    const int mem_size_r_cuda =
        p->n_mcw * p->n_cw_per_mcw * ldpc->H_rows * ldpc->Z * ldpc->H_cols * sizeof(T);
    T** dev_llr = allocate_device_array<T>(host->mem_size_llr_cuda, p->n_streams);
    T** dev_dt = allocate_device_array<T>(mem_size_dt_cuda, p->n_streams);
    T** dev_r = allocate_device_array<T>(mem_size_r_cuda, p->n_streams);
    char** dev_hd = allocate_device_array<char>(host->mem_size_hd_cuda, p->n_streams);
    char** dev_hd_packed = allocate_device_array<char>(host->mem_size_hd_cuda_packed, p->n_streams);

    const dev_mem<T> dev = {streams, dev_llr, dev_dt, dev_r, dev_hd, dev_hd_packed};
    return dev;
}

template dev_mem<char> setup_dev_memory<char>(const params* p,
                                              const ldpc_params* ldpc,
                                              const host_mem<char>* host);
template dev_mem<half> setup_dev_memory<half>(const params* p,
                                              const ldpc_params* ldpc,
                                              const host_mem<half>* host);
template dev_mem<float> setup_dev_memory<float>(const params* p,
                                                const ldpc_params* ldpc,
                                                const host_mem<float>* host);

cuda_grid setup_gpu_grid(const ldpc_params* ldpc, const int n_mcw, const int n_cw_per_mcw) {
    //////////////// Setup CUDA kernel grid dimensions ////////////////
    // CNP Kernel Dimensions
    constexpr int n_threads_per_block_mod = 32;
    const int block_size_x = (ldpc->Z + n_threads_per_block_mod - 1) / n_threads_per_block_mod *
                             n_threads_per_block_mod;  // Forces a multiple of 32 threads per block
    dim3 cnp_blocks(ldpc->H_rows, n_mcw, 1);           // y = index of MCW
    dim3 cnp_threads(block_size_x, n_cw_per_mcw, 1);   // 127, 2 .   y = index of CW in a MCW
    const auto cnp_threads_per_block = block_size_x * n_cw_per_mcw;
    const auto shared_r_cache_size = cnp_threads_per_block * ldpc->n_nz_in_row * sizeof(float);

    // VNP Kernel Dimensions
    dim3 vnp_blocks(ldpc->H_cols, n_mcw, 1);
    dim3 vnp_threads(block_size_x, n_cw_per_mcw, 1);
    const auto vnp_threads_per_block = block_size_x * n_cw_per_mcw;

    // Packing Kernel Dimensions
    dim3 pack_blocks(ldpc->H_cols, n_mcw, 1);
    dim3 pack_threads(block_size_x / 8, n_cw_per_mcw, 1);

    // Early Termination Kernel Dimensions
    dim3 early_blocks(n_mcw, n_cw_per_mcw, 1);
    dim3 early_threads(ldpc->Z, ldpc->H_rows, 1);

    const cuda_grid grid = {cnp_blocks,          cnp_threads,          vnp_blocks,   vnp_threads,
                            pack_blocks,         pack_threads,         early_blocks, early_threads,
                            shared_r_cache_size, cnp_threads_per_block};
    return grid;
}

template <typename T>
void generate_new_codewords(const params* p,
                            host_mem<T>* h,
                            const ldpc_params* l,
                            const int codewords_in_iteration,
                            const float sigma) {
    // Will generate new codewords and load them onto streams on host.

    // Loop to generate codewords for each stream.
    for (int i = 0; i < codewords_in_iteration; i++) {
        // Generating random data, encode, modulate, and add AWGN noise.
        transmit(h->info_bin, h->cwds, h->modulated_cwds, h->mem_size_cw, h->recv, l->K,
                 l->N_before_puncture, sigma, l->bg_index);

        // Uncoded BER
        int n_errors = error_check(h->modulated_cwds, h->recv, l->N_before_puncture);
        calculate_llr(h->llr, h->recv, l->N_before_puncture, sigma, l->Z);

        // IT SEEMS TO BE NOT USING MY OVERLOADS
        reduce_precision<T>(h->llr, h->llr_quan, l->N_before_puncture);

        // Divide up our data into various streams
        // We are just decoding the same thing on multiple streams :/ MEH
        copy_to_cuda_staged_memory<int>(h->info_bin, h->info_bin_cuda, h->mem_size_infobits,
                                        i * l->K, p->n_streams);
        copy_to_cuda_staged_memory<T>(h->llr_quan, h->llr_cuda, h->mem_size_llr_cuda,
                                      i * l->N_before_puncture, p->n_streams);
    }
}
template void generate_new_codewords<char>(const params* p,
                                           host_mem<char>* h,
                                           const ldpc_params* l,
                                           int codewords_in_iteration,
                                           float sigma);
template void generate_new_codewords<half>(const params* p,
                                           host_mem<half>* h,
                                           const ldpc_params* l,
                                           int codewords_in_iteration,
                                           float sigma);
template void generate_new_codewords<float>(const params* p,
                                            host_mem<float>* h,
                                            const ldpc_params* l,
                                            int codewords_in_iteration,
                                            float sigma);

h_element** create_compact_matrix(const int first_dim, const int second_dim) {
    h_element** h_compact1 = new h_element*[first_dim];
    for (int i = 0; i < first_dim; i++) {
        h_compact1[i] = new h_element[second_dim];
    }
    return h_compact1;
}

void fill_in_compact_matrices(h_element h_compact1[H_COMPACT1_COL][H_COMPACT1_ROW],
                              h_element h_compact2[H_COMPACT2_ROW][H_COMPACT2_COL],
                              const ldpc_params* ldpc) {
    // TODO: - Make this accept an arbitrary 2D Array so generateCompactParityMatrix_1
    //         and generateCompactParityMatrix_2 can be combined.

    // init the compact matrix
    const int n_rows = ldpc->H_rows;
    const int h_compact1_col = ldpc->n_nz_in_row;
    for (int i = 0; i < h_compact1_col; i++) {
        for (int j = 0; j < n_rows; j++) {
            h_compact1[i][j] = {0, 0, -1, 0};  // h[i][0-11], the same column
        }
    }

    // scan the h matrix, and generate compact mode of h
    //   walk across rows in h_base. If there is a non "-1" entry, record it in a column
    //   of h_compact
    int* h_base = ldpc->H;
    const int n_cols = ldpc->H_cols;
    for (char i = 0; i < n_rows; i++) {      // Walks across h_base row
        int k = 0;                           // row of h_compact
        for (char j = 0; j < n_cols; j++) {  // Walks across columns of h_base...
            const int address = i * n_cols + j;
            if (h_base[address] != -1) {
                h_compact1[k][i] = {i, j, (short)h_base[address], 1};
                k++;
            }
        }
    }

    // init the compact matrix
    const int h_compact2_row = ldpc->n_nz_in_col;
    for (int i = 0; i < h_compact2_row;
         i++) {  // TODO. Why does this need to be 12? Max # of 1s in a column is 6
        for (int j = 0; j < n_cols; j++) {
            h_compact2[i][j] = {0, 0, -1, 0};
        }
    }

    for (char j = 0; j < n_cols; j++) {
        int k = 0;
        for (char i = 0; i < n_rows; i++) {
            const int address = i * n_cols + j;
            if (h_base[address] != -1) {
                // although h is transposed, the (x,y) is still (iBlkRow, iBlkCol)
                h_compact2[k][j] = {i, j, (short)h_base[address], 1};
                k++;
            }
        }
    }

// PRINT OUT EACH FOR DEBUGGING PURPOSES
#ifndef NDEBUG
    printf("H COMPACT 1:\n");
    for (auto i = 0; i < 15; i++) {
        for (auto j = 0; j < 15; j++) {
            const auto test = h_compact1[i][j];
            printf("%d, %d: %d \t", i, j, test.value);
        }
        printf("\n");
    }
#endif
}

void put_h_compact_and_params_in_constant_memory(const ldpc_params* ldpc,
                                                 const params* p,
                                                 const cuda_grid* g) {
    // Create and Fill the 2 compact matrices based on H

    /*  // Dynamic allocation doesn't play well with GPU!
    h_element** h_compact1 = create_compact_matrix(ldpc->n_nz_in_row, ldpc->H_rows);
    h_element** h_compact2 = create_compact_matrix(ldpc->n_nz_in_col, ldpc->H_cols);
    */

    assert(ldpc->n_nz_in_row == H_COMPACT1_COL);
    assert(ldpc->H_rows == H_COMPACT1_ROW);
    assert(ldpc->n_nz_in_col == H_COMPACT2_ROW);
    assert(ldpc->H_cols == H_COMPACT2_COL);
    h_element h_compact1[H_COMPACT1_COL][H_COMPACT1_ROW];
    h_element h_compact2[H_COMPACT2_ROW][H_COMPACT2_COL];

    fill_in_compact_matrices(h_compact1, h_compact2, ldpc);

    // Calculate size of the compact matrices.
    const int memorySize_h_compact1 = ldpc->H_rows * ldpc->n_nz_in_row * sizeof(h_element);
    const int memorySize_h_compact2 = ldpc->H_cols * ldpc->n_nz_in_col * sizeof(h_element);
    allocate_constant_memory(h_compact1, h_compact2, ldpc, memorySize_h_compact1,
                             memorySize_h_compact2);

    put_params_in_constant_mem(ldpc->Z, p->n_cw_per_mcw, ldpc->N_before_puncture,
                               ldpc->H_rows * ldpc->Z, ldpc->H_cols, g->cnp_threads_per_block);
}