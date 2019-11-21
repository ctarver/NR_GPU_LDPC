/**
    ldpc_kernels.h

    @author Chance Tarver
    @version 0.1 11/18/19
*/
#ifndef KERNELS
#define KERNELS

// Create h compcat arrays on the device
#include "types.h"
#include "globals.h"
__global__ void check_cm();

void allocate_constant_memory(h_element h_compact1[H_COMPACT1_COL][H_COMPACT1_ROW],
                              h_element h_compact2[H_COMPACT2_ROW][H_COMPACT2_COL],
                              const ldpc_params* ldpc,
                              int memorySize_h_compact1,
                              int memorySize_h_compact2);

void put_params_in_constant_mem(unsigned int z,
                                // Lifting factor
                                unsigned int n_cw_per_mcw,
                                unsigned int n_total_vn,
                                unsigned int n_total_cn,
                                unsigned int n_blk_col,
                                unsigned int threads_per_block);

template <typename T>
__global__ void cnp_kernel_1st_iter(T*, T*, T*);

template <typename T>
__global__ void cnp_kernel(T*, T*, T*);

template <typename T>
__global__ void vnp_kernel_normal(T*, T*);

template <typename T>
__global__ void vnp_kernel_last_iter(T*, T*, char*);

__global__ void pack_hard_decision(char*, char*);

#endif