/**
    setup_exp.h
    Purpose: Setup host and device memory systems

    @author Chance Tarver
    @version 0.1 11/18/19
*/
#ifndef SETUP_EXP
#define SETUP_EXP

#include "types.h"

template <typename T>
host_mem<T> setup_host(const params* p, unsigned int n_info_bits, unsigned int n_cw_bits);

template <typename T>
dev_mem<T> setup_dev_memory(const params* p, const ldpc_params* ldpc, const host_mem<T>* host);

cuda_grid setup_gpu_grid(const ldpc_params* ldpc, int n_mcw, int n_cw_per_mcw);

template <typename T>
void generate_new_codewords(const params* p,
                            host_mem<T>* h,
                            const ldpc_params* l,
                            int codewords_in_iteration,
                            float sigma);

void put_h_compact_and_params_in_constant_memory(const ldpc_params* ldpc,
                                                 const params* p,
                                                 const cuda_grid* g);

#endif
