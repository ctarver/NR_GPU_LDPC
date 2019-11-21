/**
    gpu_decoder_top.h

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#ifndef GPU_DECODER_TOP
#define GPU_DECODER_TOP

#include "types.h"

template <typename T>
void perform_gpu_ldpc_decoding(dev_mem<T>* d,
                               host_mem<T>* h,
                               const cuda_grid* g,
                               int n_iterations,
                               int stream_index);

#endif