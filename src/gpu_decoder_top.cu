/**
    gpu_decoder_top.cu
    Purpose: The main decoding loop that calls GPU kernels

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#include <cuda_runtime.h>
#include "types.h"
#include "utils/helper_cuda.h"

#include "ldpc_kernels.h"
#include "cuda_gpu_wrapper/gpu_setup.h"

template <typename T>
void perform_gpu_ldpc_decoding(dev_mem<T>* d,
                               host_mem<T>* h,
                               const cuda_grid* g,
                               const int n_iterations,
                               int stream_index) {
    copy_to_gpu<T>(h->llr_cuda, d->llr, h->mem_size_llr_cuda, d->streams, stream_index);
    // check_cm<<<1, 1>>>();
    // Perform decoding iterations on GPU
    for (int i_ldpc_itr = 0; i_ldpc_itr < n_iterations; i_ldpc_itr++) {
        ////// Check Node Processing //////
        // 1st iteration is currently a special case.
        if (i_ldpc_itr == 0) {
            cnp_kernel_1st_iter<T><<<g->cnp_blocks, g->cnp_threads, 0, d->streams[stream_index]>>>(
                d->llr[stream_index], d->dt[stream_index], d->r[stream_index]);
        } else {  // Not iteration 1
            cnp_kernel<T><<<g->cnp_blocks, g->cnp_threads, g->shared_r_cache_size,
                            d->streams[stream_index]>>>(d->llr[stream_index], d->dt[stream_index],
                                                        d->r[stream_index]);
        }
        // checkCudaErrors(cudaDeviceSynchronize());

        ////// Variable node processing //////
        if (i_ldpc_itr < n_iterations - 1) {
            vnp_kernel_normal<T><<<g->vnp_blocks, g->vnp_threads, 0, d->streams[stream_index]>>>(
                d->llr[stream_index], d->dt[stream_index]);
        } else {
            vnp_kernel_last_iter<T><<<g->vnp_blocks, g->vnp_threads, 0, d->streams[stream_index]>>>(
                d->llr[stream_index], d->dt[stream_index], d->hard_decision[stream_index]);
            pack_hard_decision<<<g->pack_blocks, g->pack_threads, 0, d->streams[stream_index]>>>(
                d->hard_decision[stream_index], d->hard_decision_packed[stream_index]);
        }
        // checkCudaErrors(cudaDeviceSynchronize());
    }  // for ldpc_iteration
    copy_from_gpu<char>(h->hd_packed_cuda, d->hard_decision_packed, h->mem_size_hd_cuda_packed,
                        d->streams, stream_index);
}
#if CHAR_PRC
template void perform_gpu_ldpc_decoding<char>(dev_mem<char>* d,
                                              host_mem<char>* h,
                                              const cuda_grid* g,
                                              const int n_iterations,
                                              int stream_index);
#elif HALF_PRC
template void perform_gpu_ldpc_decoding<half>(dev_mem<half>* d,
                                              host_mem<half>* h,
                                              const cuda_grid* g,
                                              const int n_iterations,
                                              int stream_index);
#else
template void perform_gpu_ldpc_decoding<float>(dev_mem<float>* d,
                                               host_mem<float>* h,
                                               const cuda_grid* g,
                                               const int n_iterations,
                                               int stream_index);
#endif
