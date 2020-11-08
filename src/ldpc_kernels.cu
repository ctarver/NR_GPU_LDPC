/**
    LDPC_kernels.cu
    Purpose: All of the decoding kernels that run on the GPU.

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#include <cuda_runtime.h>
#include "types.h"
#include "utils/helper_cuda.h"

#include "globals.h"

// Create H compact arrays on the device constant memory
// used in CNP Kernel
// TODO: FIND WAY TO GET THESE VALUES FROM THE SETUP LDPC
__device__ __constant__ h_element dev_h_compact1[H_COMPACT1_COL][H_COMPACT1_ROW];
// used in VNP kernel
__device__ __constant__ h_element dev_h_compact2[H_COMPACT2_ROW][H_COMPACT2_COL];

// we can't dynamically allocate to constant so make it worse case
__device__ __constant__ char h_element_count_per_row[BLK_ROW];
__device__ __constant__ char h_element_count_per_col[BLK_COL];

// Add the current params to constant memory
__device__ __constant__ kernel_params p[1];

void allocate_constant_memory(h_element h_compact1[H_COMPACT1_COL][H_COMPACT1_ROW],
                              h_element h_compact2[H_COMPACT2_ROW][H_COMPACT2_COL],
                              const ldpc_params* ldpc,
                              const int mem_size_h_compact1,
                              const int mem_size_h_compact2) {
    // TODO: THIS LIKELY WONT WORK BECAUSE WE ARE DYNAMICALLY ALLOCATING!!!!!!
    checkCudaErrors(cudaMemcpyToSymbol(dev_h_compact1, h_compact1, mem_size_h_compact1));
    checkCudaErrors(cudaMemcpyToSymbol(dev_h_compact2, h_compact2, mem_size_h_compact2));

    const int memory_size_row = ldpc->H_rows * sizeof(char);
    const int memory_size_col = ldpc->H_cols * sizeof(char);

    checkCudaErrors(cudaMemcpyToSymbol(h_element_count_per_row, ldpc->h_element_count_per_row,
                                       memory_size_row));
    checkCudaErrors(cudaMemcpyToSymbol(h_element_count_per_col, ldpc->h_element_count_per_col,
                                       memory_size_col));
}

void put_params_in_constant_mem(const unsigned int z,
                                // Lifting factor
                                const unsigned int n_cw_per_mcw,
                                const unsigned int n_total_vn,
                                const unsigned int n_total_cn,
                                const unsigned int n_blk_col,
                                const unsigned int threads_per_block) {
    const kernel_params p_host = {z,          n_cw_per_mcw, n_total_vn,
                                  n_total_cn, n_blk_col,    threads_per_block};
    checkCudaErrors(cudaMemcpyToSymbol(p, &p_host, sizeof(kernel_params)));
}

// Debugging kernel for checking constant memory on the device
__global__ void check_cm() {
    for (auto i = 0; i < 48; i++) {
        for (auto j = 0; j < 19; j++) {
            h_element h = dev_h_compact1[j][i];
            if (h.value != -1)
                printf("%d,%d: %d\t", h.x, h.y, h.value);
        }
        printf("\n");
    }
}

//////////////// Check Node Processing Kernels ////////////////
template <typename T>
__global__ void cnp_kernel_1st_iter(T* dev_llr, T* dev_dt, T* dev_R) {
    if (threadIdx.x >= p->z) {
        return;
    }

    int iCW = threadIdx.y;  // index of CW in a MCW
    int iMCW = blockIdx.y;  // index of MCW
    int iCurrentCW = iMCW * p->n_cw_per_mcw + iCW;

    // For step 1: update dt   TODO: What is dt?
    int iBlkRow = blockIdx.x;   // block row in h_base
    int iBlkCol;                // block col in h_base. Will update as we sweep a row
    int iSubRow = threadIdx.x;  // row index in sub_block of h_base
    int iCol;                   // overall col index in h_base

    int size_llr_CW = p->n_total_vn;               // size of one llr CW block   TODO: CHECK
                                                   // THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    int size_R_CW = p->n_total_cn * p->n_blk_col;  // size of one R/dt CW block
    int shift_t;

    int offsetR =
        size_R_CW * iCurrentCW + iBlkRow * p->z + iSubRow;  // Offset for message to VN in memory;

    // For 2-min algorithm.
    int Q_sign = 0;
    char sq;

    float Q, Q_abs;
    float R_temp;
    float sign = 1.0f;      // Will be updated with each loop
    float rmin1 = 1000.0f;  // Global min for min sum.
    float rmin2 = 1000.0f;  // 2nd place min for message to actual min.

    char idx_min = 0;

    const auto actual_row = iBlkRow * p->z + iSubRow;
    // printf("In 1st iteration. dev_llr[%d] = %f\n", actual_row, (float) dev_llr[actual_row]);

    h_element h_element_t;
    int n_ones_in_this_row =
        h_element_count_per_row[iBlkRow];  // TODO: Generate this automatically?

    // The 1st recursion. Find the mins and signs.
    for (int i = 0; i < n_ones_in_this_row; i++) {  // loop through all the ZxZ sub-blocks in a row
        h_element_t = dev_h_compact1[i][iBlkRow];   // Current submatrix

        iBlkCol = h_element_t.y;
        shift_t = h_element_t.value;  // Pull out the ZxZ submatrix cyclic shift

        // if (actual_row == 0) {
        //    printf("   i = %d, iBlkCol = %d,  Shift = %d\n", i, iBlkCol, shift_t);
        //}

        shift_t =
            (iSubRow +
             shift_t);  // Where is the 1 in this SubMatrix for this thread of the full H matrix?
        if (shift_t >= p->z)
            shift_t = shift_t - p->z;

        iCol = iBlkCol * p->z + shift_t;  // Which col is the 1?

// Initial messages to CNP are the original LLR.
#if HALF_PRC
        Q = __half2float(dev_llr[size_llr_CW * iCurrentCW + iCol]);  // fetch correspoding VN LLR
#else
        Q = (float)dev_llr[size_llr_CW * iCurrentCW + iCol];  // fetch correspoding VN LLR
#endif  // HALF_PRC

        Q_abs = fabsf(Q);

        sq = Q < 0;  // 0: positive; 1: negative

        // quick version
        sign = sign * (1 - sq * 2);  // Running min_sum sign.
        Q_sign |=
            sq
            << i;  // BitPacked array of all signs for this CN. Assumes fewer than 8 ones in a row.

        if (Q_abs < rmin1) {         // We have a new min. Update!
            rmin2 = rmin1;           // Old winner is stored in rmin2
            rmin1 = Q_abs;           // Store new min here
            idx_min = i;             // Also store the index of the VN message that had it.
        } else if (Q_abs < rmin2) {  // Second place min.
            rmin2 = Q_abs;
        }
    }  // for 1st recurssion
    // Calculate the messages from CN to VN, R
    for (int i = 0; i < n_ones_in_this_row; i++) {
        sq = 1 - 2 * ((Q_sign >> i) & 0x01);  // Extract the sign for this VN.

        // Calculate message to i. 0.75L alpha for scaled minsum.
        R_temp = 0.75f * sign * sq * (i != idx_min ? rmin1 : rmin2);

        // write device
        h_element_t = dev_h_compact1[i][iBlkRow];
        int addr_temp = offsetR + h_element_t.y * p->n_total_cn;  // TODO: Condense this?
#if HALF_PRC == 1
        dev_dt[addr_temp] =
            __float2half(R_temp);  // - R1[i]; // compute the dt value for current llr.
        dev_R[addr_temp] = __float2half(R_temp);  // update R, R=R'.
#elif CHAR_PRC == 1
        dev_dt[addr_temp] = (char)R_temp;  // - R1[i]; // compute the dt value for current llr.
        dev_R[addr_temp] = (char)R_temp;   // update R, R=R'.
#else
        dev_dt[addr_temp] = R_temp;  // - R1[i]; // compute the dt value for current llr.
        dev_R[addr_temp] = R_temp;   // update R, R=R'.
#endif  // PRECISION
    }
}

#if HALF_PRC == 1
template __global__ void cnp_kernel_1st_iter<half>(half*, half*, half*);
#elif CHAR_PRC == 1

template __global__ void cnp_kernel_1st_iter<char>(char*, char*, char*);

#else

template __global__ void cnp_kernel_1st_iter<float>(
    float*,
    float*,
    float*);  // Force the object file to compile this version of the template for linking
#endif

// Kernel_1
template <typename T>
__global__ void cnp_kernel(T* dev_llr, T* dev_dt, T* dev_R) {
    if (threadIdx.x >= p->z)
        return;

    // Define cache for R: Rcache[NON_EMPTY_ELMENT][nThreadPerBlock]
    extern __shared__ float RCache[];
    int iRCacheLine = threadIdx.y * blockDim.x + threadIdx.x;

    int iCW = threadIdx.y;                          // index of CW in a MCW
    int iMCW = blockIdx.y;                          // index of MCW
    int iCurrentCW = iMCW * p->n_cw_per_mcw + iCW;  // index of current codeword

    // For step 1: update dt
    int iBlkRow = blockIdx.x;   // block row in h_base
    int iBlkCol;                // block col in h_base
    int iSubRow = threadIdx.x;  // row index in sub_block of h_base
    int iCol;                   // overall col index in h_base

    int size_llr_CW = p->n_total_vn;               // size of one llr CW block
    int size_R_CW = p->n_total_cn * p->n_blk_col;  // size of one R/dt CW block
    int offsetR = size_R_CW * iCurrentCW + iBlkRow * p->z + iSubRow;

    int shift_t;

    int Q_sign = 0;  // Use int because NR can have more thna 8 VNs connected to 1 CN

    char sq;

    float Q, Q_abs;
    float R_temp;
    float sign = 1.0f;      // Will be updated with each loop
    float rmin1 = 1000.0f;  // Global min for min sum.
    float rmin2 = 1000.0f;  // 2nd place min for message to actual min.

    // For 2-min algorithm.
    char idx_min = 0;

    h_element h_element_t;
    int n_ones_in_this_row = h_element_count_per_row[iBlkRow];

    // The 1st recursion. Find the mins and signs.
    for (int i = 0; i < n_ones_in_this_row;
         i++)  // loop through all the nonzero ZxZ sub-blocks in a row
    {
        h_element_t = dev_h_compact1[i][iBlkRow];
        iBlkCol = h_element_t.y;      // What is the column in h_base
        shift_t = h_element_t.value;  // What is the shift amount applied to this base matrix?

        // Find exact column with 1 for this row
        shift_t = (iSubRow + shift_t);
        if (shift_t >= p->z)
            shift_t = shift_t - p->z;
        iCol = iBlkCol * p->z + shift_t;

// Read the previous message to the VNs and store in shared memory "cache" for whole block
#if HALF_PRC == 1
        R_temp = __half2float(dev_R[offsetR + iBlkCol * p->n_total_cn]);
        Q = __half2float(dev_llr[size_llr_CW * iCurrentCW + iCol]) -
            R_temp;  // Recalculate the message from VN to CN.
#elif CHAR_PRC == 1
        char llr = dev_llr[size_llr_CW * iCurrentCW + iCol];
        R_temp = (float)dev_R[offsetR + iBlkCol * p->n_total_cn];
        if (llr == 127 | llr == -128)
            Q = (float)llr;
        else
            Q = (float)llr - R_temp;
#else
        R_temp = dev_R[offsetR + iBlkCol * p->n_total_cn];
        Q = dev_llr[size_llr_CW * iCurrentCW + iCol] - R_temp;
#endif  // PRECISION
        RCache[i * p->threads_per_block + iRCacheLine] = R_temp;

        // Perform min_sum
        Q_abs = fabsf(Q);

        sq = Q < 0;                  // 0: positive; 1: negative
        sign = sign * (1 - sq * 2);  // Running min_sum sign.
        Q_sign |=
            sq
            << i;  // BitPacked array of all signs for this CN. Assumes fewer than 8 ones in a row.

        if (Q_abs < rmin1) {         // We have a new min. Update!
            rmin2 = rmin1;           // Old winner is stored in rmin2
            rmin1 = Q_abs;           // Store new min here
            idx_min = i;             // Also store the index of the VN message that had it.
        } else if (Q_abs < rmin2) {  // Second place min. Needed when sending a message back to node
                                     // with global min.
            rmin2 = Q_abs;
        }
    }

    // Calculate the messages from CN to VN, R
    for (int i = 0; i < n_ones_in_this_row; i++) {
        sq = 1 - 2 * ((Q_sign >> i) & 0x01);

        // Calculate message to i. 0.75=alpha for scaled minsum.
        R_temp = 0.75f * sign * sq * (i != idx_min ? rmin1 : rmin2);

        // write device
        h_element_t = dev_h_compact1[i][iBlkRow];  // TODO, we accessed these earlier. Could we keep
                                                   // this as a local array?
        int addr_temp = h_element_t.y * p->n_total_cn + offsetR;
#if HALF_PRC == 1
        dev_dt[addr_temp] = __float2half(R_temp - RCache[i * p->threads_per_block + iRCacheLine]);
        dev_R[addr_temp] = __float2half(R_temp);  // update R, R=R'.
#elif CHAR_PRC == 1
        dev_dt[addr_temp] = (char)R_temp - RCache[i * p->threads_per_block + iRCacheLine];
        dev_R[addr_temp] = (char)R_temp;  // update R, R=R'.
#else
        dev_dt[addr_temp] =
            R_temp - RCache[i * p->threads_per_block +
                            iRCacheLine];  // TODO: Check that threadperblock is the correct value.
        dev_R[addr_temp] = R_temp;         // update R, R=R'.
#endif  // HALF_PRC
    }
}

#if HALF_PRC == 1
template __global__ void cnp_kernel<half>(half*, half*, half*);
#elif CHAR_PRC == 1

template __global__ void cnp_kernel<char>(char*, char*, char*);

#else

template __global__ void cnp_kernel<float>(float*, float*, float*);

#endif  // HALF_PRC == 1

// Kernel 2: VNP processing
template <typename T>
__global__ void vnp_kernel_normal(T* dev_llr, T* dev_dt) {
#if MODE == WIFI
    if (threadIdx.x >= p->z)
        return;
#endif

    int iCW = threadIdx.y;                          // index of CW in a MCW
    int iMCW = blockIdx.y;                          // index of MCW
    int iCurrentCW = iMCW * p->n_cw_per_mcw + iCW;  // index of current codeword

    int iBlkCol = blockIdx.x;
    int iBlkRow;
    int iSubCol = threadIdx.x;  // Each thread works on a seperate column in the submatrix
    int iRow;
    int iCol = iBlkCol * p->z + iSubCol;  // Index of exact column in full H matrix

    int size_llr_CW = p->n_total_vn;               // size of one llr CW block
    int size_R_CW = p->n_total_cn * p->n_blk_col;  // size of one R/dt CW block

    int shift_t, sf;
    int llr_index = size_llr_CW * iCurrentCW + iCol;
#if HALF_PRC == 1
    float APP = __half2float(dev_llr[llr_index]);  // Need the original LLR for this VN
#else
    float APP = (float)dev_llr[llr_index];  // Need the original LLR for this VN
#endif  // HALF_PRC == 1

    h_element h_element_t;

    int offsetDt =
        size_R_CW * iCurrentCW + iBlkCol * p->n_total_cn;  // Offset for accessing the change in Rs.

    // Loop over all ones in the column
    for (int i = 0; i < h_element_count_per_col[iBlkCol]; i++) {
        h_element_t = dev_h_compact2[i][iBlkCol];

        shift_t = h_element_t.value;  // Shift amount for this submatrix with a one.
        iBlkRow = h_element_t.x;      // Row index of this submatrix.

        // Calculate exact index of the current row we are in for full H matrix
        sf = iSubCol - shift_t;
        if (sf < 0)
            sf = sf + p->z;
        iRow = iBlkRow * p->z + sf;

// Add the previous LLR and all the deltaRs for each incoming message
#if HALF_PRC == 1
        APP += __half2float(dev_dt[offsetDt + iRow]);
#else
        APP += (float)dev_dt[offsetDt + iRow];
#endif  // HALF_PRC == 1
    }
// Write back to device memory
#if HALF_PRC == 1
    dev_llr[llr_index] = __float2half(APP);
#elif CHAR_PRC == 1
    if (APP > 127)
        APP = 127;
    else if (APP < -128)
        APP = -128;
    dev_llr[llr_index] = (char)APP;
#else
    dev_llr[llr_index] = APP;
#endif  // PRECISION == 1
    // No hard decision for non-last iteration.
}

#if HALF_PRC == 1
template __global__ void vnp_kernel_normal<half>(half*, half*);
#elif CHAR_PRC == 1

template __global__ void vnp_kernel_normal<char>(char*, char*);

#else

template __global__ void vnp_kernel_normal<float>(float*, float*);

#endif

// TODO. Try passing a hard decision flag to the kernel.
template <typename T>
__global__ void vnp_kernel_last_iter(T* dev_llr, T* dev_dt, char* dev_hd) {
#if MODE == WIFI
    if (threadIdx.x >= p->z)
        return;
#endif

    int iCW = threadIdx.y;  // index of CW in a MCW
    int iMCW = blockIdx.y;  // index of MCW
    int iCurrentCW = iMCW * p->n_cw_per_mcw + iCW;

    int iBlkCol = blockIdx.x;
    int iBlkRow;
    int iSubCol = threadIdx.x;
    int iRow;
    int iCol = iBlkCol * p->z + iSubCol;

    int size_llr_CW = p->n_total_vn;               // size of one llr CW block
    int size_R_CW = p->n_total_cn * p->n_blk_col;  // size of one R/dt CW block

    int shift_t, sf;
    int llr_index = size_llr_CW * iCurrentCW + iCol;
#if HALF_PRC == 1
    float APP = __half2float(dev_llr[llr_index]);
#else
    float APP = dev_llr[llr_index];
#endif  // HALF_PRC == 1

    h_element h_element_t;

    int offsetDt = size_R_CW * iCurrentCW + iBlkCol * p->n_total_cn;

    for (int i = 0; i < h_element_count_per_col[iBlkCol]; i++) {
        h_element_t = dev_h_compact2[i][iBlkCol];

        shift_t = h_element_t.value;
        iBlkRow = h_element_t.x;

        sf = iSubCol - shift_t;
        if (sf < 0)
            sf = sf + p->z;

        iRow = iBlkRow * p->z + sf;
#if HALF_PRC == 1
        APP += __half2float(dev_dt[offsetDt + iRow]);
#else
        APP += (float)dev_dt[offsetDt + iRow];
#endif  // HALF_PRC == 1
    }

    // hard decision
    // for the last iteration, we don't need to write back memory. instead, we directly
    // perform hard decision.
    if (APP > 0) {
        dev_hd[llr_index] = 0;
    } else {
        dev_hd[llr_index] = 1;
    }
}

#if HALF_PRC == 1
template __global__ void vnp_kernel_last_iter<half>(half*, half*, char*);
#elif CHAR_PRC == 1

template __global__ void vnp_kernel_last_iter<char>(char*, char*, char*);

#else

template __global__ void vnp_kernel_last_iter<float>(float*, float*, char*);

#endif

__global__ void pack_hard_decision(char* hd_bit, char* hd_packed) {
    const int iCW = threadIdx.y;  // index of CW in a MCW
    const int iMCW = blockIdx.y;  // index of MCW
    const int iCurrentCW = iMCW * p->n_cw_per_mcw + iCW;

    const int iBlkCol = blockIdx.x;
    const int iSubCol = threadIdx.x * 8;
    const int iCol = iBlkCol * p->z + iSubCol;

    const int size_llr_CW = p->n_total_vn;  // size of one llr CW block TODO: CHECK!!!!!!!!!!!!

    const int llr_index = size_llr_CW * iCurrentCW + iCol;

    char temp = 0;
    for (int i = 0; i < 8; i++) {
        temp |= hd_bit[llr_index + i] << i;
    }
    hd_packed[llr_index / 8] = temp;
}
