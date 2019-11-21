/**
    nr_ldpc.h
    Purpose: Everything related to LDPC for NR

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#ifndef NR_LDPC
#define NR_LDPC

#include "types.h"

ldpc_params setup_ldpc(int n_bits, float code_rate, int n_iterations);

int ldpc_encoder_orig(int* test_input,
                      int* channel_input,
                      short block_length,
                      short BG,
                      unsigned char gen_code);

#endif