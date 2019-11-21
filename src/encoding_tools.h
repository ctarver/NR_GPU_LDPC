/**
    encoding_tools.h
    Purpose: Nonspecific wireless encoding tools like modulation, awgn

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#ifndef ENCODING_TOOLS
#define ENCODING_TOOLS

float compute_noise_stddev(float snr, float code_rate, int modulation = 2);
void transmit(int* info_bin,
              int* codeword,
              float* modulate_cwds,
              unsigned long mem_size_codeword,
              float* recv,
              int n_info_bits,
              int n_coded_bits,
              float sigma,
              int bg_index);
void calculate_llr(float llr[], float recv[], int n_symbols, float sigma, int z);
int error_check(float trans[], float recv[], int n_symbols);

error_result cuda_error_check(int info[],
                              char hard_decision[],
                              int n_total_cws,
                              int codeword_len,
                              int info_length);

#endif
