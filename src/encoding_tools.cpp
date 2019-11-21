/**
    encoding_tools.cpp
    Purpose: Nonspecific wireless encoding tools like modulation, awgn

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#include <cmath>
#include "nr_ldpc.h"
#include <cstdio>

void info_gen(int* info_bin, int n_info_bits) {
    // Generate random bits for each
    // TODO: This info bin could be a char. No reason to waste 32 bits to store 1 bit.
    for (int i = 0; i < n_info_bits; i++) {
        info_bin[i] = (rand()) % 2;
    }
}

void modulation(int input[], float output[], int n_symbols) {
    for (int i = 0; i < n_symbols; i++) {
        (input[i] == 0) ? output[i] = 1.0 : output[i] = -1.0;
    }
}

float compute_noise_stddev(const float snr, const float code_rate, const int modulation = 2) {
    constexpr auto signal_power = 1;

    const auto energy_per_bit = signal_power / code_rate;

    const auto eb_n0_db = snr;
    const auto eb_n0 = pow(10.0f, eb_n0_db / 10);
    const float n0 = energy_per_bit / eb_n0;

    const auto noise_variance_sigma_2_awgn = n0 / 2;
    return sqrt(noise_variance_sigma_2_awgn);
}

void awgn(float channel_in[], float channel_out[], int n_symbols, float sigma) {
    float u1, u2, s, noise, randmum, randmum2;
    int i;

    for (i = 0; i < n_symbols; i++) {
        do {
#ifdef _WIN32
            randmum = (float)(rand()) / RAND_MAX;
            randmum2 = (float)(rand()) / RAND_MAX;
#else
            randmum = drand48();
            randmum2 = drand48();
#endif
            u1 = randmum * 2.0f - 1.0f;
            u2 = randmum2 * 2.0f - 1.0f;
            s = u1 * u1 + u2 * u2;
        } while (s >= 1);
        noise = u1 * sqrt((-2.0f * log(s)) / s);

#ifdef NONOISE
        channel_out[i] = trans[i];
#else
        channel_out[i] = channel_in[i] + noise * sigma;
#endif
    }  // for
}

void transmit(int* info_bin,
              int* codeword,
              float* modulate_cwds,
              unsigned long mem_size_codeword,
              float* recv,
              const int n_info_bits,
              const int n_coded_bits,
              const float sigma,
              const int bg_index) {
    // Generating random data, encode, modulate, and add AWGN noise.
    // TODO: Add CRC and 0 padding to stick with original bit length
    info_gen(info_bin, n_info_bits);
    memset(codeword, 0, mem_size_codeword);
    ldpc_encoder_orig(info_bin, codeword, (short)n_info_bits, bg_index, 0);  // From OAI
    modulation(codeword, modulate_cwds, n_coded_bits);
    awgn(modulate_cwds, recv, n_coded_bits, sigma);
}

void calculate_llr(float llr[], float recv[], int n_symbols, float sigma, const int z) {
    int i;
    float llr_rev;

    // Add the llr of 0 for the punctured 2 columns at start.
    const auto two_cols = 2 * z;
    for (i = 0; i < n_symbols; i++) {
        if (i < two_cols) {
            llr[i] = 0;
        } else {
            llr_rev = (recv[i - two_cols] * 2) / (sigma * sigma);  // 2r/sigma^2 ;
            llr[i] = llr_rev;
        }
    }
}

int error_check(float trans[], float recv[], int n_symbols) {
    // We are assuming BPSK
    int cnt = 0;
    for (int i = 0; i < n_symbols; i++) {
        if (recv[i] * trans[i] < 0) {
            cnt++;
        }
    }
    return cnt;
}

error_result cuda_error_check(int info[],
                              char hard_decision[],
                              const int n_total_cws,
                              const int codeword_len,
                              const int info_length) {
    // TODO: Remove dependency on #defines
    error_result this_error;
    this_error.bit_error = 0;
    this_error.frame_error = 0;
    this_error.n_bits = 0;
    this_error.n_frames = 0;

    int bit_error = 0;
    int n_bits_per_cw = 0;
    int frame_error = 0;
    char* hard_decision_t = 0;  // TODO. WHAT IS THIS?
    int* info_t = 0;

    // Loop over each codeword
    for (int i = 0; i < n_total_cws; i++) {
        n_bits_per_cw = 0;
        bit_error = 0;
        hard_decision_t = hard_decision + i * codeword_len;
        info_t = info + i * info_length;

        // Loop over all the information bits
        for (int j = 0; j < info_length; j++) {
            if (info_t[j] != hard_decision_t[j]) {
                bit_error++;
            } else {
                // printf("BAD BIT!\n");
            }
            n_bits_per_cw++;
        }

        if (bit_error != 0) {
            frame_error++;
        }
        this_error.n_frames++;  // Increment frame counter;
        this_error.bit_error += bit_error;
        this_error.n_bits += n_bits_per_cw;
    }
    this_error.frame_error = frame_error;
    return this_error;
}