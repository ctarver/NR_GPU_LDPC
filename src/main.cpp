/**
    main.cpp
    Purpose: Main starting place of our testbench for the 5G LDPC GPU

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#define LOG_LEVEL debug
#define TYPE char  // ALSO YOU need to change the thing in globals.h

#include "spdlog/spdlog.h"

#include "types.h"
#include "host_helpers.h"
#include "nr_ldpc.h"
#include "encoding_tools.h"
#include "setup_exp.h"
#include "utils/timer.h"
#include "gpu_decoder_top.h"
#include "ldpc_kernels.h"

void run_exp(const ldpc_params* ldpc,
             host_mem<TYPE>* h,
             dev_mem<TYPE>* d,
             const cuda_grid* g,
             params* p);

gpu_result run_test_on_batch(int current_total_cws,
                             int cws_in_iteration,
                             float sigma,
                             host_mem<TYPE>* h,
                             dev_mem<TYPE>* d,
                             const cuda_grid* g,
                             const ldpc_params* ldpc,
                             params* p);

template <typename T>
float test_latency(dev_mem<T>* d, host_mem<T>* h, const cuda_grid* g, int n_iterations);

template <typename T>
error_result measure_ber(const host_mem<T>* h,
                         int n_streams,
                         int n_cw_per_mcw,
                         int n_mcw,
                         int codeword_len,
                         int info_len);

template <typename T>
float test_throughput(dev_mem<T>* d,
                      host_mem<T>* h,
                      const cuda_grid* g,
                      int n_iterations,
                      int n_streams);

int main() {
    // Simulation Params
    constexpr auto snr_min = -5;
    constexpr auto snr_max = 20;
    constexpr auto snr_step = 5;
    constexpr auto n_codewords = 5000;
    constexpr auto n_info_bits =
        8424;  //  8424; MAX in 1 segment is 8424. Multiple segs not supported yet
    constexpr auto code_rate =
        1.0F / 3.0F;  // I don't have rate matching implemented yet. Stick with 1/3
    constexpr auto n_cw_per_mcw = 2;
    constexpr auto n_mcw = 20;
    constexpr auto n_streams = 6;
    constexpr auto n_ldpc_iterations = 5;
    params p = {snr_min,   snr_max,      snr_step, n_codewords, n_info_bits,
                code_rate, n_cw_per_mcw, n_mcw,    n_streams};

    long seed = 69012;
    srand(seed);

    setup_logs();
    spdlog::set_level(spdlog::level::LOG_LEVEL);
    auto logger = spdlog::get("RESULTS");
    print_log_header(p);

    const ldpc_params ldpc = setup_ldpc(p.n_info_bits, p.code_rate, n_ldpc_iterations);
    host_mem<TYPE> host = setup_host<TYPE>(&p, ldpc.K, ldpc.N_before_puncture);
    dev_mem<TYPE> dev = setup_dev_memory<TYPE>(&p, &ldpc, &host);
    const cuda_grid grid = setup_gpu_grid(&ldpc, p.n_cw_per_mcw, p.n_mcw);
    put_h_compact_and_params_in_constant_memory(&ldpc, &p, &grid);

    run_exp(&ldpc, &host, &dev, &grid, &p);
    logger->info("EXPERIMENT OVER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    return 0;
}

void run_exp(const ldpc_params* ldpc,
             host_mem<TYPE>* h,
             dev_mem<TYPE>* d,
             const cuda_grid* g,
             params* p) {
    auto logger = spdlog::get("RESULTS");

    //////////////// Main SNR Loop ////////////////
    for (auto snr = p->snr_min; snr <= p->snr_max; snr += p->snr_step) {
        logger->info("Starting SNR = {}", snr);
        const auto sigma = compute_noise_stddev(snr, ldpc->rate);

        gpu_result r = {100, 0, 0};

        const auto cws_in_iteration = p->n_cw_per_mcw * p->n_mcw;
        for (auto n_cws_completed = 0; n_cws_completed < p->n_codewords;
             n_cws_completed += cws_in_iteration) {
            r = r + run_test_on_batch(n_cws_completed, cws_in_iteration, sigma, h, d, g, ldpc, p);
        }
        const auto ber = (float)r.n_bit_errors / (float)r.n_bits_total;
        const auto fer = (float)r.n_frame_errors / (float)r.n_frames;
        const auto code_bit_throughput = r.n_bits_total / r.cpu_run_time / 1000;
        const auto user_throughput = code_bit_throughput * ldpc->rate;
        const auto codeword_rate = r.n_frames / r.cpu_run_time;
        logger->info("\t BER = \t\t\t{}", ber);
        logger->info("\t FER = \t\t\t{}", fer);
        logger->info("\t Best Latency = \t{} us", r.best_latency * 1000);
        logger->info("\t User Throughput = \t{} Mbps", user_throughput);
        logger->info("\t Codeword Throughput = \t{} Mbps", code_bit_throughput);
    }
}

gpu_result run_test_on_batch(const int current_total_cws,
                             const int cws_in_iteration,
                             const float sigma,
                             host_mem<TYPE>* h,
                             dev_mem<TYPE>* d,
                             const cuda_grid* g,
                             const ldpc_params* ldpc,
                             params* p) {
    auto logger = spdlog::get("RESULTS");
    logger->trace("Starting new batch of codewords. Currently completed {} codewords",
                  current_total_cws);

    generate_new_codewords(p, h, ldpc, cws_in_iteration, sigma);
    const auto this_latency = test_latency<TYPE>(d, h, g, ldpc->n_iterations);
    const auto cpu_time = test_throughput<TYPE>(d, h, g, ldpc->n_iterations, p->n_streams);
    const error_result this_error = measure_ber<TYPE>(h, p->n_streams, p->n_cw_per_mcw, p->n_mcw,
                                                      ldpc->N_before_puncture, ldpc->K);
    assert(this_error.n_frames == (cws_in_iteration * p->n_streams));
    return {this_latency,        cpu_time,
            this_error.n_bits,   this_error.bit_error,
            this_error.n_frames, this_error.frame_error};
}

template <typename T>
float test_latency(dev_mem<T>* d, host_mem<T>* h, const cuda_grid* g, const int n_iterations) {
    constexpr auto stream_to_use = 0;
    Timer cpu_timer;
    cpu_timer.start();
    perform_gpu_ldpc_decoding<T>(d, h, g, n_iterations, stream_to_use);
    cudaDeviceSynchronize();
    const auto latency = cpu_timer.stop_get();
    return latency;
}

template <typename T>
float test_throughput(dev_mem<T>* d,
                      host_mem<T>* h,
                      const cuda_grid* g,
                      const int n_iterations,
                      const int n_streams) {
    Timer cpu_timer;
    cpu_timer.start();
    for (int i_stream = 0; i_stream < n_streams; i_stream++) {
        perform_gpu_ldpc_decoding<T>(d, h, g, n_iterations, i_stream);
    }
    cudaDeviceSynchronize();
    return cpu_timer.stop_get();
}

template <typename T>
error_result measure_ber(const host_mem<T>* h,
                         const int n_streams,
                         const int n_cw_per_mcw,
                         const int n_mcw,
                         const int codeword_len,
                         const int info_len) {
    error_result e = {0, 0, 0, 0};
    for (int i_stream = 0; i_stream < n_streams; i_stream++) {
        unpack_char(h->hd_packed_cuda[i_stream], h->hd_cuda[i_stream], h->mem_size_hd_cuda);

        e = e + cuda_error_check(h->info_bin_cuda[i_stream], h->hd_cuda[i_stream],
                                 n_cw_per_mcw * n_mcw, codeword_len, info_len);
    }
    return e;
}