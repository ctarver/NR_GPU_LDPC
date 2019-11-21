/**
    host_helpers.cpp
    Purpose: Random hot functions that aren't directly related to ldpc.

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#include "spdlog/spdlog.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_sinks.h"

#include "host_helpers.h"
#include "cuda_gpu_wrapper/gpu_setup.h"

void setup_logs() {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    sinks.push_back(
        std::make_shared<spdlog::sinks::daily_file_sink_st>("logs/logfile.log", 23, 59));
    // create synchronous  loggers
    auto ldpc_log = std::make_shared<spdlog::logger>("LDPC", begin(sinks), end(sinks));
    auto hw_logger = std::make_shared<spdlog::logger>("GPU", begin(sinks), end(sinks));
    auto results_logger = std::make_shared<spdlog::logger>("RESULTS", begin(sinks), end(sinks));
    auto debug_logger = std::make_shared<spdlog::logger>("DEBUG", begin(sinks), end(sinks));
    spdlog::register_logger(ldpc_log);
    spdlog::register_logger(hw_logger);
    spdlog::register_logger(results_logger);
    spdlog::register_logger(debug_logger);
}

void print_log_header(params p) {
    auto ldpc_log = spdlog::get("LDPC");
    ldpc_log->info("NEW SIMULATION:");
    ldpc_log->info("   Testing from SNR {} to {} and stepping by {}", p.snr_min, p.snr_max,
                   p.snr_step);
    ldpc_log->info("   We will do {} codewords using {} cws per mcw with {} mcw per transfer",
                   p.n_codewords, p.n_cw_per_mcw, p.n_mcw);
    ldpc_log->info("   Information Bits in Code: {}", p.n_info_bits);
    ldpc_log->info("   Code rate               : {}", p.code_rate);

    auto gpu_log = spdlog::get("GPU");
    gpu_log->info("{}", print_devices());
}

signed char quantize(const double D, const double x, const unsigned char B) {
    double qxd = floor(x / D);
    const short maxlev = 1 << (B - 1);  //(char)(pow(2,B-1));

    // printf("x=%f,qxd=%f,maxlev=%d\n",x,qxd, maxlev);

    if (qxd <= -maxlev)
        qxd = -maxlev;
    else if (qxd >= maxlev)
        qxd = maxlev - 1;

    return ((signed char)qxd);
}

void unpack_char(char* packed_input, char* unpacked_output, const int size) {
    for (int i = 0; i < size; i++) {
        unpacked_output[i] = (packed_input[i / 8] & (1 << (i & 7))) >> (i & 7);
    }
}