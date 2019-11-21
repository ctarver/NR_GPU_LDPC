/**
    host_helpers.h

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#ifndef HOST_HELPERS
#define HOST_HELPERS

#include "types.h"

void setup_logs();

void print_log_header(params exp);

signed char quantize(double D, double x, unsigned char B);

void unpack_char(char* packed_input, char* unpacked_output, int size);

#endif  // !HOST_HELPERS
