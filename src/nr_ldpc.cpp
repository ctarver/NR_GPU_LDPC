/**
    nr_ldpc.cpp
    Purpose: Everything related to LDPC for NR. The bottom functions are from OAI.

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#include "types.h"
#include "spdlog/spdlog-inl.h"

#include "h_matrices/h_bg_1_i_0.h"
#include "h_matrices/h_bg_1_i_1.h"
#include "h_matrices/h_bg_1_i_2.h"
#include "h_matrices/h_bg_1_i_3.h"
#include "h_matrices/h_bg_1_i_4.h"
#include "h_matrices/h_bg_1_i_5.h"
#include "h_matrices/h_bg_1_i_6.h"
#include "h_matrices/h_bg_1_i_7.h"
#include "h_matrices/h_bg_2_i_0.h"
#include "h_matrices/h_bg_2_i_1.h"
#include "h_matrices/h_bg_2_i_2.h"
#include "h_matrices/h_bg_2_i_3.h"
#include "h_matrices/h_bg_2_i_4.h"
#include "h_matrices/h_bg_2_i_5.h"
#include "h_matrices/h_bg_2_i_6.h"
#include "h_matrices/h_bg_2_i_7.h"

#include "Gen_shift_value.h"

ldpc_params setup_ldpc(const int A, const float rate, const int n_iterations) {
    // setup_ldpc. See TS 38.212

    auto ldpc_log = spdlog::get("LDPC");

    // Calculate number of bits in transport block after CRC.
    const int l = (A > 3824) ? 24 : 16;
    const int B = A + l;  // Size before segmentation.
    ldpc_log->info("Adding additional {} bits of CRC. New size is {}", l, B);

    // Choose base graph.
    int bg_index;  // Which base graph we use. In set {0, 1}.
    int K_cb;      // Max code block size for a given base graph.
    int K_b;
    if ((A <= 292) | (A < 3824 & rate <= 0.67) | (rate <= 0.25)) {
        bg_index = 2;
        K_cb = 3840;
        if (B > 640) {
            K_b = 10;
        } else if (B > 560) {
            K_b = 9;
        } else if (B > 192) {
            K_b = 8;
        } else {
            K_b = 6;
        }
    } else {
        bg_index = 1;
        K_cb = 8448;
        K_b = 22;
    }
    ldpc_log->info("Using BG {}", bg_index);

    // Calculate number of segments. 5.2.2
    int C;            // Number of code blocks.
    int B_prime;      // Number of total bits after segmentation (segmentation adds additional CRCs)
    if (B <= K_cb) {  // Less than max size. Only 1 code block.
        C = 1;
        B_prime = B;
    } else {
        const int L = 24;  // This L is for code blocks. Not the same as earlier L.
        C = ceil(B / (K_cb - L));
        B_prime = B + C * L;
    }

    const int K_prime = B_prime / C;  // Number of bits per code block.
    ldpc_log->info("Number of code blocks is {} with {} bits per each", C, K_prime);
    if (C > 1) {
        ldpc_log->critical(
            "Number of code blocks is Greater than 1! I haven't added support for Segmentation!!!");
    }

    // Find min value of Z in all sets of lifting sizes in Table 5.3.2-1, denoted as Z_c , such that
    // K_b * Z_c > K'
    const int Z_min = K_prime / K_b;
    int Z_c = 1000;  // Invalid value. We just need it to be high so we catch the first Z below this
                     // and above Z_min
    int i_ls;

    // Construct the whole table 5.3.2-1
    int base_z[] = {2, 3, 5, 7, 9, 11, 13, 15};  // first element in each row of the table
    int set_index[8][8];
    for (int i = 0; i < 8; i++) {
        int temp = base_z[i];
        // Fill out a row in the set index table
        for (int i_z = 0; i_z < 8; i_z++) {
            set_index[i][i_z] = (temp < 385) ? temp : 0;
            temp = temp * 2;

            // Check to see if this is the minimum one so far.
            if (set_index[i][i_z] >= Z_min & set_index[i][i_z] < Z_c) {
                Z_c = set_index[i][i_z];
                i_ls = i;
            }
        }
    }
    ldpc_log->info("Set Index is {}", i_ls);
    ldpc_log->info("Using a lifting factor Z of {}", Z_c);

    // K = 22Z for LDPC base graph 1 and x K = 10Z for LDPC base graph 2;
    const int K = (bg_index == 1) ? 22 * Z_c : 10 * Z_c;
    const int N = (bg_index == 1) ? 66 * Z_c : 50 * Z_c;
    // Notice above is 2*Z_c less than then number of H_cols.

    const int H_rows = (bg_index == 1) ? 46 : 42;
    const int H_cols = (bg_index == 1) ? 68 : 52;
    ldpc_log->info("H_base is {} by {} ", H_rows, H_cols);
    ldpc_log->info("Final K after 0 padding to fit lift factor options: {}", K);
    ldpc_log->info("Final N : {}", N);

    // Copy everything that matters to ldpc struct
    ldpc_params ldpc = {A, rate, bg_index, Z_c, K, N, H_rows, H_cols};

    if (bg_index == 1) {
        switch (i_ls) {
            case 0:
                ldpc.H = &h_base_1_i0[0];
                break;
            case 1:
                ldpc.H = &h_base_1_i1[0];
                break;
            case 2:
                ldpc.H = &h_base_1_i2[0];
                break;
            case 3:
                ldpc.H = &h_base_1_i3[0];
                break;
            case 4:
                ldpc.H = &h_base_1_i4[0];
                break;
            case 5:
                ldpc.H = &h_base_1_i5[0];
                break;
            case 6:
                ldpc.H = &h_base_1_i6[0];
                break;
            case 7:
                ldpc.H = &h_base_1_i7[0];
                break;
            default:
                ldpc.H = nullptr;
                ldpc_log->critical("NOT VALID SET INDEX!! {}", i_ls);
        }
    } else if (bg_index == 2) {
        switch (i_ls) {
            case 0:
                ldpc.H = &h_base_2_i0[0];
                break;
            case 1:
                ldpc.H = &h_base_2_i1[0];
                break;
            case 2:
                ldpc.H = &h_base_2_i2[0];
                break;
            case 3:
                ldpc.H = &h_base_2_i3[0];
                break;
            case 4:
                ldpc.H = &h_base_2_i4[0];
                break;
            case 5:
                ldpc.H = &h_base_2_i5[0];
                break;
            case 6:
                ldpc.H = &h_base_2_i6[0];
                break;
            case 7:
                ldpc.H = &h_base_2_i7[0];
                break;
            default:
                ldpc.H = nullptr;
                ldpc_log->critical("NOT VALID SET INDEX!! {}", i_ls);
        }
    } else {
        ldpc_log->critical("NOT VALID BG Index!! BG = {}", bg_index);
    }

    // Variables that will go in GPU constant memory eventually.
    // For each row or col in H, it says how many non-zero elements exists.
    char* nz_h_per_row = new char[H_rows];
    char* nz_h_per_col = new char[H_cols];

    // Count the number of nonzero entries in H_base
    int max_in_a_row = 0;
    int max_in_a_col = 0;
    for (int i = 0; i < H_rows; i++) {
        int row_temp_count = 0;
        for (int j = 0; j < H_cols; j++) {  // Count a row
            const int address = i * H_cols + j;
            if (ldpc.H[address] != -1) {
                row_temp_count++;
            }
        }
        nz_h_per_row[i] = row_temp_count;
        if (row_temp_count > max_in_a_row)
            max_in_a_row = row_temp_count;
    }
    for (int j = 0; j < H_cols; j++) {
        int col_temp_count = 0;
        for (int i = 0; i < H_rows; i++) {  // Count a row
            const int address = i * H_cols + j;
            if (ldpc.H[address] != -1) {
                col_temp_count++;
            }
        }
        nz_h_per_col[j] = col_temp_count;
        if (col_temp_count > max_in_a_col) {
            max_in_a_col = col_temp_count;
        }
    }

    ldpc.n_nz_in_row = max_in_a_row;
    ldpc.n_nz_in_col = max_in_a_col;

    ldpc.h_element_count_per_row = nz_h_per_row;
    ldpc.h_element_count_per_col = nz_h_per_col;
    ldpc.n_iterations = n_iterations;
    ldpc.N_before_puncture = N + 2 * Z_c;
    return ldpc;
}

// EVERYTHING BELOW IS STOLEN FROM OAI!

short* choose_generator_matrix(short BG, short Zc) {
    short* Gen_shift_values = NULL;

    if (BG == 1) {
        switch (Zc) {
            case 2:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_2;
                break;

            case 3:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_3;
                break;

            case 4:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_4;
                break;

            case 5:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_5;
                break;

            case 6:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_6;
                break;

            case 7:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_7;
                break;

            case 8:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_8;
                break;

            case 9:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_9;
                break;

            case 10:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_10;
                break;

            case 11:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_11;
                break;

            case 12:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_12;
                break;

            case 13:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_13;
                break;

            case 14:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_14;
                break;

            case 15:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_15;
                break;

            case 16:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_16;
                break;

            case 18:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_18;
                break;

            case 20:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_20;
                break;

            case 22:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_22;
                break;

            case 24:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_24;
                break;

            case 26:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_26;
                break;

            case 28:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_28;
                break;

            case 30:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_30;
                break;

            case 32:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_32;
                break;

            case 36:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_36;
                break;

            case 40:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_40;
                break;

            case 44:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_44;
                break;

            case 48:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_48;
                break;

            case 52:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_52;
                break;

            case 56:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_56;
                break;

            case 60:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_60;
                break;

            case 64:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_64;
                break;

            case 72:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_72;
                break;

            case 80:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_80;
                break;

            case 88:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_88;
                break;

            case 96:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_96;
                break;

            case 104:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_104;
                break;

            case 112:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_112;
                break;

            case 120:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_120;
                break;

            case 128:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_128;
                break;

            case 144:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_144;
                break;

            case 160:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_160;
                break;

            case 176:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_176;
                break;

            case 192:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_192;
                break;

            case 208:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_208;
                break;

            case 224:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_224;
                break;

            case 240:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_240;
                break;

            case 256:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_256;
                break;

            case 288:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_288;
                break;

            case 320:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_320;
                break;

            case 352:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_352;
                break;

            case 384:
                Gen_shift_values = (short*)Gen_shift_values_BG1_Z_384;
                break;
        }
    } else if (BG == 2) {
        switch (Zc) {
            case 2:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_2;
                break;

            case 3:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_3;
                break;

            case 4:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_4;
                break;

            case 5:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_5;
                break;

            case 6:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_6;
                break;

            case 7:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_7;
                break;

            case 8:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_8;
                break;

            case 9:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_9;
                break;

            case 10:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_10;
                break;

            case 11:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_11;
                break;

            case 12:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_12;
                break;

            case 13:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_13;
                break;

            case 14:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_14;
                break;

            case 15:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_15;
                break;

            case 16:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_16;
                break;

            case 18:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_18;
                break;

            case 20:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_20;
                break;

            case 22:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_22;
                break;

            case 24:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_24;
                break;

            case 26:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_26;
                break;

            case 28:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_28;
                break;

            case 30:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_30;
                break;

            case 32:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_32;
                break;

            case 36:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_36;
                break;

            case 40:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_40;
                break;

            case 44:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_44;
                break;

            case 48:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_48;
                break;

            case 52:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_52;
                break;

            case 56:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_56;
                break;

            case 60:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_60;
                break;

            case 64:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_64;
                break;

            case 72:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_72;
                break;

            case 80:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_80;
                break;

            case 88:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_88;
                break;

            case 96:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_96;
                break;

            case 104:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_104;
                break;

            case 112:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_112;
                break;

            case 120:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_120;
                break;

            case 128:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_128;
                break;

            case 144:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_144;
                break;

            case 160:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_160;
                break;

            case 176:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_176;
                break;

            case 192:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_192;
                break;

            case 208:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_208;
                break;

            case 224:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_224;
                break;

            case 240:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_240;
                break;

            case 256:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_256;
                break;

            case 288:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_288;
                break;

            case 320:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_320;
                break;

            case 352:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_352;
                break;

            case 384:
                Gen_shift_values = (short*)Gen_shift_values_BG2_Z_384;
                break;
        }
    }

    return Gen_shift_values;
}

// LDPC Encoder from OAI
int ldpc_encoder_orig(int* test_input,
                      int* channel_input,
                      short block_length,
                      short BG,
                      unsigned char gen_code) {
    unsigned char c[22 * 384];  // padded input, unpacked, max size
    unsigned char d[68 * 384];  // coded output, unpacked, max size
    unsigned char channel_temp, temp;
    short *Gen_shift_values, *no_shift_values, *pointer_shift_values;
    short Zc;
    // initialize for BG == 1
    short Kb = 22;
    short nrows = 46;  // parity check bits
    short ncols = 22;  // info bits

    int i, i1, i2, i3, i4, i5, temp_prime, var;
    int no_punctured_columns, removed_bit;
    // Table of possible lifting sizes
    short lift_size[51] = {2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
                           15,  16,  18,  20,  22,  24,  26,  28,  30,  32,  36,  40,  44,
                           48,  52,  56,  60,  64,  72,  80,  88,  96,  104, 112, 120, 128,
                           144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384};

    int nind = 0;
    int indlist[1000];
    int indlist2[1000];

    // determine number of bits in codeword
    // if (block_length>3840)
    if (BG == 1) {
        // BG=1;
        Kb = 22;
        nrows = 46;  // parity check bits
        ncols = 22;  // info bits
    }
    // else if (block_length<=3840)
    else if (BG == 2) {
        // BG=2;
        nrows = 42;  // parity check bits
        ncols = 10;  // info bits

        if (block_length > 640)
            Kb = 10;
        else if (block_length > 560)
            Kb = 9;
        else if (block_length > 192)
            Kb = 8;
        else
            Kb = 6;
    }

    // find minimum value in all sets of lifting size
    Zc = 0;
    for (i1 = 0; i1 < 51; i1++) {
        if (lift_size[i1] >= (double)block_length / Kb) {
            Zc = lift_size[i1];
            // printf("%d\n",Zc);
            break;
        }
    }
    if (Zc == 0) {
        printf("ldpc_encoder_orig: could not determine lifting size\n");
        return (-1);
    }

    Gen_shift_values = choose_generator_matrix(BG, Zc);
    if (Gen_shift_values == NULL) {
        printf("ldpc_encoder_orig: could not find generator matrix\n");
        return (-1);
    }

    // printf("ldpc_encoder_orig: BG %d, Zc %d, Kb %d\n",BG, Zc, Kb);

    // load base graph of generator matrix
    if (BG == 1) {
        no_shift_values = (short*)no_shift_values_BG1;
        pointer_shift_values = (short*)pointer_shift_values_BG1;
    } else if (BG == 2) {
        no_shift_values = (short*)no_shift_values_BG2;
        pointer_shift_values = (short*)pointer_shift_values_BG2;
    } else {
    }

    no_punctured_columns = (int)((nrows - 2) * Zc + block_length - block_length * 3) / Zc;
    removed_bit = (nrows - no_punctured_columns - 2) * Zc + block_length - (block_length * 3);
    // printf("%d\n",no_punctured_columns);
    // printf("%d\n",removed_bit);
    // unpack input
    memset(c, 0, sizeof(unsigned char) * ncols * Zc);
    memset(d, 0, sizeof(unsigned char) * nrows * Zc);

    for (i = 0; i < block_length; i++) {
        c[i] = (char)test_input[i];
    }

    // parity check part

    if (gen_code == 1) {
        char fname[100];
        sprintf(fname, "ldpc_BG%d_Zc%d_byte.c", BG, Zc);
        FILE* fd = fopen(fname, "w");
        // AssertFatal(fd != NULL, "cannot open %s\n", fname);
        sprintf(fname, "ldpc_BG%d_Zc%d_16bit.c", BG, Zc);
        FILE* fd2 = fopen(fname, "w");
        // AssertFatal(fd2 != NULL, "cannot open %s\n", fname);

        int shift;
        char data_type[100];
        char xor_command[100];
        int mask;

        fprintf(fd, "#include \"PHY/sse_intrin.h\"\n");
        fprintf(fd2, "#include \"PHY/sse_intrin.h\"\n");

        if ((Zc & 31) == 0) {
            shift = 5;  // AVX2 - 256-bit SIMD
            mask = 31;
            strcpy(data_type, "__m256i");
            strcpy(xor_command, "_mm256_xor_si256");
        } else if ((Zc & 15) == 0) {
            shift = 4;  // SSE4 - 128-bit SIMD
            mask = 15;
            strcpy(data_type, "__m128i");
            strcpy(xor_command, "_mm_xor_si128");

        } else if ((Zc & 7) == 0) {
            shift = 3;  // MMX  - 64-bit SIMD
            mask = 7;
            strcpy(data_type, "__m64");
            strcpy(xor_command, "_mm_xor_si64");
        } else {
            shift = 0;  // no SIMD
            mask = 0;
            strcpy(data_type, "uint8_t");
            strcpy(xor_command, "scalar_xor");
            fprintf(fd, "#define scalar_xor(a,b) ((a)^(b))\n");
            fprintf(fd2, "#define scalar_xor(a,b) ((a)^(b))\n");
        }
        fprintf(fd, "// generated code for Zc=%d, byte encoding\n", Zc);
        fprintf(fd2, "// generated code for Zc=%d, 16bit encoding\n", Zc);
        fprintf(fd, "static inline void ldpc_BG%d_Zc%d_byte(uint8_t *c,uint8_t *d) {\n", BG, Zc);
        fprintf(fd2, "static inline void ldpc_BG%d_Zc%d_16bit(uint16_t *c,uint16_t *d) {\n", BG,
                Zc);
        fprintf(fd, "  %s *csimd=(%s *)c,*dsimd=(%s *)d;\n\n", data_type, data_type, data_type);
        fprintf(fd2, "  %s *csimd=(%s *)c,*dsimd=(%s *)d;\n\n", data_type, data_type, data_type);
        fprintf(fd, "  %s *c2,*d2;\n\n", data_type);
        fprintf(fd2, "  %s *c2,*d2;\n\n", data_type);
        fprintf(fd, "  int i2;\n");
        fprintf(fd2, "  int i2;\n");
        fprintf(fd, "  for (i2=0; i2<%d; i2++) {\n", Zc >> shift);
        fprintf(fd2, "  for (i2=0; i2<%d; i2++) {\n", Zc >> (shift - 1));
        for (i2 = 0; i2 < 1; i2++) {
            // t=Kb*Zc+i2;

            // calculate each row in base graph

            fprintf(fd, "     c2=&csimd[i2];\n");
            fprintf(fd, "     d2=&dsimd[i2];\n");
            fprintf(fd2, "     c2=&csimd[i2];\n");
            fprintf(fd2, "     d2=&dsimd[i2];\n");

            for (i1 = 0; i1 < nrows; i1++) {
                channel_temp = 0;
                fprintf(fd, "\n//row: %d\n", i1);
                fprintf(fd2, "\n//row: %d\n", i1);
                fprintf(fd, "     d2[%d]=", (Zc * i1) >> shift);
                fprintf(fd2, "     d2[%d]=", (Zc * i1) >> (shift - 1));

                nind = 0;

                for (i3 = 0; i3 < ncols; i3++) {
                    temp_prime = i1 * ncols + i3;

                    for (i4 = 0; i4 < no_shift_values[temp_prime]; i4++) {
                        var = (int)((i3 * Zc +
                                     (Gen_shift_values[pointer_shift_values[temp_prime] + i4] + 1) %
                                         Zc) /
                                    Zc);
                        int index =
                            var * 2 * Zc +
                            (i3 * Zc +
                             (Gen_shift_values[pointer_shift_values[temp_prime] + i4] + 1) % Zc) %
                                Zc;

                        indlist[nind] =
                            ((index & mask) * ((2 * Zc) >> shift) * Kb) + (index >> shift);
                        indlist2[nind++] =
                            ((index & (mask >> 1)) * ((2 * Zc) >> (shift - 1)) * Kb) +
                            (index >> (shift - 1));
                    }
                }
                for (i4 = 0; i4 < nind - 1; i4++) {
                    fprintf(fd, "%s(c2[%d],", xor_command, indlist[i4]);
                    fprintf(fd2, "%s(c2[%d],", xor_command, indlist2[i4]);
                }
                fprintf(fd, "c2[%d]", indlist[i4]);
                fprintf(fd2, "c2[%d]", indlist2[i4]);
                for (i4 = 0; i4 < nind - 1; i4++) {
                    fprintf(fd, ")");
                    fprintf(fd2, ")");
                }
                fprintf(fd, ";\n");
                fprintf(fd2, ";\n");
            }
            fprintf(fd, "  }\n}\n");
            fprintf(fd2, "  }\n}\n");
        }
        fclose(fd);
        fclose(fd2);
    } else if (gen_code == 0) {
        for (i2 = 0; i2 < Zc; i2++) {
            // t=Kb*Zc+i2;

            // rotate matrix here
            for (i5 = 0; i5 < Kb; i5++) {
                temp = c[i5 * Zc];
                memmove(&c[i5 * Zc], &c[i5 * Zc + 1], (Zc - 1) * sizeof(unsigned char));
                c[i5 * Zc + Zc - 1] = temp;
            }

            // calculate each row in base graph
            for (i1 = 0; i1 < nrows - no_punctured_columns; i1++) {
                channel_temp = 0;

                for (i3 = 0; i3 < Kb; i3++) {
                    temp_prime = i1 * ncols + i3;

                    for (i4 = 0; i4 < no_shift_values[temp_prime]; i4++) {
                        channel_temp =
                            channel_temp ^
                            c[i3 * Zc + Gen_shift_values[pointer_shift_values[temp_prime] + i4]];
                    }
                }
                d[i2 + i1 * Zc] = channel_temp;
            }
        }
    }
    // CHANCE ADDED THE FOLLOWING FOR COMPATABILITY WITH OUR CODE
    // TODO: Dynamic allocate
    int e[22 * 384];  // padded input, unpacked, max size. Will hold c.
    int f[46 * 384];  // coded output, unpacked, max size. Will hold d.

    // Copy the information bits into our container
    for (int i = 0; i < (ncols * Zc); i++) {
        e[i] = (int)c[i];
    }

    // Copy the parity bits into our container
    for (int i = 0; i < (nrows * Zc); i++) {
        f[i] = (int)d[i];
    }

    // information part and puncture columns
    memcpy(&channel_input[0], &e[2 * Zc], (block_length - 2 * Zc) * sizeof(int));
    memcpy(&channel_input[block_length - 2 * Zc], &f[0],
           ((nrows - no_punctured_columns) * Zc - removed_bit) * sizeof(int));
    // memcpy(channel_input,c,Kb*Zc*sizeof(unsigned char));
    return 0;
}
