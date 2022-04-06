# GPU LDPC Decoder
The repo presents a GPU Implementation of an LDPC decoder using the CUDA language for NVIDIA based GPUs.
This work is based on the work from https://github.com/robertwgh/cuLDPC/ done by G. Wang, M. Wu, B. Yin and J. R. Cavallaro.

Please read the [Overview](Overview.md) documnent for details on LDPC and the GPU optimizations in the project.

## Installation 
### Before installing
Install the CUDA Toolkit. I am using what is currently the [latest one, 10.1.](https://developer.nvidia.com/cuda-downloads?)

### Building the Project
Clone the repo.
```
git clone git@github.com:ctarver/NR_GPU_LDPC.git
```

In the root of the repo create a build directory and cd to it.
```
mkdir build
cd build
```

Run CMAKE to generate the makefile. You must be in a recent CMAKE version (>3.8) for it to find your CUDA installation.
```
cmake ..
```

Run make. 
```
make
```
That's it! There should now be an executible in the build directory. 


## Running The Code
If you went through the installation, there should be an executible in the build directory. Run it like this:
```
./build/GPU_LDPC_Decoder snri_min snr_max snr_step
```
MY ARGS AREN'T WORKING. FOR THE TIME BEING, MANUALLY ADJUST THE BOUNDS IN THE main.cu FILE AND RECOMPILE!!

Here, `snr_min` is the lowest SNR you wish to consider in your simulation, `snr_max` is the largest SNR you wish to consider, and `snr_step` is the step size between SNR points. 

### Example Output
```
CUDA LDPC Decoder
Current params:
     SNR Range :0--10 (step = 1)
     Num Trials:1
Device Name:GeForce GTX 950M
    Device Major           :5
    Device Minor           :0
    Clock Rate             :928000
    SM Count               :5
    Total Global Memory    :2147483648
    Total Constant Memory  :65536
    Shared Memory per Block:49152
    Max Threads per Block  :1024
    Registers per Block    :65536
    Warp Size              :32
    Memory Pitch           :2147483647
    maxGridSize[0],[1],[2]  :2147483647,65535,65535
    maxThreadsDim[0],[1],[2]:1024,1024,64
    textureAlignment      :512
    deviceOverlap         :1
    zero-copy data transfers:1

Grid Dimensions:
    CNP Block Dimensions: 12, 10    (Total blocks : 120)
       Thread Dimensions: 96, 8     (Total threads: 768)
    VNP Block Dimensions: 24, 10    (Total blocks : 120)
       Thread Dimensions: 96, 8     (Total threads: 768)

SNR_STEP = 1
Starting SNR = 0
Starting new batch of codewords. Total Unique Codewords = 80
=================================
SNR = 0
Number of codewords decoded:   640
Total Number of User bits:     737280
Number of bit flips:           214704
Number of frame errors:        640
BER:                           0.291211
FER:                           1
CPU elapsed time:              16.9904 (ms)
Average latency per transfer:  16.9904 (ms)
CPU Estimated User Throughput: 43.3939Mbps
              Codeword Thrpt.: 37.6683 kiloCodewordsPerSec
              Codeword bps.:   86.7879 Mbps

Starting SNR = 1
Starting new batch of codewords. Total Unique Codewords = 80
=================================
SNR = 1
Number of codewords decoded:   640
Total Number of User bits:     737280
Number of bit flips:           189296
Number of frame errors:        640
BER:                           0.256749
FER:                           1
CPU elapsed time:              16.8284 (ms)
Average latency per transfer:  16.8284 (ms)
CPU Estimated User Throughput: 43.8116Mbps
              Codeword Thrpt.: 38.0309 kiloCodewordsPerSec
              Codeword bps.:   87.6232 Mbps
Starting SNR = 2
Starting new batch of codewords. Total Unique Codewords = 80
=================================
SNR = 2
Number of codewords decoded:   640
Total Number of User bits:     737280
Number of bit flips:           158816
Number of frame errors:        640
BER:                           0.215408
FER:                           1
CPU elapsed time:              16.9066 (ms)
Average latency per transfer:  16.9066 (ms)
CPU Estimated User Throughput: 43.6089Mbps
              Codeword Thrpt.: 37.8549 kiloCodewordsPerSec
              Codeword bps.:   87.2178 Mbps

Starting SNR = 3
Starting new batch of codewords. Total Unique Codewords = 80
=================================
SNR = 3
Number of codewords decoded:   640
Total Number of User bits:     737280
Number of bit flips:           117664
Number of frame errors:        640
BER:                           0.159592
FER:                           1
CPU elapsed time:              16.8983 (ms)
Average latency per transfer:  16.8983 (ms)
CPU Estimated User Throughput: 43.6303Mbps
              Codeword Thrpt.: 37.8735 kiloCodewordsPerSec
              Codeword bps.:   87.2606 Mbps

Starting SNR = 4
Starting new batch of codewords. Total Unique Codewords = 80
=================================
SNR = 4
Number of codewords decoded:   640
Total Number of User bits:     737280
Number of bit flips:           39848
Number of frame errors:        640
BER:                           0.0540473
FER:                           1
CPU elapsed time:              16.9055 (ms)
Average latency per transfer:  16.9055 (ms)
CPU Estimated User Throughput: 43.612Mbps
              Codeword Thrpt.: 37.8576 kiloCodewordsPerSec
              Codeword bps.:   87.2239 Mbps

Starting SNR = 5
Starting new batch of codewords. Total Unique Codewords = 80
=================================
SNR = 5
Number of codewords decoded:   640
Total Number of User bits:     737280
Number of bit flips:           80
Number of frame errors:        48
BER:                           0.000108507
FER:                           0.075
CPU elapsed time:              16.9872 (ms)
Average latency per transfer:  16.9872 (ms)
CPU Estimated User Throughput: 43.402Mbps
              Codeword Thrpt.: 37.6754 kiloCodewordsPerSec
              Codeword bps.:   86.804 Mbps

Starting SNR = 6
Starting new batch of codewords. Total Unique Codewords = 80
=================================
SNR = 6
Number of codewords decoded:   640
Total Number of User bits:     737280
Number of bit flips:           0
Number of frame errors:        0
BER:                           0
FER:                           0
CPU elapsed time:              16.862 (ms)
Average latency per transfer:  16.862 (ms)
CPU Estimated User Throughput: 43.7244Mbps
              Codeword Thrpt.: 37.9552 kiloCodewordsPerSec
              Codeword bps.:   87.4487 Mbps
```
## Availble PARAMS
The paramters to run the simulation are controlled via input args for the SNR range of the simulation, and the [params.h](src/params.h) file. 
* __NSTREAMS__: The number of NVIDIA CUDA Streams. Each stream operates independently. This allows us to reduce latency by having individual streams working on a small number of codewords. This also helps to hide memory transfers since while one stream may be copying, the others may be computing.
* __N_CW_PER_MCW__: The number of codewords in a single macro codeword. A macro codeword is decoded simulateously across different dimesnions in the CUDA grid. 
* __N_MCW__: The number of macrocodewords to be decoded sequentially on a single stream. 

## TODOs:
* Add support for Early Termination
* Make the memory trransfer take place in fp16
* Run this through nvidia profiler
* Fix ARGS

## Contributing
Please fork the repo to your own account. Make any changes you wish. When a fully implemnted feature is ready, please submit a pull request to this repo. 

