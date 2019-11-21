
# CMakeLists.txt
# @author: Chance Tarver
# @date: Nov 21, 2019

cmake_minimum_required (VERSION 3.8)
project(5GNR_LDPC CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED 14)

find_package(CUDA  REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(${CMAKE_SOURCE_DIR}/lib/)

set(SRC_FILES
	"src/main.cpp" 
	"src/host_helpers.cpp" 
	"src/cuda_gpu_wrapper/gpu_setup.cu" 
	"src/nr_ldpc.cpp"
	"src/encoding_tools.cpp"
	"src/setup_exp.cu"
	"src/gpu_decoder_top.cu"
	"src/ldpc_kernels.cu"
)
add_executable (5GNR_LDPC_Testbench ${SRC_FILES})