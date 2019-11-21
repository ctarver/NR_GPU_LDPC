# Overview of Operation. 
This file presents an overview of the code structure and how it relates to the LDPC Decoding Algorithm.

## Main Code Structure
The code launches from main.cu. Here we take in any args then lauch the simulator in the `run_test()` function. 

### Run Test
The run test function lives in the main.cu file. This is the overall simulation wrapper. 
It will allocate memory on the host and device, generate codewords, pass them through an AWGN channel, then pass them to the GPU for decoding.

### Decoding Kernels
The decoding is done on CUDA kernels. They are cnp_processing, vnp_processing which live in ldpc_kernels.cu.
There is a special kernel for the 1st iteration of CNP and last iteration of VNP (where hard decision takes place).

## LDPC Review
The code is generated from a base parity check matrix (`h_base`). 
Each element in `h_base` represents one submatrix which is an idenity matrix with a right circular shift. 
The size of the submatrix is given by `Z` which is known as the lifting factor. 

For example the 802.11 base matrix is:
```
	57, -1, -1, -1, 50, -1, 11, -1, 50, -1, 79, -1,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	3, -1, 28, -1,  0, -1, -1, -1, 55,  7, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	30, -1, -1, -1, 24, 37, -1, -1, 56, 14, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1,
	62, 53, -1, -1, 53, -1, -1,  3, 35, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1,
	40, -1, -1, 20, 66, -1, -1, 22, 28, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1,
	0, -1, -1, -1,  8, -1, 42, -1, 50, -1, -1,  8, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1,
	69, 79, 79, -1, -1, -1, 56, -1, 52, -1, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1,
	65, -1, -1, -1, 38, 57, -1, -1, 72, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1,
	64, -1, -1, -1, 14, 52, -1, -1, 30, -1, -1, 32, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1,
	-1, 45, -1, 70,  0, -1, -1, -1, 77,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1,
	2, 56, -1, 57, 35, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,
	24, -1, 61, -1, 60, -1, -1, 27, 51, -1, -1, 16,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0
```
The lifiting factor, $Z$, is 81, so each submatrix represents an 81x81 matrix. 
If the entry in `h_base` is -1, then the submatrix is filled with 0s. 
If the entry is 0, then the submatrix is a standard idenity matrix.
  
## Decoding
Decoding is done by expanding the full parity check matrix into a Tanner Graph of check nodes and variable nodes. 
Check nodes are represented by rows in **H** amd variable nodes are columns where each column will represent a log-likihood ratio for a corresponding bit in the codeword.
Messages are exchanged beween variable nodes and check nodes for multiple iterations. 
The variable nodes will hopefully converge to some LLR. 
After the max number of iterations are performed, a hard decision is made converting the LLR to a 0 or a 1.
  
### Min-Sum
This is what is used in this work.
Min sum has lower complexity when compared to sum product, usually at the cost of a few dB in BER performace.
This loss can be recovered with a scaling or an offset. 
This work use a scaling factor (\alpha = 0.75). 
  
#### Check Node Processing
In the first iteration, the messages from the VNs to the CNs are set to be the inital LLR of the received codeword bits.
Each check node will recieve messages from the VNs it is connected to and will compute messages to send back. 
This is give below:
R_{mn} = \alpha * product_{n' \in N\n} sign (Q_{mn'}^{old}) * min_{n' \in N\n} |Q_{mn'}^{old}|
Here, R_{mn} is the message from CN m to VN n, and \alpha is the scaling factor for scaled min sum (0.75 is standard). We then take the product of the signs of all other messages we recieved from other connected VNs (represented by the set N) and multiply this by the minium of the absolute value of all those messages. 

#### Variable Node Processing
The VNs will recieve messages, R_mn, from each of the connected check nodes. 
It will then update its estimated LLR as follows:
L_n^{new} = L_n^{old} + \sum (R_mn^new - R_mn^old)

The messages back to each CN is given as the difference in the new LLR and the message we received from that CN:
Q_mn^{new} = L_n - R_mn

After all iterations are complete, we perform hard decision.
if L_n < 0, x_n = 1
else      , x_n = 0;

## GPU Optimizations
### H_compact. 
Instead of storing the full H matrix, we store only the nonzero entries. 
We have an `h_compact1` for CNP where we store the nonzero entries for each row, 
and we have an `h_compact2` for VNP corresponding to nonzero columns. 
Each of these are stored in the fast, constant globabl memory. 
This is readonly which is fine for a parity check matrix.
This is estimated to imrpove throughput by 8%. 

### Check Node Processing
#### Two Min
The two-min algorithm for min sum is used for the CNP kernel.
Here, we walk through all the nonzero elements in a row (given efficiently from `h_compact1`). 
We access the proper element of the messages from the VNs according to what we read from `h_compact1`.
If it is the minimum that we've seen so far while walking across nonzero entries, we record it for computing the messages back later.
We also must keep track of the second minimum that we see for the case of sending a message back to the CN which had the min.

#### Char Signs.
For the messages to VNs, we need a product of signs. 
As we walk accross nonzero entires in the parity check matrix, we record the current sign of the VN message, Q, in a char.
This is bit shift according to the index of the nonzero entry. 
So at the end of the first recurrsion of the two min algorithm, we have one char, `Q_sign`, storing all the signs of all the VN to CN messages. 

#### Variable Node Messages
The variable node messages are not computed in the variable node processing kernels; those only update the LLR.
Instead this is easily computed in each CN as needed since it is just Q = L - R. 
This prevents having to store this in global during VN procssing only for the CNP kernel to have to access it from global. 
The CNP kernel also update the change in variable node messages, R_mn^new - R_mn^old, for the VNs while it is touching that data.
This change in messages is stored in `dev_dt`

### MacroCodewords
It is important to keep the GPU busy. 
Macrocodewords are created to do that in this GPU code. 
Each macrocodeword contains multiple, independent codewords that will be deocded simultaneouly. 
Each is addresssed over the y dimension in the thread grid. 
Multiple macrocodewords can be sent to the GPU at a time. 
These are addressed over the y-dimension of the block grid. 

### Streams
Streams are like independent processes running on the GPU.
The main idea of the stream is to on one thread, copy data and start processing and then immeaditly, while the inital stream is still computing, copy the next data in on its own stream.
This has the benefit of reducing latency since we can copy a small number of codewords at once. 
This also has the benefit of hiding some of the memory copy since while we are copying data for one stream, other streams will be computing, so the GPU computation resources are never idle.

### Grid Dimensions
This is more of a note than an optimization.
CNP:
Blocks are organized as (submatrix row index, MCW index) and threads are (subrow index, codeword inside a MCW index).
VNP:
Blocks are organized as (submatrix column index, MCW index) and threads are (subrow index, codeword inside a MCW index).

So overall, a block will work on a submatrix row or column of the original `hbase`. 
