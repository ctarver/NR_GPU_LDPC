# Linking with Other Projects
This code can be adapted to link with other cmake projects with a few steps:
* Changing the code:
  * Add decoder source to other project
  * Create wrapper functions to interface with the other code
  * Wrap the wrappers in extern C {} if the other project is C based instead of C++.
  * Include your CUDA library's header in the other projects code
  * Add a function call from the other project to the wrapper function
* Changing CMake:  
  * Add CUDA support to the CMake of the other project
  * Create a library in the other project's cmake
  * Link your new CUDA library with the desired target

## Changing the code
### Add LDPC Decoder Source Files to Other Project
For now, I am just copying them over to live in the other project. That way, it can have its own version that I can modify. Alternatively, one could use gitsubmodules or something to pull it into the project. 

### Create Wrapper Functions to Interface with Other Code. 
Don't call the CUDA kernels directly from some other C file. Instead, have the C file call host code that live in a .cu in the decoder project. Maybe create an api.cu. Design a function to launch the decoder given whatever format the other project has inputs and needs output. Maybe something like:
```C
void ldpc_gpu_decoder(char* inputs, char* outputs);
```

### Extern C
The CUDA code is compiled with a C++ compiler under the hood of nvcc. This creates symbol mangling when linking with a C project which creates symbols differently then C++. To avoid this, surround whatever wrapper api function you create with an `extern C{ }` so that it can be detected by your C project. None of the functions or kernels called from your api function need this; only the functions that need to interface with the C project. 

### Create a Header File
The api.cu you make will need a header file with declerations of the functions that need to be called from the other project. Make one. Maybe something like ldpc_lib.h.

### Include the Header File in the Other Project
Include that header file in the other project:
```C
#include ldpc_lib.h
```

###  Call Your CUDA Library Function
Somewhere in the other project, call the function you created in the api.cu. 
```C
ldpc_gpu_decoder(&inputs, &outputs);
```

## Changing CMake Build Instructions
### Add CUDA Support
CMake got official CUDA support around version 3.8. YOU MUST USE SOMETHING MORE RECENT THAN THIS (you technically don't have to, but these instructions only work for recent versions of CMake).

Make sure the minium version of CMake is set apprpriataly in the other project's main CMakeLists.txt file. I used 3.14, but it should work as dar back at 3.8.
```CMake
cmake_minimum_required (VERSION 3.14)
```

Find the line that declares the project. If there isn't one (this wasn't required in old CMake version), make one. Add CUDA to the list of languages.
```CMake
project (OpenAirInterface
        LANGUAGES CUDA CXX)
```

### Create a Library
Create a library that we'll eventually link up with the other project's code. Anywhere in CMake do the following:
```CMake
add_library(your_target_name
        ${OPENAIR1_DIR}/PHY/CODING/TESTBENCH/linking_example.cu)
set_target_properties(your_target_name PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
# We need C++ 11 features in NVCC so we can use stoi
set_property(TARGET your_target_name PROPERTY CUDA_STANDARD 11)
```

The add_library command needs all .cu files that should be included in the library. 

### Link the Library
Find the target you want to link to. In my case, it was ldpctest. Simply add the name of the library from the previous step to the target_link_libraries() command.
```CMake
add_executable(ldpctest  
  ${OPENAIR1_DIR}/PHY/CODING/TESTBENCH/ldpctest.c
  ${OPENAIR_DIR}/common/utils/backtrace.c)
target_link_libraries(ldpctest SIMU PHY PHY_NR m ${ATLAS_LIBRARIES} your_target_name)
```

## TEST! 
Everything should work. 

## Other Options
There is a CMake property called CUDA_SEPARABLE_COMPILATION. I'm not sure what it does, but some people use it. 
```CMake
set_property(TARGET ldpctest PROPERTY CUDA_SEPARABLE_COMPILATION ON)
```
