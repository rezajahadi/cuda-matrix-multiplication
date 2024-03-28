# CUDA Matrix-Vector Multiplication

This repository contains CUDA code for performing matrix-vector multiplication using row-wise decomposition. The CUDA kernel launches multiple threads to efficiently compute the result in parallel on a GPU.

The code calculates the speedup for matrix-vector multiplication with varying thread configurations and prints the speedup for each configuration. The test was started from threads_per_block = 32 to 32*20.

## Results

Speed up vs number of threads per block:

![image](https://github.com/rezajahadi/cuda-matrix-multiplication/assets/91501414/fbc50c82-a0e7-4187-9c25-5eff1173ac50)

We can clearly see that the speedup decreases as the number of threads per blocks increased.


## Execution Environment

The code was run on the `wes-00-00` GPU node of Wesley.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C compiler (e.g., GCC) for compiling host code
- `sshpass` utility for password-based SSH authentication (install using `sudo apt-get install sshpass`)

## Usage

### Compilation

Compile the code using the provided Makefile:

```bash
nvcc -g -G script.cu -o script
```

This will generate an executable named `matrix_mult`.

### Execution

Run the executable, providing the number of threads as an argument:

./matrix_mult <threadsnum>


Replace `<threadsnum>` with the desired number of threads per block. The program will calculate the parallel execution time and print the speedup for each thread configuration.

### Example

To run the program with 32 threads per block:

./matrix_mult 32

### Submit a job

In order to submit a job in the interaction section:

```bash
qsub -I -l host=wes-00-00
```

## Compile and Deploy

I provided a bash script that automates the process of compiling and executing the CUDA code on the specified remote server with the provided file and input value 'n'


## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for any improvements or bug fixes.


Ensure to provide clear instructions on how to use the script and what parameters it expects. Adjust the file paths and server credentials in the script according to your environment.
