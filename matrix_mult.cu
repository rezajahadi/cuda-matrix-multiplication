#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel function for matrix-vector multiplication using row-wise decomposition
__global__ void matrixVectorMultiply(float *output, float *inputMatrix, float *inputVector, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += inputMatrix[row * N + col] * inputVector[col];
        }
        output[row] = sum;
    }
}

// Function to generate random values for matrix elements
void generateRandomMatrix(float *matrix, int N) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = (float)rand() / RAND_MAX; // Random float between 0 and 1
        }
    }
}

int main() {
    int N = 640;

    for (int i = 1; i <= 20; i++) {
        int threadsnum = 32 * i;

        // Start measuring sequential execution time
        clock_t sequential_start = clock();

        // Allocate memory for input matrix, input vector, and output vector
        float *inputMatrix = (float *)malloc(N * N * sizeof(float));
        float *inputVector = (float *)malloc(N * sizeof(float));
        float *outputVector = (float *)malloc(N * sizeof(float));

        // Generate random values for input matrix and input vector
        generateRandomMatrix(inputMatrix, N);
        for (int j = 0; j < N; j++) {
            inputVector[j] = (float)rand() / RAND_MAX; // Random float
        }

        // Allocate memory on the device
        float *d_inputMatrix, *d_inputVector, *d_outputVector;
        cudaMalloc((void **)&d_inputMatrix, N * N * sizeof(float));
        cudaMalloc((void **)&d_inputVector, N * sizeof(float));
        cudaMalloc((void **)&d_outputVector, N * sizeof(float));

        // Copy data from host to device
        cudaMemcpy(d_inputMatrix, inputMatrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_inputVector, inputVector, N * sizeof(float), cudaMemcpyHostToDevice);

        // Define kernel launch configuration
        int threadsPerBlock = threadsnum;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record start event for parallel execution time
        cudaEventRecord(start);

        // Launch kernel for matrix-vector multiplication
        matrixVectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_outputVector, d_inputMatrix, d_inputVector, N);

        // Record stop event for parallel execution time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time for parallel execution
        float parallel_milliseconds = 0;
        cudaEventElapsedTime(&parallel_milliseconds, start, stop);

        // Calculate speedup
        float speedup = 2.88666f / parallel_milliseconds;

        // Print speedup for the current i
        printf("Threads: %d, Speedup: %.5f\n", threadsnum, speedup);

        // Free device memory
        cudaFree(d_inputMatrix);
        cudaFree(d_inputVector);
        cudaFree(d_outputVector);

        // Free host memory
        free(inputMatrix);
        free(inputVector);
        free(outputVector);

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
