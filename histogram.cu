#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

__global__ void histogram(int *hist_data, int *bin_data)
{
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(&bin_data[hist_data[gtid]], 1);
}

template <int blockSize>
__global__ void histogram_smem(int *hist_data, int *bin_data) {
	__shared__ int smeme[256];
	int gtid = blockIdx.x * blockSize + threadIdx.x;
	int tid = threadIdx.x;
	smem[tid] = 0;
	__syncthreads();
	for (int i = gtid; i < N; i += gridDim.x * blockSize) {
		int val = hist_data[i];
		atomicAdd(&smem[val], 1);
	}
	__syncthreads();
	atomicAdd(&bin_data[tid], smem[tid]);
}

bool CheckResult(int *out, int *groudtruth, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (out[i] != groudtruth[i])
        {
            return false;
        }
    }
    return true;
}

int main()
{
    float milliseconds = 0;
    const int N = 25600000;
    int *hist = (int *)malloc(N * sizeof(int));
    int *bin = (int *)malloc(256 * sizeof(int));
    int *bin_data;
    int *hist_data;
    cudaMalloc((void **)&bin_data, 256 * sizeof(int));
    cudaMalloc((void **)&hist_data, N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        hist[i] = i % 256;
    }

    int *groudtruth = (int *)malloc(256 * sizeof(int));
    ;
    for (int j = 0; j < 256; j++)
    {
        groudtruth[j] = 100000;
    }

    cudaMemcpy(hist_data, hist, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int gridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //histogram<<<Grid, Block>>>(hist_data, bin_data);
    histgram<blockSize><<<Grid, Block>>>(hist_data, bin_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(bin, bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(bin, groudtruth, 256);
    if (is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        for (int i = 0; i < 256; i++)
        {
            printf("%lf ", bin[i]);
        }
        printf("\n");
    }
    printf("histogram latency = %f ms\n", milliseconds);

    cudaFree(bin_data);
    cudaFree(hist_data);
    free(bin);
    free(hist);
    return 0;
}