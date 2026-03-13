#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#ifndef UINT_MAX
#define UINT_MAX 0xFFFFFFFFu
#endif

__global__ void histogram(int *hist_data, int *bin_data)
{
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(&bin_data[hist_data[gtid]], 1);
}

template <int blockSize>
__global__ void histogram_smem(int *hist_data, int *bin_data, int N) {
	__shared__ int smem[256];
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

// 要求: blockSize 是 power-of-two，且 <= 1024
template<int blockSize>
__global__ void histogram_bitonic(const int *hist_data, int *bin_data, int N) {
    // per-block shared buffer to hold one tile of values (unsigned)
    extern __shared__ unsigned svals[]; // size = blockSize * sizeof(unsigned)
    const int tid = threadIdx.x;
    const int base = blockIdx.x * blockSize;

    // grid-stride over tiles: each iteration we process one tile of length blockSize
    for (int tile_base = base; tile_base < N; tile_base += gridDim.x * blockSize) {
        int idx = tile_base + tid;
        // load or pad with UINT_MAX (so padded values go to end after sorting)
        unsigned v = (idx < N) ? (unsigned)hist_data[idx] : UINT_MAX;
        svals[tid] = v;
        __syncthreads();

        // --- bitonic sort in shared memory (classic) ---
        // ascending sort (small -> large), UINT_MAX will be largest so padded go to end
        for (int k = 2; k <= blockSize; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                int partner = tid ^ j;
                if (partner < blockSize) {
                    unsigned a = svals[tid];
                    unsigned b = svals[partner];
                    bool ascending = ((tid & k) == 0);
                    bool do_swap = false;
                    if (ascending) {
                        if (a > b) do_swap = true;
                    } else {
                        if (a < b) do_swap = true;
                    }
                    if (do_swap) {
                        svals[tid] = b;
                        svals[partner] = a;
                    }
                }
                __syncthreads();
            }
        }
        // --- sorted tile in svals[0..blockSize-1] ---

        // run-length encode and atomicAdd to global bins
        // simple approach: let thread 0 scan the sorted tile and emit run updates
        if (tid == 0) {
            unsigned prev = svals[0];
            if (prev != UINT_MAX) {
                int cnt = 1;
                for (int i = 1; i < blockSize; ++i) {
                    unsigned cur = svals[i];
                    if (cur == UINT_MAX) break;
                    if (cur == prev) {
                        ++cnt;
                    } else {
                        // do one global atomic per run
                        atomicAdd(&bin_data[prev], cnt);
                        prev = cur;
                        cnt = 1;
                    }
                }
                // last run
                atomicAdd(&bin_data[prev], cnt);
            }
        }
        __syncthreads();
    }
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
    //histogram_smem<blockSize><<<Grid, Block>>>(hist_data, bin_data, N);
    histogram_bitonic_sorting<blockSize><<<Grid, Block>>>(hist_data, bin_data, N);
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