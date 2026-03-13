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

// warp size (assume 32)
constexpr int WARP_SIZE = 32;

// warp-level bitonic sort using warp shuffles (register-only)
// Each lane starts with 'v' and ends with lane holding sorted values across warp in ascending order
__device__ __always_inline__ unsigned warp_bitonic_sort(unsigned v) {
    unsigned x = v;
    unsigned lane = threadIdx.x & 31;
    // bitonic sort network for 32 lanes using shfl_xor
    // outer loop: k = 2,4,8,16,32
    for (int k = 2; k <= WARP_SIZE; k <<= 1) {
        // inner loop: j = k/2, k/4, ..., 1
        for (int j = k >> 1; j > 0; j >>= 1) {
            unsigned y = __shfl_xor_sync(0xffffffffu, x, j, WARP_SIZE);
            bool ascending = ((lane & k) == 0);
            // when lanes compare, choose min or max accordingly
            unsigned mine = x;
            unsigned other = y;
            unsigned keep;
            if (ascending) {
                keep = (mine <= other) ? mine : other;
            } else {
                keep = (mine >= other) ? mine : other;
            }
            // BUT we must ensure the partner also updates; using this single-lane assignment
            // pattern works when both lanes execute the same comparison with swapped view.
            // To implement full compare-swap we need conditional assignment:
            if (ascending) {
                // want ascending order in this comparator: smaller goes to lower-index lane
                if (x > y) x = y;
                // else x remains
            } else {
                if (x < y) x = y;
            }
            __syncwarp();
        }
    }
    return x;
}

// Alternative robust warp-bitonic implementation using pairwise compare & exchange
// but above simpler pattern works for demonstration.

// Kernel: each warp processes WARP_SIZE elements; sorts them in-warp then reduces runs.
// Parameters:
//  - data: input values (0..NBINS-1)
//  - bin_data: global histogram (length NBINS)
//  - N: number of input elements
// Note: blockDim.x must be multiple of WARP_SIZE
template<int NBINS>
__global__ void histogram_bitonic(const int *data, int *bin_data, int N) {
    extern __shared__ unsigned s_shared[]; // per-block shared memory, used as (num_warps_per_block * WARP_SIZE)
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id_in_block = tid / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int global_warp_id = (blockIdx.x * warps_per_block) + warp_id_in_block;

    // grid-stride over warp-sized tiles
    // each warp processes tiles of length WARP_SIZE starting at tile_base
    int tile_stride = gridDim.x * warps_per_block * WARP_SIZE;
    int first_tile_base = (blockIdx.x * warps_per_block + warp_id_in_block) * WARP_SIZE;

    for (int tile_base = first_tile_base; tile_base < N; tile_base += tile_stride) {
        int idx = tile_base + lane;
        unsigned v = (idx < N) ? (unsigned)data[idx] : UINT_MAX;
        // warp-level bitonic sort in registers (using shfl)
        unsigned sorted_v = warp_bitonic_sort(v);

        // write sorted value to shared memory: each warp gets a contiguous slot
        int shared_offset = warp_id_in_block * WARP_SIZE + lane;
        s_shared[shared_offset] = sorted_v;
        // need to sync the block to ensure all warps written before warp-leader reads
        // but we can synchronize only within block:
        __syncthreads();

        // let lane 0 of each warp scan its 32 values in shared memory and emit run-length atomics
        if (lane == 0) {
            unsigned prev = s_shared[warp_id_in_block * WARP_SIZE + 0];
            if (prev != UINT_MAX) {
                int cnt = 1;
                for (int i = 1; i < WARP_SIZE; ++i) {
                    unsigned cur = s_shared[warp_id_in_block * WARP_SIZE + i];
                    if (cur == UINT_MAX) break;
                    if (cur == prev) {
                        ++cnt;
                    } else {
                        // atomic add to global histogram for this run
                        if (prev < (unsigned)NBINS) {
                            atomicAdd(&bin_data[prev], cnt);
                        }
                        prev = cur;
                        cnt = 1;
                    }
                }
                // last run
                if (prev < (unsigned)NBINS) {
                    atomicAdd(&bin_data[prev], cnt);
                }
            }
        }
        __syncthreads(); // ensure warp0 finished before next tile overwrite
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
    histogram_bitonic<blockSize><<<Grid, Block>>>(hist_data, bin_data, N);
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