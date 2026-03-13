// file: warp_bitonic.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <climits>
#include <cstdint>

#ifndef UINT_MAX
#define UINT_MAX 0xFFFFFFFFu
#endif

constexpr int WARP_SIZE = 32;

// warp-level bitonic sort for one value per lane (unsigned)
// returns the lane's sorted value (ascending within warp)
__device__ __forceinline__ unsigned warp_bitonic_sort32(unsigned x) {
    const unsigned full_mask = 0xFFFFFFFFu;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // classic bitonic network implemented with shuffles
    for (int k = 2; k <= WARP_SIZE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            unsigned y = __shfl_xor_sync(full_mask, x, j, WARP_SIZE);
            bool ascending = ((lane & k) == 0);
            bool greater = x > y;
            // if ascending and x>y, take partner (y)
            // if descending and x<=y, take partner (y)
            // otherwise keep x
            unsigned newx = ((ascending && greater) || (!ascending && !greater)) ? y : x;
            x = newx;
            // no __syncwarp needed; shfl is warp-synchronous
        }
    }
    return x;
}

// Kernel: each warp sorts 32 consecutive integers in-place in global memory.
// data: int* (values assumed non-negative or cast to unsigned). N = number of elements.
__global__ void warp_bitonic_sort_kernel(unsigned *data, int N) {
    // global thread id
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = gtid >> 5;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    // grid-stride loop over warp-sized tiles
    for (int base_warp = warp_id; base_warp < ( (N + WARP_SIZE - 1) / WARP_SIZE ); base_warp += total_warps) {
        int base_idx = base_warp * WARP_SIZE;
        int idx = base_idx + lane;
        unsigned v = (idx < N) ? data[idx] : UINT_MAX; // pad if out-of-range

        unsigned sorted_v = warp_bitonic_sort32(v);

        if (idx < N) data[idx] = sorted_v;
    }
}

// ---------------- Host test ----------------
int main(int argc, char **argv) {
    // test parameters
    float milliseconds = 0;
    const int N = 1024;
    const int threadsPerBlock = 128; // must be multiple of 32
    const int blocks = ( (N + threadsPerBlock - 1) / threadsPerBlock );

    // allocate & init host data
    unsigned *h = (unsigned*)malloc(N * sizeof(unsigned));
    std::mt19937 rng(12345);
    std::uniform_int_distribution<unsigned> ud(0, 1000000u);
    for (int i = 0; i < N; ++i) h[i] = ud(rng);

    // copy to device
    unsigned *d;
    cudaMalloc(&d, N * sizeof(unsigned));
    cudaMemcpy(d, h, N * sizeof(unsigned), cudaMemcpyHostToDevice);

    dim3 grid(blocks);
    dim3 block(threadsPerBlock);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // launch kernel
    warp_bitonic_sort_kernel<<<grid, block>>>(d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaDeviceSynchronize();

    // copy back
    unsigned *out = (unsigned*)malloc(N * sizeof(unsigned));
    cudaMemcpy(out, d, N * sizeof(unsigned), cudaMemcpyDeviceToHost);

    // verify: each warp (32-chunk) must be sorted ascending
    bool ok = true;
    int chunks = (N + WARP_SIZE - 1) / WARP_SIZE;
    for (int c = 0; c < chunks; ++c) {
        int base = c * WARP_SIZE;
        int high = std::min(N, base + WARP_SIZE);
        for (int i = base + 1; i < high; ++i) {
            if (out[i-1] > out[i]) {
                printf("Chunk %d not sorted at %d : %u > %u\n", c, i-1, out[i-1], out[i]);
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }

    if (ok) printf("PASS: each warp-slice (size 32) is sorted ascending.\n");
    else printf("FAIL: some warp-slice not sorted.\n");
    printf("latency = %f ms\n", milliseconds);

    // cleanup
    cudaFree(d);
    free(h);
    free(out);
    return 0;
}