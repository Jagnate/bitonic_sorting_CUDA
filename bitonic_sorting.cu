// file: bitonic_naive.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <cassert>

// Naive bitonic sort in one block.
// Requirements:
//  - N must be power of two and <= max threads per block (e.g., 1024).
//  - launch with <<<1, N, N * sizeof(int)>>>
// This sorts in-place the array d_keys (ascending).
__global__ void bitonic_sort_naive(int *d_keys, int N) {
    int tid = threadIdx.x;
    if (tid >= N) return;

    // bitonic sort network
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j; // partner index
            if (ixj > tid) { // ensure one swap per pair
                // determine sort direction for this stage
                bool ascending = ((tid & k) == 0);
                int a = d_keys[tid];
                int b = d_keys[ixj];
                if (ascending) {
                    if (a > b) {
                        d_keys[tid] = b;
                        d_keys[ixj] = a;
                    }
                } else {
                    if (a < b) {
                        d_keys[tid] = b;
                        d_keys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// shared memory optimisation
__global__ void bitonic_sort_v1(int *d_keys, int N) {
    extern __shared__ int s[]; // size N
    int tid = threadIdx.x;

    // load to shared mem
    if (tid < N) s[tid] = d_keys[tid];
    __syncthreads();

    // bitonic sort network
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j; // partner index
            if (ixj > tid) { // ensure one swap per pair
                // determine sort direction for this stage
                bool ascending = ((tid & k) == 0);
                int a = s[tid];
                int b = s[ixj];
                if (ascending) {
                    if (a > b) {
                        s[tid] = b;
                        s[ixj] = a;
                    }
                } else {
                    if (a < b) {
                        s[tid] = b;
                        s[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // write back
    if (tid < N) d_keys[tid] = s[tid];
}

// warp divergence optimisation
__global__ void bitonic_sort_v2(int *d_keys, int N) {
    extern __shared__ int s[]; // size N
    int tid = threadIdx.x;

    // load to shared mem
    if (tid < N) s[tid] = d_keys[tid];
    __syncthreads();

    // bitonic sort network
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            int a = s[tid];
            int b = s[ixj];

            // branchless min/max (ternary compiles to select, not branch)
            int minv = (a < b) ? a : b;
            int maxv = (a < b) ? b : a;

            // decide per-thread whether this thread should keep min or max
            bool ascending = ((tid & k) == 0);

            // write only your own slot (partner will write its own). No races.
            s[tid] = ascending ? minv : maxv;

            // synchronization as before
            __syncthreads();
        }
    }

    // write back
    if (tid < N) d_keys[tid] = s[tid];
}

// Kernel: each block sorts one tile of BLOCK_N ints in-place
// Requirements:
//  - BLOCK_N must be power of two and <= blockDim.x (we use one thread per element)
//  - launch <<<numTiles, BLOCK_N, BLOCK_N * sizeof(int)>>>
//  - global array must be padded / sized >= numTiles * BLOCK_N
template<int BLOCK_N>
__global__ void bitonic_block_sort(int *d_keys, int numTiles) {
    extern __shared__ int s[]; // size = BLOCK_N
    int tid = threadIdx.x;     // 0..BLOCK_N-1
    int tileId = blockIdx.x;   // which tile this block sorts

    if (tileId >= numTiles) return;

    int base = tileId * BLOCK_N;
    // load into shared memory (coalesced)
    s[tid] = d_keys[base + tid];
    __syncthreads();

    // classic bitonic network (block-local)
    for (int k = 2; k <= BLOCK_N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            int a = s[tid];
            int b = s[ixj];

            // branchless min/max (ternary compiles to select, not branch)
            int minv = (a < b) ? a : b;
            int maxv = (a < b) ? b : a;

            // decide per-thread whether this thread should keep min or max
            bool ascending = ((tid & k) == 0);

            // write only your own slot (partner will write its own). No races.
            s[tid] = ascending ? minv : maxv;

            // synchronization as before
            __syncthreads();
        }
    }

    // write back
    d_keys[base + tid] = s[tid];
}

// Host test
int main(int argc, char **argv) {
    // N must be a power of two and <= 1024 (typical max threads per block)
    const int N = 1024; // you can change to 32/64/128/... but must be power of two
    size_t bytes = N * sizeof(int);
    float milliseconds = 0;
    const int BLOCK_N = 256;                // power of two, <= 1024
    int total = 1 << 20;                    // total elements to sort (example)
    int numTiles = (total + BLOCK_N - 1) / BLOCK_N;

    // pad host array to numTiles * BLOCK_N with large sentinel (or INT_MAX)
    int padded = numTiles * BLOCK_N;
    int *h = (int*)malloc(padded * sizeof(int));
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> ud(0, 10000);
    for (int i = 0; i < N; ++i) h[i] = ud(rng);
    // padding for block-level optimisation
    for (int i = total; i < padded; ++i) h[i] = INT_MAX;

    // copy to device
    int *d;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);

    // launch: single block, N threads, shared memory N * sizeof(int)
    // dim3 blocks(1);
    // dim3 threads(N);
    // size_t sharedBytes = N * sizeof(int);

    // blocking
    int threads = BLOCK_N;
    int blocks = numTiles;
    size_t shmem = BLOCK_N * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //bitonic_sort_naive<<<blocks, threads, sharedBytes>>>(d, N);
    //bitonic_sort_v1<<<blocks, threads, sharedBytes>>>(d, N);
    //bitonic_sort_v2<<<blocks, threads, sharedBytes>>>(d, N);
    bitonic_block_sort<BLOCK_N><<<blocks, threads, shmem>>>(d, numTiles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();

    // copy back
    int *out = (int*)malloc(bytes);
    cudaMemcpy(out, d, bytes, cudaMemcpyDeviceToHost);

    // verify
    bool ok = true;
    for (int i = 1; i < N; ++i) {
        if (out[i-1] > out[i]) { ok = false; break; }
    }
    if (ok) printf("PASS: sorted ascending\n");
    else printf("FAIL: not sorted\n");

    // for comparison, sort on CPU and check equality
    std::sort(h, h + N);
    bool same = true;
    for (int i = 0; i < N; ++i) if (h[i] != out[i]) { same = false; break; }
    printf("Matches std::sort? %s\n", same ? "YES" : "NO");
    printf("latency = %f ms\n", milliseconds);

    // cleanup
    free(h);
    free(out);
    cudaFree(d);
    return 0;
}