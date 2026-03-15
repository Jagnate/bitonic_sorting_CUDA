// bitonic_anyN_pow2.cu
// 要求：N 是 2 的幂（调用者保证）
// 使用方法参见底部示例

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <climits>

// 返回不大于 v 的最大 2 的幂
static inline int prevPow2(int v) {
    int p = 1;
    while ((p << 1) <= v) p <<= 1;
    return p;
}

// ---------------- tile 内 bitonic sort kernel ----------------
// 每个 block 处理一个 tile，tileSize 必须是 2 的幂，并且我们启动时设置 blockDim.x == tileSize
__global__ void bitonic_sort_tile(int *d_keys, int tileSize, int N) {
    extern __shared__ int s[]; // tileSize * sizeof(int)
    int tid = threadIdx.x;
    int tileId = blockIdx.x;
    int base = tileId * tileSize;

    int gidx = base + tid;
    // 载入（因为 N 为 2 的幂，且 tileSize <= N 且 tiles = N / tileSize，最后 tile 都完整）
    s[tid] = d_keys[gidx];
    __syncthreads();

    // shared memory 上的标准 bitonic 网络（只在 tileSize 内）
    for (int k = 2; k <= tileSize; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            // ixj < tileSize 始终成立因为 tid < tileSize 且 j < tileSize（safe），但保留检查以更安全
            if (ixj < tileSize) {
                int a = s[tid];
                int b = s[ixj];
                int minv = (a < b) ? a : b;
                int maxv = (a < b) ? b : a;
                bool ascending = ((tid & k) == 0);
                s[tid] = ascending ? minv : maxv;
            }
            __syncthreads();
        }
    }

    // 写回全局内存
    d_keys[gidx] = s[tid];
}

// ---------------- 全局 (k, j) 阶段 kernel ----------------
// 每次 kernel 负责整个数组上一组 compare-swap 对 (k, j)
// 约定：N 是 2 的幂
__global__ void bitonic_stage(int *d_keys, int N, int k, int j) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned)N) return;

    unsigned int ixj = idx ^ j;
    if (ixj >= (unsigned)N) return;

    // 必须在同一个 k-block 内
    if ((idx / k) != (ixj / k)) return;

    // 只让 idx < ixj 的线程处理整对，避免 race（单线程读写两个位置）
    if (idx < ixj) {
        int a = d_keys[idx];
        int b = d_keys[ixj];
        int minv = (a < b) ? a : b;
        int maxv = (a < b) ? b : a;
        bool ascending = ((idx & k) == 0);
        if (ascending) {
            d_keys[idx] = minv;
            d_keys[ixj] = maxv;
        } else {
            d_keys[idx] = maxv;
            d_keys[ixj] = minv;
        }
    }
}

// ---------------- Host wrapper ----------------
// d_keys: device pointer (长度至少 N)
// N: 元素个数，必须为 2 的幂
// stream: 可选 CUDA stream（默认 0）
// 该函数在 d_keys 上原地排序（升序）
void bitonic_sort_global(int *d_keys, int N, cudaStream_t stream = 0) {
    if (N <= 1) return;

    float milliseconds = 0;
    // 查询设备能力（device 0）
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;           // 常见 1024
    size_t sharedBytesPerBlock = prop.sharedMemPerBlock; // bytes

    // 计算合适 tileSize（最大为 N），受线程上限和 sharedMem 限制
    int maxTileByShared = (int)(sharedBytesPerBlock / sizeof(int));
    if (maxTileByShared < 1) maxTileByShared = 1;
    int tileLimit = maxThreads < maxTileByShared ? maxThreads : maxTileByShared;
    int tileSize = prevPow2(tileLimit);
    if (tileSize < 2) tileSize = 2;
    if (tileSize > N) tileSize = N; // 如果 N 很小

    // 我们要求 tileSize 为 2 的幂且 blockDim.x == tileSize
    int tiles = N / tileSize; // 因为 N 为 2 的幂且 tileSize 为 2 的幂，整除成立

    // ---------- 1) 每个 tile 局部排序 ----------
    dim3 block(tileSize);
    dim3 grid(tiles);
    size_t sharedBytes = tileSize * sizeof(int);
    // ---------- 2) 全局合并阶段（host-driven k/j loops） ----------
    // 线程配置用于全局阶段
    int threadsPerBlock = 256;
    if (threadsPerBlock > maxThreads) threadsPerBlock = maxThreads;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // launch local sort: 每个 block 排序 tileSize 个元素
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // ---------- 1) 每个 tile 局部排序 ----------
    bitonic_sort_tile<<<grid, block, sharedBytes, stream>>>(d_keys, tileSize, N);
    cudaGetLastError();
    cudaStreamSynchronize(stream); // 必要的全局同步，以保证局部排序完成

    // ---------- 2) 全局合并阶段（host-driven k/j loops） ----------;
    // 从 tileSize 的两个一组开始合并，直到 k == N
    for (int k = 2 * tileSize; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_stage<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_keys, N, k, j);
            cudaGetLastError();
            cudaStreamSynchronize(stream); // 每个 j 阶段必须全局同步
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();
    printf("latency = %f ms\n", milliseconds);
}

// ---------------- 示例主程序 ----------------
#include <algorithm>
int main() {
    const int N = 1 << 20; // 示例: 65536，必须是 2 的幂
    size_t bytes = N * sizeof(int);

    // host 初始化
    int *h = (int*)malloc(bytes);
    for (int i = N - 1; i >= 0; i--) h[i] = i;

    // device
    int *d;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);

    // 排序
    bitonic_sort_global(d, N);

    // 拷回并验证
    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
    // 验证
    bool ok = true;
    for (int i = 1; i < N; ++i) {
        if (h[i-1] > h[i]) { ok = false; break; }
    }
    printf("sorted? %s\n", ok ? "YES" : "NO");

    // 清理
    cudaFree(d);
    free(h);
    return ok ? 0 : 1;
}