# CUDA Bitonic Sorting Optimization Exploration

## Overview

This project explores **progressive optimization techniques for Bitonic Sorting on GPU using CUDA**.  
The goal is to study how different GPU programming techniques impact the performance of a parallel sorting algorithm.

Bitonic sort is a classic **sorting network algorithm** that is well suited for parallel architectures such as GPUs due to its structured comparison pattern.

In this project, several implementations are developed and optimized step by step:

1. Naive tiled implementation
2. Shared memory optimization
3. Bank conflict mitigation using padding
4. Warp-level primitive optimization

We evaluate their performance and analyze the trade-offs between different optimization strategies.

---

# Bitonic Sort Algorithm

## Bitonic Sequence

A **bitonic sequence** is a sequence that first monotonically increases and then monotonically decreases.

Example:

```
1 3 7 8 6 5 4 2
```

This sequence increases and then decreases.

Bitonic sort works by:

1. Constructing bitonic sequences
2. Performing **bitonic merge operations** to produce sorted sequences.

---

## Bitonic Sorting Network

The following diagram illustrates the sorting network for **8 elements**:

```
Stage 1
(0,1) (2,3) (4,5) (6,7)

Stage 2
(0,2) (1,3) (4,6) (5,7)

Stage 3
(1,2) (5,6)

Stage 4
(0,4) (1,5) (2,6) (3,7)

Stage 5
(2,4) (3,5)

Stage 6
(1,2) (3,4) (5,6)
```

Graphically this can be visualized as:

```
0 ──┐     ┌─────────────
1 ──┘─────┘
2 ──┐     ┌─────────────
3 ──┘─────┘
4 ──┐     ┌─────────────
5 ──┘─────┘
6 ──┐     ┌─────────────
7 ──┘─────┘
```

Each horizontal line represents a data element, and each vertical comparator swaps elements if necessary.

The algorithm complexity is:

```
O(n log² n)
```

Although asymptotically slower than quicksort, the **deterministic comparison pattern** makes it ideal for GPU parallelization.

---

# GPU Implementation Strategy

Because a CUDA thread block has a limited number of threads (typically **1024**), large arrays must be processed using **tiling**:

1. Divide the array into **tiles**
2. Each block sorts one tile
3. Global merge steps combine sorted tiles

Optimization strategies explored include:

- reducing global memory traffic
- using shared memory
- reducing shared memory bank conflicts
- leveraging warp-level primitives

---

# Project Structure

```
.
├── bitontic_sorting.cu
├── bitontic_sorting_block.cu
├── bitontic_sorting_block_v1.cu
├── bitontic_sorting_block_v2.cu
├── bitontic_sorting_block_v3.cu
└── README.md
```

---

# Implementation Versions

## 1. `bitontic_sorting.cu`

A simple bitonic sort kernel running **entirely within a single block**.

### Characteristics

- Uses shared memory
- Only supports

```
N < 1024
```

due to CUDA block size limits.

This version demonstrates the **basic bitonic sorting network on GPU**.

---

## 2. `bitontic_sorting_block.cu`

A **naive tiled implementation** that supports larger arrays.

### Approach

1. Divide the array into tiles
2. Each CUDA block sorts one tile
3. Global merge stages combine tiles

### Optimization

None.

This version serves as the **baseline implementation**.

---

## 3. `bitontic_sorting_block_v1.cu`

Introduces **shared memory optimization**.

### Idea

Instead of performing all operations on global memory:

```
global memory → shared memory
sort in shared memory
shared memory → global memory
```

### Benefit

Shared memory is much faster than global memory, which reduces memory latency.

---

## 4. `bitontic_sorting_block_v2.cu`

Attempts to mitigate **shared memory bank conflicts** using **padding**.

### Bank Conflict Problem

Shared memory is divided into multiple memory banks.  
When multiple threads access the same bank simultaneously, accesses are serialized.

The bitonic access pattern

```
ixj = tid ^ j
```

can cause many threads to access the same bank.

### Proposed Solution

Introduce padding in shared memory:

```
padded_idx = tid + tid / warpSize
```

This offsets addresses to reduce bank conflicts.

---

## 5. `bitontic_sorting_block_v3.cu`

Introduces **warp-level primitives** for optimization.

### Key Idea

When:

```
j < warpSize (32)
```

the comparison partner must be within the **same warp**.

Instead of using shared memory:

```
shared memory
+ __syncthreads()
```

we use warp shuffle instructions:

```
__shfl_xor_sync()
```

### Advantages

- avoids shared memory
- avoids bank conflicts
- removes synchronization overhead
- uses fast register-level communication

---

# Experimental Setup

Input size:

```
N = 2^25
```

GPU execution time is measured for each implementation.

---

# Performance Results

| Version | Optimisation | Latency (ms) |
|-------|-------------|-------------|
| naive | - | 4.992576 |
| v1 | shared memory optimisation | 4.490467 |
| v2 | bank conflict optimisation | 4.853584 |
| v3 | warp-level optimisation | 2.872055 |

### Speedup Compared to Naive

| Version | Speedup |
|-------|--------|
| v1 | 1.11x |
| v2 | 1.03x |
| v3 | 1.74x |

Warp-level optimization provides the **largest improvement**.

---

# Analysis

## Why Shared Memory Helps

Shared memory reduces global memory accesses, which significantly lowers latency.

This explains the improvement from:

```
naive → v1
```

---

## Why Padding (v2) Did Not Improve Performance

Although padding reduces bank conflicts, the performance decreased slightly.

Possible reasons include:

### 1. Extra Index Computation

Padding introduces additional index calculations:

```
padded_idx = tid + tid / warpSize
```

These arithmetic operations increase instruction count.

On GPUs, this extra overhead can outweigh the benefits of reduced bank conflicts.

---

### 2. Bank Conflicts Were Not the Main Bottleneck

Profiling suggests that bank conflicts were not severe enough to dominate execution time.

Therefore eliminating them provides little benefit.

---

### 3. Reduced Memory Efficiency

Padding modifies memory layout, which may negatively impact memory access patterns.

---

## Why Warp-Level Optimization Works Best

The warp-level version eliminates several overheads simultaneously:

- no shared memory access
- no bank conflicts
- no block-level synchronization
- register-to-register communication

Because shuffle instructions operate **within a warp**, they are extremely efficient.

---

# Key Takeaways

1. Shared memory optimization provides moderate performance improvements.
2. Bank conflict mitigation via padding may introduce additional overhead.
3. Warp-level primitives (`__shfl_sync`) are highly effective for intra-warp communication.
4. Understanding GPU architecture (warps, memory hierarchy, synchronization) is essential for performance optimization.

---

# Build and Run

Compile with:

```bash
nvcc -O3 bitontic_sorting_block_v3.cu -o bitonic
```

Run:

```bash
./bitonic
```

---

# Future Work

Possible future improvements include:

- using cooperative groups for cross-block synchronization
- implementing hierarchical bitonic merge
- comparing with GPU radix sort or thrust::sort