#ifndef CORRELATE_H
#define CORRELATE_H

// Part 1: Sequential baseline
void correlate_seq(int ny, int nx, const float* data, float* result);

// Part 2: OpenMP parallel
void correlate_par(int ny, int nx, const float* data, float* result);

// Part 3: Fully optimized (SIMD + OpenMP + cache-friendly layout)
void correlate_opt(int ny, int nx, const float* data, float* result);

#endif
