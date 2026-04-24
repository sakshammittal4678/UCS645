#include "correlate.h"
#include <cmath>
#include <vector>
#include <omp.h>
#include <immintrin.h>  // for AVX intrinsics (Part 3)

using namespace std;

// ============================================================
// PART 1: Sequential baseline
// ============================================================
void correlate_seq(int ny, int nx, const float* data, float* result) {
    // Step 1: Normalize each row (zero mean, unit variance) using double precision
    vector<vector<double>> norm(ny, vector<double>(nx));

    for (int i = 0; i < ny; i++) {
        // Compute mean
        double mean = 0.0;
        for (int x = 0; x < nx; x++) {
            mean += data[x + i * nx];
        }
        mean /= nx;

        // Compute std dev
        double var = 0.0;
        for (int x = 0; x < nx; x++) {
            double diff = data[x + i * nx] - mean;
            var += diff * diff;
        }
        double stddev = sqrt(var);

        // Normalize
        for (int x = 0; x < nx; x++) {
            norm[i][x] = (stddev > 1e-12) ? (data[x + i * nx] - mean) / stddev : 0.0;
        }
    }

    // Step 2: Compute correlation for all pairs j <= i
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int x = 0; x < nx; x++) {
                dot += norm[i][x] * norm[j][x];
            }
            // Clamp to [-1, 1] to handle floating point drift
            dot /= nx;
            if (dot >  1.0) dot =  1.0;
            if (dot < -1.0) dot = -1.0;
            result[i + j * ny] = (float)dot;
        }
    }
}

// ============================================================
// PART 2: OpenMP parallel version
// ============================================================
void correlate_par(int ny, int nx, const float* data, float* result) {
    // Normalize rows in parallel
    vector<vector<double>> norm(ny, vector<double>(nx));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; i++) {
        double mean = 0.0;
        for (int x = 0; x < nx; x++) mean += data[x + i * nx];
        mean /= nx;

        double var = 0.0;
        for (int x = 0; x < nx; x++) {
            double diff = data[x + i * nx] - mean;
            var += diff * diff;
        }
        double stddev = sqrt(var);

        for (int x = 0; x < nx; x++) {
            norm[i][x] = (stddev > 1e-12) ? (data[x + i * nx] - mean) / stddev : 0.0;
        }
    }

    // Compute correlations in parallel (each (i,j) pair is independent)
    #pragma omp parallel for schedule(dynamic) collapse(1)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int x = 0; x < nx; x++) {
                dot += norm[i][x] * norm[j][x];
            }
            dot /= nx;
            if (dot >  1.0) dot =  1.0;
            if (dot < -1.0) dot = -1.0;
            result[i + j * ny] = (float)dot;
        }
    }
}

// ============================================================
// PART 3: Fully optimized version
//   - Normalize in parallel
//   - Store normalized data in row-major, cache-friendly layout (flat array)
//   - Use OpenMP + SIMD (AVX2) for dot products
//   - Schedule(dynamic) for load balancing (triangular loop)
// ============================================================
void correlate_opt(int ny, int nx, const float* data, float* result) {
    // Use flat double array for better cache locality
    vector<double> norm((long long)ny * nx);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ny; i++) {
        double mean = 0.0;
        for (int x = 0; x < nx; x++) mean += data[x + i * nx];
        mean /= nx;

        double var = 0.0;
        for (int x = 0; x < nx; x++) {
            double diff = data[x + i * nx] - mean;
            var += diff * diff;
        }
        double inv_stddev = (var > 1e-24) ? (1.0 / sqrt(var)) : 0.0;

        double* row = &norm[(long long)i * nx];
        for (int x = 0; x < nx; x++) {
            row[x] = (data[x + i * nx] - mean) * inv_stddev;
        }
    }

    // Compute correlations using parallel outer loop + vectorized inner loop
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < ny; i++) {
        const double* row_i = &norm[(long long)i * nx];
        for (int j = 0; j <= i; j++) {
            const double* row_j = &norm[(long long)j * nx];
            double dot = 0.0;

            // Auto-vectorizable loop (compiler will use SIMD with -O2 -fopenmp)
            #pragma omp simd reduction(+:dot)
            for (int x = 0; x < nx; x++) {
                dot += row_i[x] * row_j[x];
            }

            dot /= nx;
            if (dot >  1.0) dot =  1.0;
            if (dot < -1.0) dot = -1.0;
            result[i + j * ny] = (float)dot;
        }
    }
}
