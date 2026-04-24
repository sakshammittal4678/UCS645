#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <random>
#include <omp.h>
#include "correlate.h"

using namespace std;
using namespace chrono;

// Generate random input matrix
void generateData(float* data, int ny, int nx, unsigned seed = 42) {
    mt19937 gen(seed);
    uniform_real_distribution<float> dis(0.0f, 100.0f);
    for (int i = 0; i < ny * nx; i++) {
        data[i] = dis(gen);
    }
}

// Check if two result matrices are close enough
bool resultsMatch(const float* a, const float* b, int size, float tol = 1e-4f) {
    for (int i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > tol) {
            cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << endl;
            return false;
        }
    }
    return true;
}

void printUsage(const char* prog) {
    cerr << "Usage: " << prog << " <ny> <nx> [num_threads] [mode]" << endl;
    cerr << "  ny           - number of vectors (rows)" << endl;
    cerr << "  nx           - length of each vector (columns)" << endl;
    cerr << "  num_threads  - number of OpenMP threads (default: max available)" << endl;
    cerr << "  mode         - which version to run: seq | par | opt | all (default: all)" << endl;
    cerr << endl;
    cerr << "Examples:" << endl;
    cerr << "  " << prog << " 500 2000           -> runs all 3 versions" << endl;
    cerr << "  " << prog << " 500 2000 1 seq     -> sequential only" << endl;
    cerr << "  " << prog << " 500 2000 4 par     -> parallel only (4 threads)" << endl;
    cerr << "  " << prog << " 500 2000 4 opt     -> optimized only (4 threads)" << endl;
    cerr << "  " << prog << " 500 2000 4 all     -> all 3 versions (4 threads)" << endl;
}

int main(int argc, char* argv[]) {
    // ---- Parse command-line arguments ----
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    int ny = atoi(argv[1]);
    int nx = atoi(argv[2]);
    int num_threads = omp_get_max_threads();
    string mode = "all";

    if (argc >= 4) {
        num_threads = atoi(argv[3]);
    }
    if (argc >= 5) {
        mode = argv[4];
    }

    if (ny <= 0 || nx <= 0 || num_threads <= 0) {
        cerr << "Error: ny, nx, and num_threads must be positive integers." << endl;
        return 1;
    }

    if (mode != "seq" && mode != "par" && mode != "opt" && mode != "all") {
        cerr << "Error: mode must be one of: seq, par, opt, all" << endl;
        printUsage(argv[0]);
        return 1;
    }

    omp_set_num_threads(num_threads);

    cout << "=============================================" << endl;
    cout << "  Pairwise Correlation Assignment" << endl;
    cout << "=============================================" << endl;
    cout << "Matrix size : " << ny << " x " << nx << endl;
    cout << "OMP threads : " << num_threads << endl;
    cout << "Mode        : " << mode << endl;
    cout << "Result size : " << ny << " x " << ny << " (upper triangle)" << endl;
    cout << endl;

    // ---- Allocate data ----
    long long data_size   = (long long)ny * nx;
    long long result_size = (long long)ny * ny;

    vector<float> data(data_size);
    vector<float> res_seq(result_size, 0.0f);
    vector<float> res_par(result_size, 0.0f);
    vector<float> res_opt(result_size, 0.0f);

    generateData(data.data(), ny, nx);

    double time_seq = -1, time_par = -1, time_opt = -1;

    // ---- Part 1: Sequential ----
    if (mode == "seq" || mode == "all") {
        cout << "--- Part 1: Sequential ---" << endl;
        auto t0 = high_resolution_clock::now();
        correlate_seq(ny, nx, data.data(), res_seq.data());
        auto t1 = high_resolution_clock::now();
        time_seq = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
        cout << "Time: " << time_seq << " ms" << endl;
    }

    // ---- Part 2: OpenMP Parallel ----
    if (mode == "par" || mode == "all") {
        cout << "\n--- Part 2: OpenMP Parallel ---" << endl;
        auto t2 = high_resolution_clock::now();
        correlate_par(ny, nx, data.data(), res_par.data());
        auto t3 = high_resolution_clock::now();
        time_par = duration_cast<microseconds>(t3 - t2).count() / 1000.0;
        cout << "Time: " << time_par << " ms" << endl;

        if (time_seq > 0) {
            cout << "Speedup vs Sequential: " << (time_seq / time_par) << "x" << endl;
            cout << "Correctness: " << (resultsMatch(res_seq.data(), res_par.data(), result_size) ? "PASS" : "FAIL") << endl;
        } else {
            cout << "(Run with mode=all to see speedup and correctness check)" << endl;
        }
    }

    // ---- Part 3: Optimized ----
    if (mode == "opt" || mode == "all") {
        cout << "\n--- Part 3: Optimized (SIMD + OpenMP + cache) ---" << endl;
        auto t4 = high_resolution_clock::now();
        correlate_opt(ny, nx, data.data(), res_opt.data());
        auto t5 = high_resolution_clock::now();
        time_opt = duration_cast<microseconds>(t5 - t4).count() / 1000.0;
        cout << "Time: " << time_opt << " ms" << endl;

        if (time_seq > 0) {
            cout << "Speedup vs Sequential: " << (time_seq / time_opt) << "x" << endl;
            cout << "Correctness: " << (resultsMatch(res_seq.data(), res_opt.data(), result_size) ? "PASS" : "FAIL") << endl;
        } else {
            cout << "(Run with mode=all to see speedup and correctness check)" << endl;
        }
    }

    // ---- Summary (only when running all) ----
    if (mode == "all") {
        cout << "\n=============================================" << endl;
        cout << "  Summary" << endl;
        cout << "=============================================" << endl;
        cout << "Sequential : " << time_seq << " ms" << endl;
        cout << "Parallel   : " << time_par << " ms  (speedup " << (time_seq/time_par) << "x)" << endl;
        cout << "Optimized  : " << time_opt << " ms  (speedup " << (time_seq/time_opt) << "x)" << endl;
        cout << "=============================================" << endl;
    }

    return 0;
}
