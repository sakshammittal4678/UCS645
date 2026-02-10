#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>
#include <iomanip>
#include <random>

using namespace std;

/* ---- Scoring Parameters ---- */
const int MATCH_VAL = 2;
const int MISMATCH_VAL = -1;
const int GAP_VAL = -2;

/* ---- Random DNA Generator ---- */
string generateDNA(int n, int seed) {
    mt19937 gen(seed);
    uniform_int_distribution<int> pick(0, 3);
    char bases[4] = {'A','T','G','C'};

    string s;
    s.reserve(n);
    for (int i = 0; i < n; ++i)
        s.push_back(bases[pick(gen)]);
    return s;
}

inline int scorePair(char a, char b) {
    return (a == b) ? MATCH_VAL : MISMATCH_VAL;
}

/* -------- Wavefront (Diagonal) Parallel -------- */
void smithWave(const string& A, const string& B,
               int threads, double &timeTaken)
{
    int n = A.size(), m = B.size();
    vector<vector<int>> H(n+1, vector<int>(m+1, 0));

    double t0 = omp_get_wtime();

    int diagCount = n + m - 1;
    for (int d = 1; d <= diagCount; ++d) {

        int rStart = max(1, d - m + 1);
        int rEnd   = min(n, d);

        #pragma omp parallel for num_threads(threads) schedule(guided)
        for (int r = rStart; r <= rEnd; ++r) {
            int c = d - r + 1;
            if (c < 1 || c > m) continue;

            int diag = H[r-1][c-1] + scorePair(A[r-1], B[c-1]);
            int up   = H[r-1][c] + GAP_VAL;
            int left = H[r][c-1] + GAP_VAL;

            H[r][c] = max({0, diag, up, left});
        }
    }

    double t1 = omp_get_wtime();
    timeTaken = t1 - t0;
}

/* -------- Row Parallel -------- */
void smithRow(const string& A, const string& B,
              int threads, double &timeTaken)
{
    int n = A.size(), m = B.size();
    vector<vector<int>> M(n+1, vector<int>(m+1, 0));

    double t0 = omp_get_wtime();

    for (int i = 1; i <= n; ++i) {
        #pragma omp parallel for num_threads(threads)
        for (int j = 1; j <= m; ++j) {

            int diag = M[i-1][j-1] + scorePair(A[i-1], B[j-1]);
            int up   = M[i-1][j] + GAP_VAL;
            int left = M[i][j-1] + GAP_VAL;

            M[i][j] = max({0, diag, up, left});
        }
    }

    double t1 = omp_get_wtime();
    timeTaken = t1 - t0;
}

/* -------- Result Printer -------- */
void showStats(const vector<int>& T,
               const vector<double>& times,
               const string& label)
{
    cout << "\n[" << label << "]\n";
    cout << left << setw(10) << "Thr"
         << setw(16) << "Time"
         << setw(14) << "Speedup"
         << "Eff(%)\n";
    cout << string(45,'-') << "\n";

    double base = times[0];
    for (size_t i = 0; i < times.size(); ++i) {
        double sp = base / times[i];
        double ef = (sp / T[i]) * 100.0;

        cout << setw(10) << T[i]
             << setw(16) << fixed << setprecision(6) << times[i]
             << setw(14) << setprecision(2) << sp
             << setprecision(1) << ef << "\n";
    }
}

int main() {

    const int SIZE = 500;

    string dnaA = generateDNA(SIZE, 7);
    string dnaB = generateDNA(SIZE, 21);

    /* add similarity block */
    for (int i = 100; i < 150; ++i)
        dnaB[i] = dnaA[i];

    cout << "Parallel Smith-Waterman Test\n";
    cout << "Length = " << SIZE << "\n\n";

    vector<int> threadList = {1,2,4,8,12,16};
    int maxT = omp_get_max_threads();

    vector<double> waveTimes, rowTimes;

    for (int t : threadList) {
        if (t > maxT) continue;
        double tt;
        smithWave(dnaA, dnaB, t, tt);
        waveTimes.push_back(tt);
    }

    showStats(threadList, waveTimes, "Wavefront");

    for (int t : threadList) {
        if (t > maxT) continue;
        double tt;
        smithRow(dnaA, dnaB, t, tt);
        rowTimes.push_back(tt);
    }

    showStats(threadList, rowTimes, "Row Method");

    return 0;
}

