#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <iomanip>

using namespace std;

/* ---- Simulation Parameters ---- */
const double EPSILON = 1.0;
const double SIGMA   = 1.0;
const double CUTOFF  = 2.5 * SIGMA;
const double CUTOFF2 = CUTOFF * CUTOFF;

struct Particle {
    double x, y, z;     // position
    double fx, fy, fz;  // force
};

/* ---- Lennard-Jones Pair Interaction ---- */
inline double lj_pair(
    const Particle& a,
    const Particle& b,
    double& fx, double& fy, double& fz)
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;

    double r2 = dx*dx + dy*dy + dz*dz;
    if (r2 <= 0.0 || r2 > CUTOFF2) {
        fx = fy = fz = 0.0;
        return 0.0;
    }

    double inv_r2 = 1.0 / r2;
    double s2 = SIGMA * SIGMA * inv_r2;
    double s6 = s2 * s2 * s2;
    double s12 = s6 * s6;

    double potential = 4.0 * EPSILON * (s12 - s6);
    double coeff = 24.0 * EPSILON * inv_r2 * (2.0 * s12 - s6);

    fx = coeff * dx;
    fy = coeff * dy;
    fz = coeff * dz;

    return potential;
}

int main() {

    const int N = 1000;
    vector<Particle> P(N);

    /* ---- Random Initialization ---- */
    mt19937 gen(42);
    uniform_real_distribution<double> dist(0.0, 80.0);

    for (auto &p : P) {
        p.x = dist(gen);
        p.y = dist(gen);
        p.z = dist(gen);
        p.fx = p.fy = p.fz = 0.0;
    }

    vector<int> threadList = {1,2,4,8,12,16};

    cout << "\nLennard-Jones Force Simulation\n";
    cout << "Particles: " << N << "\n\n";

    cout << left
         << setw(10) << "Thr"
         << setw(14) << "Time(s)"
         << setw(14) << "Speedup"
         << setw(14) << "Eff(%)"
         << "Energy\n";
    cout << string(65, '-') << "\n";

    double baseTime = 0.0;

    for (int T : threadList) {
        if (T > omp_get_max_threads()) continue;

        for (auto &p : P)
            p.fx = p.fy = p.fz = 0.0;

        double totalEnergy = 0.0;
        double start = omp_get_wtime();

        #pragma omp parallel num_threads(T) reduction(+:totalEnergy)
        {
            #pragma omp for schedule(guided)
            for (int i = 0; i < N; ++i) {

                double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;

                for (int j = i + 1; j < N; ++j) {

                    double fx, fy, fz;
                    double pe = lj_pair(P[i], P[j], fx, fy, fz);

                    fx_acc += fx;
                    fy_acc += fy;
                    fz_acc += fz;

                    #pragma omp atomic
                    P[j].fx -= fx;
                    #pragma omp atomic
                    P[j].fy -= fy;
                    #pragma omp atomic
                    P[j].fz -= fz;

                    totalEnergy += pe;
                }

                P[i].fx += fx_acc;
                P[i].fy += fy_acc;
                P[i].fz += fz_acc;
            }
        }

        double end = omp_get_wtime();
        double elapsed = end - start;

        if (T == 1) baseTime = elapsed;

        double sp = baseTime / elapsed;
        double eff = (sp / T) * 100.0;

        cout << setw(10) << T
             << setw(14) << fixed << setprecision(6) << elapsed
             << setw(14) << setprecision(2) << sp
             << setw(13) << setprecision(1) << eff << "%"
             << setprecision(3) << totalEnergy << "\n";
    }

    cout << string(65, '-') << "\n";
    return 0;
}

