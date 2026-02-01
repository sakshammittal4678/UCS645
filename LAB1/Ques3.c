#include <stdio.h>
#include <omp.h>

int main() {
    long int NUM_STEPS=1000000000000000000;
    double step = 1.0 / (double)NUM_STEPS;
    double x, pi, sum;
    double st, et, seq_duration;

    sum = 0.0;

    st = omp_get_wtime();

    for (int i = 0; i < NUM_STEPS; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    pi = step * sum;

    et = omp_get_wtime();
    seq_duration = et - st;

    printf("Sequential Result\n");
    printf("Pi = %.10f\n", pi);
    printf("Time = %f sec\n\n", seq_duration);


    printf("Parallel Results\n");

    for (int threads = 2; threads <= 20; threads++) {

        double par_duration, speedup;
        sum = 0.0;

        omp_set_num_threads(threads);

        st = omp_get_wtime();

        #pragma omp parallel for private(x) reduction(+:sum)
        for (int i = 0; i < NUM_STEPS; i++) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        pi = step * sum;

        et = omp_get_wtime();
        par_duration = et - st;

        printf("Threads = %d   Pi = %.10f   Time = %f   Speedup = %.2f\n",threads, pi, par_duration, seq_duration / par_duration);
    }

    return 0;
}