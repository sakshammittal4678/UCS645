#include <stdio.h>
#include <omp.h>

double A[1000][1000], B[1000][1000], C[1000][1000];

int main()
{
    int N=1000;


    int i, j, k;
    double st, et, seq_duration;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = i;
            B[i][j] = i*2;
            C[i][j] = 0.0;
        }
    }

    printf("Sequential:\n");

    st = omp_get_wtime();

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    et = omp_get_wtime();
    seq_duration = et - st;

    printf("Sequential Time = %f\n", seq_duration);


    printf("\n1D Parallel:\n");

    for (int threads = 2; threads <= 20; threads++) {

        double par_duration;

        omp_set_num_threads(threads);

        /* reset C */
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                C[i][j] = 0.0;

        st = omp_get_wtime();

        #pragma omp parallel for private(j, k)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        et = omp_get_wtime();
        par_duration = et - st;
        printf("Threads=%d  Time=%f  Speedup=%.2f\n", threads, par_duration, seq_duration / par_duration);
    }


    printf("\n2D Parallel:\n");

    for (int threads = 2; threads <= 20; threads++) {

        double par_duration;

        omp_set_num_threads(threads);

        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                C[i][j] = 0.0;

        st = omp_get_wtime();

        #pragma omp parallel for collapse(2) private(k)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        et = omp_get_wtime();
        par_duration = et - st;
        printf("Threads=%d  Time=%f  Speedup=%.2f\n", threads, par_duration, seq_duration/par_duration);
    }

    return 0;
}