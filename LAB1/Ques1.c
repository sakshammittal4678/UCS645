#include<stdio.h>
#include<omp.h>
#include<math.h>

int main(){
    int n=65536;
    double a=10;
    double x[n],y[n];
    double st,et;
    for(int i=0;i<n;i++){
        x[i]=i;
        y[i]=i;
    }
    st=omp_get_wtime();
    for(int i=0;i<n;i++){
        x[i]=a*x[i]+y[i];
    }
    et=omp_get_wtime();
    double seq_duration=et-st;
    printf("Sequential execution time: %f\n",seq_duration);
    for(int threads=2;threads<=20;threads++){
        for(int i=0;i<n;i++){
            x[i]=i;
        }
        omp_set_num_threads(threads);
        st=omp_get_wtime();
        #pragma omp parallel for
        for(int i=0;i<n;i++){
            x[i]=a*x[i]+y[i];
        }
        et=omp_get_wtime();
        double par_duration=et-st;

        printf("Threads: %d  Duration:%f  Speedup:%f\n",threads,par_duration,seq_duration/par_duration);
    }
}