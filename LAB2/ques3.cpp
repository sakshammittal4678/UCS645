#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <algorithm>

using namespace std;

/* ---- Physical Constants ---- */
const double ALPHA = 0.01;
const double DX = 0.1;
const double DY = 0.1;
const double DT = 0.001;

const int SIZE = 512;
const int STEPS = 100;

using Matrix = vector<vector<double>>;

/* -------- Simulator Class -------- */
class PlateSolver {
    int N, T;
    Matrix A, B;

    void seedHeat() {
        int mid = N / 2;
        int radius = N / 12;

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double d = sqrt((i-mid)*(i-mid) + (j-mid)*(j-mid));
                A[i][j] = (d < radius) ? 100.0 : 0.0;
                B[i][j] = 0.0;
            }
        }
    }

    double avgTemp(const Matrix &M) {
        double s = 0;
        for (int i=0;i<N;i++)
            for (int j=0;j<N;j++)
                s += M[i][j];
        return s / (N*N);
    }

public:
    PlateSolver(int n, int steps):N(n),T(steps) {
        A.assign(N, vector<double>(N,0));
        B.assign(N, vector<double>(N,0));
        seedHeat();
    }

    double solve(int threads, string mode, double &avg) {
        double t0 = omp_get_wtime();

        for (int t=0;t<T;t++) {

            if(mode=="static") {
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
                for(int i=1;i<N-1;i++)
                    for(int j=1;j<N-1;j++){
                        double lap =
                          (A[i+1][j]-2*A[i][j]+A[i-1][j])/(DX*DX) +
                          (A[i][j+1]-2*A[i][j]+A[i][j-1])/(DY*DY);
                        B[i][j] = A[i][j] + ALPHA*DT*lap;
                    }
            }
            else if(mode=="dynamic") {
#pragma omp parallel for collapse(2) schedule(dynamic,32) num_threads(threads)
                for(int i=1;i<N-1;i++)
                    for(int j=1;j<N-1;j++){
                        double lap =
                          (A[i+1][j]-2*A[i][j]+A[i-1][j])/(DX*DX) +
                          (A[i][j+1]-2*A[i][j]+A[i][j-1])/(DY*DY);
                        B[i][j] = A[i][j] + ALPHA*DT*lap;
                    }
            }
            else {
#pragma omp parallel for collapse(2) schedule(guided) num_threads(threads)
                for(int i=1;i<N-1;i++)
                    for(int j=1;j<N-1;j++){
                        double lap =
                          (A[i+1][j]-2*A[i][j]+A[i-1][j])/(DX*DX) +
                          (A[i][j+1]-2*A[i][j]+A[i][j-1])/(DY*DY);
                        B[i][j] = A[i][j] + ALPHA*DT*lap;
                    }
            }

            swap(A,B);
        }

        avg = avgTemp(A);
        return omp_get_wtime() - t0;
    }

    double solveBlocked(int threads, double &avg) {
        const int BLOCK=32;
        double t0 = omp_get_wtime();

        for(int t=0;t<T;t++){
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
            for(int bi=1; bi<N-1; bi+=BLOCK)
                for(int bj=1; bj<N-1; bj+=BLOCK){
                    int ei=min(bi+BLOCK,N-1);
                    int ej=min(bj+BLOCK,N-1);

                    for(int i=bi;i<ei;i++)
                        for(int j=bj;j<ej;j++){
                            double lap =
                              (A[i+1][j]-2*A[i][j]+A[i-1][j])/(DX*DX) +
                              (A[i][j+1]-2*A[i][j]+A[i][j-1])/(DY*DY);
                            B[i][j] = A[i][j] + ALPHA*DT*lap;
                        }
                }
            swap(A,B);
        }

        avg = avgTemp(A);
        return omp_get_wtime()-t0;
    }
};

/* -------- Utility Printing -------- */
void showHeader(){
    cout<<"2D Heat Diffusion Simulation\n";
    cout<<"Grid: "<<SIZE<<"x"<<SIZE<<"\n";
    cout<<"Steps: "<<STEPS<<"\n\n";
}

void testSchedules(){
    vector<string> modes={"static","dynamic","guided"};
    vector<int> threads={1,2,4,8,12,16};

    for(auto &m:modes){
        cout<<"["<<m<<" scheduling]\n";
        cout<<left<<setw(10)<<"Thr"<<setw(14)<<"Time"
            <<setw(14)<<"Speedup"<<"Eff%\n";
        cout<<string(45,'-')<<"\n";

        double base=0;
        for(int t:threads){
            if(t>omp_get_max_threads()) continue;

            PlateSolver sim(SIZE,STEPS);
            double avg;
            double tm=sim.solve(t,m,avg);

            if(t==1) base=tm;

            double sp=base/tm;
            double ef=(sp/t)*100;

            cout<<setw(10)<<t
                <<setw(14)<<fixed<<setprecision(5)<<tm
                <<setw(14)<<setprecision(2)<<sp
                <<setprecision(1)<<ef<<"%\n";
        }
        cout<<"\n";
    }
}

void testBlocking(){
    vector<int> threads={1,2,4,8,12,16};

    cout<<"[Cache Blocking]\n";
    cout<<left<<setw(10)<<"Thr"<<setw(14)<<"Time"
        <<setw(14)<<"Speedup"<<"Eff%\n";
    cout<<string(45,'-')<<"\n";

    double base=0;
    for(int t:threads){
        if(t>omp_get_max_threads()) continue;

        PlateSolver sim(SIZE,STEPS);
        double avg;
        double tm=sim.solveBlocked(t,avg);

        if(t==1) base=tm;

        double sp=base/tm;
        double ef=(sp/t)*100;

        cout<<setw(10)<<t
            <<setw(14)<<fixed<<setprecision(5)<<tm
            <<setw(14)<<setprecision(2)<<sp
            <<setprecision(1)<<ef<<"%\n";
    }
}

int main(){
    showHeader();
    testSchedules();
    testBlocking();
    return 0;
}

