#include "rng.h"

#ifdef _MKL
VSLStreamStatePtr stream;
#endif

void set_seed(long seed){
    init_genrand64(seed);
    #ifdef _MKL
    vslNewStream(&stream, VSL_BRNG_MT2203, seed);
    #endif
}


int *get_poisson_array_single(int N, const double *lambda){
    int *poisson_arr = (int*) malloc(sizeof(int) * N);
    for (int n=0; n<N; n++){
        double p = 1.00;
        int step = 0;
        double exp_l = exp(lambda[n]);
        while (p > exp_l){
            p *= genrand64_real2();
            step++;
        }
        poisson_arr[n] = step==0? 0: step-1;
    }
    return poisson_arr;
}


#ifdef _MKL

int *get_poisson_array_mkl(int N, const double *lambda){
    int *poisson_arr = (int*) malloc(sizeof(int) * N);
    viRngPoissonV(VSL_RNG_METHOD_POISSONV_POISNORM, stream, N, poisson_arr, lambda);
    return poisson_arr;
}


void end_stream(){
    vslDeleteStream(&stream);
}

#endif