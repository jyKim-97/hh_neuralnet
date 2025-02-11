#include "rng.h"

#ifdef USE_MKL
VSLStreamStatePtr stream = NULL;
#endif

void set_seed(long seed){
    init_genrand64(seed);
    #ifdef USE_MKL
    vslNewStream(&stream, VSL_BRNG_MT2203, seed);
    #endif
}


void set_seed_by_id(long seed, int rng_id){
    init_genrand64_by_id(seed, rng_id);
}


double *exp_lambda = NULL;
void init_exp_lambda(int N, const double *lambda){
    exp_lambda = (double*) malloc(sizeof(double) * N);
    for (int n=0; n<N; n++){
        exp_lambda[n] = exp(-lambda[n]);
    }
}


int *get_poisson_array_single(int N, const double *lambda){
    if (exp_lambda == NULL){
        init_exp_lambda(N, lambda);
    }

    int *poisson_arr = (int*) malloc(sizeof(int) * N);
    for (int n=0; n<N; n++){
        poisson_arr[n] = pick_random_poisson(exp_lambda[n]);
    }
    return poisson_arr;
}


int pick_random_poisson(double exp_l){
    double p = 1.0;
    int step = 0;
    while (p > exp_l){
        p *= genrand64_real2();
        step++;
    }
    return step==0? 0: step-1;
}


void free_poisson(void){
    free(exp_lambda);
    exp_lambda = NULL;
}


#ifdef USE_MKL

int *get_poisson_array_mkl(int N, const double *lambda){
    if (stream == NULL){
        printf("Seed is not generated!\n");
        exit(1);
    }

    int *poisson_arr = (int*) malloc(sizeof(int) * N);
    viRngPoissonV(VSL_RNG_METHOD_POISSONV_POISNORM, stream, N, poisson_arr, lambda);
    return poisson_arr;
}


void end_stream(){
    vslDeleteStream(&stream);
}

#endif