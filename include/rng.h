
#ifndef _RNG
#define _RNG

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mt64.h"

#ifdef USE_MKL
#include "mkl_vsl.h"
#endif

int *get_poisson_array_mkl(int N, const double *lambda);
void end_stream();

// #ifdef _MKL
// #include "mkl_vsl.h"
// int *get_poisson_array_mkl(int N, const double *lambda);
// void end_stream();
//     #define get_poisson_array(N, lambda) get_poisson_array_mkl(N, lambda)
// #else
//     #define get_poisson_array(N, exp_l) get_poisson_array_single(N, exp_l)
// #endif

void set_seed(long seed);
int *get_poisson_array_single(int N, const double *exp_l);
int pick_random_poisson(double exp_l);
void init_exp_lambda(int N, const double *lambda);
void free_poisson(void);

#ifdef __cplusplus
}
#endif

#endif

