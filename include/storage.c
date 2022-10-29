#include "storage.h"

#define save_as_float
#define _fs_storage 2000
// #define CASTTING()

void save(int N, int nstep, double* arr, FILE *fp){
    if ((int) (nstep*_dt) % _fs_storage != 0) return;

    #ifdef save_as_float
    float *arr_f = (float*) malloc(sizeof(float) * N);
    for (int n=0; n<N; n++) arr_f[n] = (float) arr[n];
    fwrite(arr_f, sizeof(float), N, fp);
    free(arr_f);
    #else
    fwrite(arr, sizeof(double), N, fp);
    #endif    
}

