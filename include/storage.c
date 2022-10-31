#include "storage.h"

#define save_as_float
// #define _fs_storage 2000

// const double _fs_storage = 2000;
#define _fs_save 2000
// int _nstep_save = 1000./_fs_storage/_dt;
int _nstep_save = 1000./_fs_save/_dt;

#include <stdio.h>
void save(int N, int nstep, double* arr, FILE *fp){
    if (nstep % _nstep_save != 0) return;

    #ifdef save_as_float
    float *arr_f = (float*) malloc(sizeof(float) * N);
    for (int n=0; n<N; n++) arr_f[n] = (float) arr[n];
    fwrite(arr_f, sizeof(float), N, fp);
    free(arr_f);
    #else
    fwrite(arr, sizeof(double), N, fp);
    #endif    
}

// NOTE: file open할 때 기존에 있는 파일 체크하는 코드 필요
FILE *open_file(const char *fname, const char *option){
    // check file exists
    FILE *fp = fopen(fname, "r");
    if (fp == NULL){
        fprintf(stderr, "File %s exists\n", fname);
        return NULL;
    }
    fclose(fp);

    fp = fopen(fname, option);
    return fp;
}


