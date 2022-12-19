#include "storage.h"

#define save_as_float
// #define _fs_storage 2000

// const double _fs_storage = 2000;
double _fs_save = 2000;
extern double _dt;
int _nstep_save = -1;

#include <stdio.h>
void save(int N, int nstep, double* arr, FILE *fp){
    if (_nstep_save == -1){
        if (_fs_save == -1){
            _nstep_save = 1;
        } else {
            _nstep_save = 1000./_fs_save/_dt;
        }
    }
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


void change_sampling_rate(double fs){
    // if fs = -1 -> save all data
    _fs_save = fs;
}

// NOTE: file open할 때 기존에 있는 파일 체크하는 코드 필요
FILE *open_file(const char *fname, const char *option){

    // printf("n_step_save: %d, fs: %d, dt: %f\n", _nstep_save, _fs_save, _dt);

    // check file exists
    FILE *fp = fopen(fname, "r");
    if (fp != NULL){
        fprintf(stderr, "File %s exists\n", fname);
        fclose(fp);
        return NULL;
    }

    fp = fopen(fname, option);
    return fp;
}


