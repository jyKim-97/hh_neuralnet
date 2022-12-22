#include "storage.h"

#define save_as_float
// #define _fs_storage 2000

// const double _fs_storage = 2000;
double _fs_save = 2000;
extern double _dt;
int _nstep_save = -1;

static void set_nstep();

#include <stdio.h>
void save(int N, int nstep, double* arr, FILE *fp){
    set_nstep();
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


void write_signal_d(int len, double *arr, FILE *fp){
    set_nstep();
    int len_save = len / _nstep_save;
    float *arr_save = (float*) malloc(sizeof(float)*len_save);
    for (int n=0; n<len_save; n++){
        arr_save[n] = (float) arr[n*_nstep_save];
    }
    fwrite(arr_save, sizeof(float), len_save, fp);
    free(arr_save);
}


void write_signal_f(int len, float *arr, FILE *fp){
    set_nstep();
    int len_save = len / _nstep_save;
    float *arr_save = (float*) malloc(sizeof(float)*len_save);
    for (int n=0; n<len_save; n++){
        arr_save[n] = arr[n*_nstep_save];
    }
    fwrite(arr_save, sizeof(float), len_save, fp);
    free(arr_save);
}


static void set_nstep(){
    if (_nstep_save == -1){
        _nstep_save = 1000./_fs_save/_dt;
        if (_fs_save <= 0){
            _nstep_save = 1;
        }
    }
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


