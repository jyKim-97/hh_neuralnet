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
    if (is_save_step(nstep) == 0) return;

    #ifdef save_as_float
    float *arr_f = (float*) malloc(sizeof(float) * N);
    for (int n=0; n<N; n++) arr_f[n] = (float) arr[n];
    fwrite(arr_f, sizeof(float), N, fp);
    free(arr_f);
    #else
    fwrite(arr, sizeof(double), N, fp);
    #endif    
}

// 0 (False), 1 (True)
int is_save_step(int nstep){
    if (nstep % _nstep_save == 0){
        return 1;
    } else {
        return 0;
    }
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


FILE *open_file(const char *fname, const char *option){

    FILE *fp = fopen(fname, "r");
    if (fp != NULL){
        fprintf(stderr, "File %s exists\n", fname);
        fclose(fp);
        return NULL;
    }

    fp = fopen(fname, option);
    return fp;
}


FILE *open_file_wdir(const char *fdir, const char *fname, const char *option){
    char *fbuf = path_join(fdir, fname);
    FILE *fp = open_file(fbuf, option);
    return fp;
}


char *path_join(const char *fdir, const char *fname){
    char buf[200];
    strcpy(buf, fdir);
    int l = (int) strlen(fdir);
    char last = fdir[l-1];
    if (last != '/') strcat(buf, "/");

    strcat(buf, fname);

    // get len
    int len = 0;
    char *c = buf;
    while (*c != '\0'){
        len++;
        c++;
    }

    char *buf_return = (char*) malloc(sizeof(char) * (len+1));
    for (int n=0; n<len; n++){
        buf_return[n] =  buf[n];
    }
    buf_return[len] = '\0';

    return buf_return;
}
