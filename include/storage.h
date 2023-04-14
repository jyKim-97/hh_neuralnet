#ifndef _storage
#define _storage

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
 
int is_save_step(int nstep);
void save(int N, int nstep, double* arr, FILE *fp);
void change_sampling_rate(double fs);
void write_signal_d(int len, double *arr, FILE *fp);
void write_signal_f(int len, float *arr, FILE *fp);
// char *path_join(const char *fdir, const char *fname);
void path_join(char fout[], const char *fdir, const char *fname);
FILE *open_file(const char *fname, const char *option);
FILE *open_file_wdir(const char *fdir, const char *fname, const char *option);

#endif