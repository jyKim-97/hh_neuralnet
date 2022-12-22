#ifndef _storage
#define _storage

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
 
void save(int N, int nstep, double* arr, FILE *fp);
void change_sampling_rate(double fs);
FILE *open_file(const char *fname, const char *option);
void write_signal_d(int len, double *arr, FILE *fp);
void write_signal_f(int len, float *arr, FILE *fp);

#endif