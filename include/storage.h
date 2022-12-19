#ifndef _storage
#define _storage

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "model.h"
 
void save(int N, int nstep, double* arr, FILE *fp);
void change_sampling_rate(double fs);
FILE *open_file(const char *fname, const char *option);

#endif