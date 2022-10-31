#ifndef _UTIL_H
#define _UTIL_H

#include <sys/time.h>
#include <string.h>
#include <stdlib.h>

#define _len_progbar 50
#define _block_size 500

typedef struct _progbar_t {

    int max_step;
    int div;
    struct timeval tic;

} progbar_t;


void init_progressbar(progbar_t *bar, int max_step);
void progressbar(progbar_t *bar, int nstep);
double get_dt(struct timeval tic, struct timeval toc);
void print_elapsed(struct timeval start_t);
// maintaining code
void print_variable(double *x, int n_print_node);

double *copy_array(int N, double *arr);
void *realloc_check(int target_size, void *arr);
void append_double(double **arr, int id, double value);
void append_int(double **arr, int id, int value);

#endif