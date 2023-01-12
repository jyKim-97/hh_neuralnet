#ifndef _UTIL_H
#define _UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/time.h>
#include <string.h>
#include <stdlib.h>

#define _len_progbar 50
#define _block_size 500
#define MAX_IND_NUM 10
#define CHECKPOINT(tic) clock_gettime(CLOCK_MONOTONIC, &tic) 


typedef struct _progbar_t {

    int max_step;
    int div;
    struct timeval tic;

} progbar_t;


typedef struct _index_t {
    int nstep;
    int num_id;
    int len;
    int id[MAX_IND_NUM];
    int id_max[MAX_IND_NUM];
} index_t;


void init_progressbar(progbar_t *bar, int max_step);
void progressbar(progbar_t *bar, int nstep);
double get_dt(struct timeval tic, struct timeval toc);
// void print_elapsed(struct timeval start_t);
void checkpoint(void);
void print_elapsed(void);

void set_index_obj(index_t *idxer, int num_index, int max_ind[]);
void next_index(index_t *idxer);
void update_index(index_t *idxer, int nstep);

// maintaining code
void print_variable(double *x, int n_print_node);
double *copy_array(int N, double *arr);
void append_double(double **arr, int id, double value);
void append_int(int **arr, int id, int value);
double *linspace(double x0, double x1, int len_x);

#ifdef __cplusplus
}
#endif


#endif



