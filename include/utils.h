#ifndef _UTIL
#define _UTIL

#include <sys/time.h>
#include <string.h>

#define _len_progbar 50

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

#endif