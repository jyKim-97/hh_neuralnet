#include <stdio.h>
#include "utils.h"


void init_progressbar(progbar_t *bar, int max_step)
{
    bar->max_step = max_step;
    bar->div = max_step/_len_progbar;
    gettimeofday(&(bar->tic), NULL);

    fprintf(stderr, "[");
    for (int n=0; n<_len_progbar; n++){
        fprintf(stderr, " ");
    }
    fprintf(stderr, "] ( )");
}


void progressbar(progbar_t *bar, int nstep)
{
    if ((nstep+1)%(bar->div) == 0){
        fprintf(stderr, "\r[");
        int i, nbar = (nstep+1)/(bar->div);
        for (i=0; i<nbar; i++){
            fprintf(stderr, "=");
        }
        for (i=nbar; i<_len_progbar; i++){
            fprintf(stderr, " ");
        }
        fprintf(stderr, "]");
        // get time
        struct timeval toc;
        double dt, pred_end_time;

        gettimeofday(&toc, NULL);
        dt = get_dt(bar->tic, toc);
        pred_end_time = dt * ((double) bar->max_step) / ((double) nstep);

        fprintf(stderr, "(%5.1fs / %5.1fs)", dt, pred_end_time);
        fprintf(stderr, "\r");
    }
}


double get_dt(struct timeval tic, struct timeval toc)
{
    int sec, msec;

    sec = toc.tv_sec - tic.tv_sec;
    msec = (toc.tv_usec - tic.tv_usec)/1e3;

    if (msec < 0){
        sec -= 1;
        msec += 1e3;
    }

    double dt = sec + ((double) msec) * 1e-3;

    return dt;
}


void print_elapsed(struct timeval start_t)
{
    int sec, msec, usec, x;
    struct timeval end_t;

    gettimeofday(&end_t, NULL);


    sec = end_t.tv_sec - start_t.tv_sec;
    usec = end_t.tv_usec - start_t.tv_usec;
    x = usec / 1e3;
    msec = x;
    usec -= x * 1e3;

    if (usec < 0){
        msec -= 1;
        usec += 1e3;
    }
    if (msec < 0){
        sec -= 1;
        msec += 1e3;
    }

    printf("elapsed time = %ds %dms %dus\n", sec, msec, usec);

}


void print_variable(double *x, int n_print_node)
{
    for (int n=0; n<n_print_node; n++){
        fprintf(stderr, "%5.2f, ", x[n]);
    }
    fprintf(stderr, "\n");
}


// utility functions for HH net
