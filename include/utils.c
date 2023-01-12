#include <stdio.h>
#include "utils.h"

int check_state=0;
struct timeval tic_g, toc_g;


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


void checkpoint(void){
    if (check_state == 0){
        gettimeofday(&tic_g, NULL);
        check_state = 1;
    } else {
        gettimeofday(&toc_g, NULL);
        check_state = 0;
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


void print_elapsed(void)
{
    int sec, msec, usec, x;

    sec = toc_g.tv_sec - tic_g.tv_sec;
    usec = toc_g.tv_usec - tic_g.tv_usec;
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


// void print_elapsed(struct timeval start_t)
// {
//     int sec, msec, usec, x;
//     struct timeval end_t;

//     gettimeofday(&end_t, NULL);


//     sec = end_t.tv_sec - start_t.tv_sec;
//     usec = end_t.tv_usec - start_t.tv_usec;
//     x = usec / 1e3;
//     msec = x;
//     usec -= x * 1e3;

//     if (usec < 0){
//         msec -= 1;
//         usec += 1e3;
//     }
//     if (msec < 0){
//         sec -= 1;
//         msec += 1e3;
//     }

//     printf("elapsed time = %ds %dms %dus\n", sec, msec, usec);

// }


void print_variable(double *x, int n_print_node)
{
    for (int n=0; n<n_print_node; n++){
        fprintf(stderr, "%5.2f, ", x[n]);
    }
    fprintf(stderr, "\n");
}


/* vector function */
double *copy_array(int N, double *arr){
    double *arr_cp = (double*) malloc(sizeof(double) * N);
    memcpy(arr_cp, arr, sizeof(double) * N);
    return arr_cp;
}


static void *realloc_check(int target_size, void *arr){
    void *ptr = realloc(arr, target_size);
    if (ptr == NULL){
        fprintf(stderr, "Re-allocating error! target size: %d\n", target_size);
        return NULL;
    }
    return ptr;
}


void append_double(double **arr, int id, double value){
    if (id % _block_size == 0){
        *arr = (double*) realloc_check(sizeof(double) * (id + _block_size), *arr);
    }
    (*arr)[id] = value;
}


void append_int(int **arr, int id, int value){
    if (id % _block_size == 0){
        *arr = (int*) realloc_check(sizeof(int) * (id + _block_size), *arr);
    }
    (*arr)[id] = value;
}


void set_index_obj(index_t *idxer, int num_index, int max_ind[]){
    idxer->nstep = 0;
    idxer->num_id = num_index;
    idxer->len = 1;
    for (int n=0; n<num_index; n++){
        idxer->id[n] = 0;
        idxer->id_max[n] = max_ind[n];
        idxer->len *= max_ind[n];
    }
}


void next_index(index_t *idxer){

    if (idxer->nstep+1 == idxer->len){
        printf("Index out of range\n");
        return;
    }

    int nid = idxer->num_id-1;
    idxer->nstep++;
    idxer->id[nid]++;
    while (idxer->id[nid] >= idxer->id_max[nid]){
        idxer->id[nid] -= idxer->id_max[nid];
        idxer->id[nid-1]++;
        nid--;
    }
}


void update_index(index_t *idxer, int nstep){
    if (nstep >= idxer->len){
        printf("Out of range\n");
        return;
    }

    idxer->nstep = nstep;
    int div = idxer->len;
    for (int n=0; n<idxer->num_id; n++){
        div /= idxer->id_max[n];
        idxer->id[n] = nstep / div;
        nstep -= idxer->id[n] * div;
    }
}


double *linspace(double x0, double x1, int len_x){
    double *x = (double*) malloc(sizeof(double) * len_x);
    if (len_x == 1){
        // printf("Too few length selected. x is set to %.2f\n", x0);
        x[0] = x0;
        return x;
    }
    for (int n=0; n<len_x; n++){
        x[n] = n*(x1-x0)/(len_x-1)+x0;
    }
    return x;
}