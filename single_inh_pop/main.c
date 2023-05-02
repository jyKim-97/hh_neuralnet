#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "utils.h"
#include "storage.h"
#include "model2.h"
#include "neuralnet.h"
#include "measurement2.h"

#ifdef MULTI
#include "mpifor.h"
#endif

extern double _dt;
extern wbneuron_t neuron;

int N = 1000;
double teq = 500;
double tmax = 2000;
char fdir[100] = "./tmp";

index_t idxer;
#ifdef MULTI
extern int world_rank, world_size;
#endif

void run(nn_info_t info, double iapp, int job_id);
nn_info_t set_info(void);
void print_arr(FILE *fp, char *arr_name, int arr_size, double *arr);
#ifdef MULTI
void multi_param_set();
void multi_run(int job_id, void *idxer_void);
#endif

int main(int argc, char **argv){
    #ifdef MULTI
    init_mpi(&argc, &argv);
    set_seed(time(NULL) * world_size * 5 + world_rank*2);
    #endif

    if (argc > 1){ sprintf(fdir, "%s", argv[1]); }

    #ifdef MULTI
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);

    multi_param_set();
    for_mpi(idxer.len, multi_run, &idxer);

    if (world_rank == 0){
        gettimeofday(&toc, NULL);
        printf("Simulation done, total elapsed: %.3f hour\n", get_dt(tic, toc)/3600.);
    }

    end_mpi();

    #else

    double ic = 1;
    if (argc == 3){ ic = atof(argv[2]); }

    printf("Print result to %s, current input: %.3f\n", fdir, ic);

    nn_info_t info = set_info();
    run(info, ic, -1);
    
    #endif
    
}


#ifdef MULTI
double *w_set, *p_set, *ic_set;
 
void multi_param_set(){

    int nitr = 2;
    int num_controls = 4;
    int max_len[] = {20, 20, 20, nitr};

    w_set = linspace(0.001, 0.5, max_len[0]);
    p_set = linspace(0.001, 0.95, max_len[1]);
    ic_set = linspace(0.2, 3, max_len[2]);
    set_index_obj(&idxer, 4, max_len);

    if (world_rank == 0){
        FILE *fp = open_file_wdir(fdir, "control_params.txt", "w");
        for (int n=0; n<num_controls; n++){
            fprintf(fp, "%d,", max_len[n]);
        }
        fprintf(fp, "\n");

        print_arr(fp, "w_set",  max_len[0], w_set);
        print_arr(fp, "p_set", max_len[1], p_set);
        print_arr(fp, "ic_set", max_len[2], ic_set);
        fclose(fp);
    }

}


void multi_run(int job_id, void *idxer_void){

    index_t *idxer = (index_t*) idxer_void;
    update_index(idxer, job_id);
    print_job_start(job_id, idxer->len);

    // set environment
    nn_info_t info = init_build_info(N, 1);
    info.w[0][0] = w_set[idxer->id[0]];
    info.p_out[0][0] = w_set[idxer->id[1]];
    double ic = ic_set[idxer->id[2]];

    run(info, ic, job_id);
}

#endif


void run(nn_info_t info, double iapp, int job_id){

    int nmax = tmax/_dt;
    int neq  = teq / _dt;
    int flag_eq = 0;
    
    build_ei_rk4(&info);

    int cell_range[3] = {0, 0, N};
    init_measure(N, nmax, 2, cell_range);
    add_checkpoint(0);    

    #ifndef MULTI
    progbar_t bar;
    init_progressbar(&bar, nmax);
    #endif
    for (int nstep=0; nstep<nmax; nstep++){

        if ((flag_eq == 0) && (nstep == neq)){
            flush_measure();
            add_checkpoint(nstep);
            flag_eq = 1;
        }

        update_rk4(nstep, iapp);
        measure(nstep, &neuron);

        #ifndef MULTI
        progressbar(&bar, nstep);
        #endif
    }

    #ifndef MULTI
    printf("\n");

    char fbuf[200];
    path_join(fbuf, fdir, "spk.dat");
    export_spike(fbuf);

    path_join(fbuf, fdir, "lfp.dat");
    export_lfp(fbuf);
    summary_t obj = flush_measure();

    path_join(fbuf, fdir, "result.txt");
    export_result(&obj, fbuf);

    extern desyn_t syns[MAX_TYPE];
    path_join(fbuf, fdir, "ntk.txt");
    print_syn_network(&syns[0], fbuf);

    #else
    char fbuf[200];

    char fname_res[100];
    summary_t obj = flush_measure();
    sprintf(fname_res, "id%06d_result.txt", job_id);
    path_join(fbuf, fdir, fname_res);
    export_result(&obj, fbuf);

    char fname_spk[100];
    sprintf(fname_spk, "id%06d_spk.dat", job_id);
    path_join(fbuf, fdir, fname_spk);
    export_spike(fbuf);

    char fname_lfp[100];
    sprintf(fname_lfp, "id%06d_lfp.dat", job_id);
    path_join(fbuf, fdir, fname_lfp);
    export_lfp(fbuf);

    #endif

    destroy_neuralnet();
}


nn_info_t set_info(void){
    // nn_info_t info = {0,};
    nn_info_t info = init_build_info(N, 1);

    info.p_out[0][0] = 0.8;
    info.w[0][0] = 0.1;

    info.taur[0] = 0.5;
    info.taud[0] = 2.5;

    info.t_lag = 0.;
    info.const_current = true;

    return info;
}


void print_arr(FILE *fp, char *arr_name, int arr_size, double *arr){
    fprintf(fp, "%s:", arr_name);
    for (int n=0; n<arr_size; n++){
        fprintf(fp, "%f,", arr[n]);
    }
    fprintf(fp, "\n");
}
