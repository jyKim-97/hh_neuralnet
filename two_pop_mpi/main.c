#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "utils.h"
#include "storage.h"
#include "model2.h"
#include "neuralnet.h"
#include "measurement2.h"
#include "mpifor.h"


extern double _dt;
extern wbneuron_t neuron;
// #define _debug
#define GetIntSize(arr) sizeof(arr)/sizeof(int)

void run(int job_id, void *idxer_void);
nn_info_t set_info(void);
void set_control_parameters();
void end_simulation();
void print_arr(FILE *fp, char *arr_name, int arr_size, double *arr);


int N = 1000;
double iapp = 0;
double tmax = 2500;
double teq = 500;
char fdir[100] = "./tmp";
index_t idxer;

extern int world_size, world_rank;

int main(int argc, char **argv){
    init_mpi(&argc, &argv);
    if (argc == 2){ sprintf(fdir, "%s", argv[1]); }

    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
    set_control_parameters();
    set_seed(world_rank * 100);
    for_mpi(idxer.len, run, &idxer);
    // run(0, &idxer);

    if (world_rank == 0){
        gettimeofday(&toc, NULL);
        printf("Simulation done, total elapsed: %.3f hour", get_dt(tic, toc)/3600.);
    }

    end_simulation();
}

double *g_inh, *p_inh, *alpha, *beta, *nu_ext;
// alpha: gE = alpha * gI
// beta:  pE = beta * gE
void set_control_parameters(){

    int nitr = 2;
    int max_len[6] = {10, 10, 5, 5, 4, nitr};
    int num_controls = GetIntSize(max_len);
    set_index_obj(&idxer, num_controls, max_len);

    alpha  = linspace(   0.,  0.1, max_len[0]);
    beta   = linspace(   0.,   1., max_len[1]);
    g_inh  = linspace(0.001,  0.5, max_len[2]);
    p_inh  = linspace( 0.01, 0.95, max_len[3]);
    nu_ext = linspace(  500, 4000, max_len[4]);

    if (world_rank == 0){
        FILE *fp = open_file_wdir(fdir, "control_params.txt", "w");
        for (int n=0; n<GetIntSize(max_len); n++){
            fprintf(fp, "%d,", max_len[n]);
        }
        fprintf(fp, "\n");

        print_arr(fp, "alpha",  max_len[0], alpha);
        print_arr(fp, "beta",   max_len[1], beta);
        print_arr(fp, "g_inh",  max_len[2], g_inh);
        print_arr(fp, "p_inh",  max_len[3], p_inh);
        print_arr(fp, "nu_ext", max_len[4], nu_ext);
        fclose(fp);

        nn_info_t info = set_info();
        write_info(&info, path_join(fdir, "info.txt"));
    }
}


void end_simulation(){
    free(alpha);
    free(beta);
    free(g_inh);
    free(p_inh);
    free(nu_ext);
    end_mpi();
}


int flag_eq = 0;
void run(int job_id, void *idxer_void){

    nn_info_t info = set_info();
    index_t *idxer = (index_t*) idxer_void;
    update_index(idxer, job_id);

    // alpha: gE = alpha * gI
    // beta:  pE = beta * gE
    info.w[1][0] = g_inh[idxer->id[2]];
    info.w[1][1] = info.w[1][0];
    info.p_out[1][0] = p_inh[idxer->id[3]];
    info.p_out[1][1] = info.p_out[1][0];
    info.nu_ext = nu_ext[idxer->id[4]];

    info.w[0][0] = alpha[idxer->id[0]] * info.w[1][0];
    info.w[0][1] = info.w[0][0];
    info.p_out[0][0] = beta[idxer->id[1]] * info.p_out[1][0];
    info.p_out[0][1] = info.p_out[0][0];

    build_ei_rk4(&info);
    print_job_start(job_id, idxer->len);

    int nmax = tmax/_dt;
    #ifdef _debug
    extern desyn_t syns[MAX_TYPE];
    print_syn_network(&syns[0], path_join(fdir, "ntk_e_debug.txt"));
    print_syn_network(&syns[1], path_join(fdir, "ntk_i_debug.txt"));
    write_info(&info, path_join(fdir, "info_debug.txt"));

    progbar_t bar;
    init_progressbar(&bar, nmax);
    #endif

    init_measure(N, nmax, 2, NULL);
    for (int nstep=0; nstep<nmax; nstep++){
        if ((flag_eq == 0) && (nstep * _dt >= teq)){
            flush_measure();
            flag_eq = 1;
        }

        update_rk4(nstep, iapp);
        measure(nstep, &neuron);

        #ifdef _debug
        progressbar(&bar, nstep);
        #endif
    }
    summary_t obj = flush_measure();

    char fname_res[100];
    sprintf(fname_res, "id%06d_result.txt", job_id);
    export_result(&obj, path_join(fdir, fname_res));

    char fname_spk[100];
    sprintf(fname_spk, "id%06d_spk.dat", job_id);
    export_spike(path_join(fdir, fname_spk));

    char fname_lfp[100];
    sprintf(fname_lfp, "id%06d_lfp.dat", job_id);
    export_lfp(path_join(fdir, fname_lfp));

    print_job_end(job_id, idxer->len);
}


nn_info_t set_info(void){
    // nn_info_t info = {0,};
    nn_info_t info = get_empty_info();

    info.N = N;
    info.num_types = 2;
    info.type_range[0] = info.N * 0.8;
    info.type_range[1] = info.N;

    info.p_out[0][0] = 0.;
    info.p_out[0][1] = 0.;
    info.p_out[1][0] = 0.01;
    info.p_out[1][1] = 0.01;

    info.w[0][0] = 0.;
    info.w[0][1] = 0.;
    info.w[1][0] = 0.001;
    info.w[1][1] = 0.001;

    info.t_lag = 0.5;
    info.nu_ext = 500;
    info.w_ext  = 0.001;
    info.const_current = false;

    return info;
}


void print_arr(FILE *fp, char *arr_name, int arr_size, double *arr){
    fprintf(fp, "%s:", arr_name);
    for (int n=0; n<arr_size; n++){
        fprintf(fp, "%f,", arr[n]);
    }
    fprintf(fp, "\n");
}
