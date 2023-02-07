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
    set_seed(time(NULL) * world_size * 5 + world_rank*2);
    for_mpi(idxer.len, run, &idxer);
    // run(0, &idxer);
    // for (int n=world_rank; n<idxer.len; n+=world_size){
    //     run(n, &idxer);
    // }

    if (world_rank == 0){
        gettimeofday(&toc, NULL);
        printf("Simulation done, total elapsed: %.3f hour\n", get_dt(tic, toc)/3600.);
    }

    end_simulation();
}

double *p_exc, *nu_ext, *id_set;

double p_gamma=0.03;


double a_set[2] = {4, 2.8};
double b_set[2] = {3, 4.5};
double tr_set[2] = {0.5, 1};
double td_set[2] = {2, 6};


void set_control_parameters(){

    int nitr = 4;
    int max_len[] = {30, 30, 2, nitr};
    int num_controls = GetIntSize(max_len);
    set_index_obj(&idxer, num_controls, max_len);

    p_exc  = linspace(0.01, 0.2, max_len[0]);
    nu_ext = linspace(700, 8000, max_len[1]);
    id_set = linspacee(0, 1, max_len[2]);

    if (world_rank == 0){
        FILE *fp = open_file_wdir(fdir, "control_params.txt", "w");
        for (int n=0; n<num_controls; n++){
            fprintf(fp, "%d,", max_len[n]);
        }
        fprintf(fp, "\n");

        print_arr(fp, "p_exc", max_len[0], p_exc);
        print_arr(fp, "nu_ext", max_len[1], nu_ext);
        print_arr(fp, "id_set", max_len[2], id_set);
        fclose(fp);
    }
}


void end_simulation(){
    free(p_exc);
    free(nu_ext);
    free(id_set);
    end_mpi();
}


int flag_eq = 0;
void run(int job_id, void *idxer_void){

    nn_info_t info = set_info();
    index_t *idxer = (index_t*) idxer_void;
    update_index(idxer, job_id);
    print_job_start(job_id, idxer->len);

    double pe = p_exc[idxer->id[0]];
    int id = id_set[idxer->id[2]];
    double p_alpha = a_set[id];
    double p_beta = b_set[id];
    double tr = tr_set[id];
    double td = td_set[id];

    info.p_out[0][0] = pe;
    info.p_out[0][1] = pe;
    info.p_out[1][1] = p_beta * pe;
    info.p_out[1][0] = p_beta * pe;

    info.w[0][0] = p_gamma * sqrt(p_exc[0])/sqrt(pe);
    info.w[0][1] = info.w[0][0];
    info.w[1][1] = p_alpha * info.w[0][0];
    info.w[1][0] = info.w[1][1];

    info.nu_ext_mu = nu_ext[idxer->id[1]];

    info.taur[1] = tr;
    info.taud[1] = td;

    char fname_info[100];
    sprintf(fname_info, "id%06d_info.txt", job_id);
    write_info(&info, path_join(fdir, fname_info));

    build_ei_rk4(&info);

    int nmax = tmax/_dt;
    #ifdef _debug
    extern desyn_t syns[MAX_TYPE];
    print_syn_network(&syns[0], path_join(fdir, "ntk_e_debug.txt"));
    print_syn_network(&syns[1], path_join(fdir, "ntk_i_debug.txt"));
    write_info(&info, path_join(fdir, "info_debug.txt"));

    progbar_t bar;
    init_progressbar(&bar, nmax);
    #endif

    init_measure(info.N, nmax, 2, NULL);
    add_checkpoint(0);
    for (int nstep=0; nstep<nmax; nstep++){
        if ((flag_eq == 0) && (nstep * _dt >= teq)){
            add_checkpoint(nstep);
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
    nn_info_t info = init_build_info(N, 2);

    info.p_out[0][0] = 0.005;
    info.p_out[0][1] = 0.005;
    info.p_out[1][0] = 0.05;
    info.p_out[1][1] = 0.05;

    info.w[0][0] = 0.05; //
    info.w[0][1] = 0.05; //
    info.w[1][0] = 0.1;
    info.w[1][1] = 0.1;

    info.taur[0] = 0.3;
    info.taud[0] = 1;
    // info.taur[1] = 0.5;
    // info.taud[1] = 2;
    info.taur[1] = 1;
    info.taud[1] = 6;

    info.t_lag = 0.;
    info.nu_ext_mu = 1000;
    info.nu_ext_sd = 0;
    info.w_ext_mu = 0.002;
    info.w_ext_sd = 0;
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
