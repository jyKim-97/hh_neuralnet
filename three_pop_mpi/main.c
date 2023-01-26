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
nn_info_t allocate_setting(int job_id, index_t *idxer);
void print_arr(FILE *fp, char *arr_name, int arr_size, double *arr);
void allocate_multiple_ext(nn_info_t *info);


int N = 2000;
double iapp = 0;
double tmax = 2500;
// double tmax = 1500;
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

    if (world_rank == 0){
        gettimeofday(&toc, NULL);
        printf("Simulation done, total elapsed: %.3f hour\n", get_dt(tic, toc)/3600.);
    }

    end_simulation();
}


double *alpha_set, *beta_set, *id_rank; // alpha_set: pE->?, beta_set: pI->?

/* === Default parameters === */
double a_set[2] = {9.4, 4.5};
double b_set[2] = {3, 2.5};
double c = 0.01;
double d_set[2] = {14142.14, 15450.58};
double plim[][2] = {{0.051, 0.234},
                    {0.028, 0.105}};
/* === Default parameters === */

void set_control_parameters(){

    int nitr = 2;
    int max_len[] = {13, 13, 3, nitr};
    int num_controls = GetIntSize(max_len);
    set_index_obj(&idxer, num_controls, max_len);

    alpha_set = linspace(0, 2, max_len[0]);
    beta_set  = linspace(0, 1, max_len[1]);
    id_rank = linspace(0, 2, max_len[2]);

    if (world_rank == 0){
        FILE *fp = open_file_wdir(fdir, "control_params.txt", "w");
        for (int n=0; n<num_controls; n++){
            fprintf(fp, "%d,", max_len[n]);
        }
        fprintf(fp, "\n");

        print_arr(fp, "alpha_set", max_len[0], alpha_set);
        print_arr(fp, "beta_set", max_len[1], beta_set);
        print_arr(fp, "id_rank", max_len[2], id_rank);
        fclose(fp);
    }
}


void end_simulation(){
    free(alpha_set);
    free(beta_set);
    free(id_rank);
    end_mpi();
}


nn_info_t allocate_setting(int job_id, index_t *idxer){
    update_index(idxer, job_id);

    nn_info_t info = set_info();
    double alpha = alpha_set[idxer->id[0]];
    double beta  = beta_set[idxer->id[1]];
    int rank = id_rank[idxer->id[2]];

    double pe_f, pe_s;
    switch (rank){
        case 0:
            pe_f = plim[0][0];
            pe_s = plim[1][0];
            break;

        case 1:
            pe_f = (plim[0][0]+plim[0][1])/2.;
            pe_s = (plim[1][0]+plim[1][1])/2.;
            break;

        case 2:
            pe_f = plim[0][1];
            pe_s = plim[1][1];
            break;

        default:
            printf("Unexpected rank %d\n", rank);
            exit(1);
            break;
    }

    info.p_out[0][0] = pe_f;
    info.p_out[0][1] = pe_f;
    info.p_out[0][2] = alpha * pe_f;
    info.p_out[0][3] = alpha * pe_f;

    double pi_f = b_set[0] * pe_f;
    info.p_out[1][0] = pi_f;
    info.p_out[1][1] = pi_f;
    info.p_out[1][2] = beta * pi_f;
    info.p_out[1][3] = beta * pi_f;

    info.p_out[2][0] = alpha * pe_s;
    info.p_out[2][1] = alpha * pe_s;
    info.p_out[2][2] = pe_s;
    info.p_out[2][3] = pe_s;

    double pi_s = b_set[1] * pe_s;
    info.p_out[3][0] = beta * pi_s;
    info.p_out[3][1] = beta * pi_s;
    info.p_out[3][2] = pi_s;
    info.p_out[3][3] = pi_s;

    double we_f = c * sqrt(0.01) / sqrt(pe_f);
    info.w[0][0] = we_f;
    info.w[0][1] = we_f;
    info.w[0][2] = sqrt(alpha) * we_f;
    info.w[0][3] = sqrt(alpha) * we_f;

    double wi_f = a_set[0] * we_f;
    info.w[1][0] = wi_f;
    info.w[1][1] = wi_f;
    info.w[1][2] = sqrt(beta) * wi_f;
    info.w[1][3] = sqrt(beta) * wi_f;

    double we_s = c * sqrt(0.01) / sqrt(pe_s);
    info.w[2][0] = sqrt(alpha) * we_s;
    info.w[2][1] = sqrt(alpha) * we_s;
    info.w[2][2] = we_s;
    info.w[2][3] = we_s;

    double wi_s = a_set[1] * we_s;
    info.w[3][0] = sqrt(beta) * wi_s;
    info.w[3][1] = sqrt(beta) * wi_s;
    info.w[3][2] = wi_s;
    info.w[3][3] = wi_s;

    info.taur[0] = 0.5;
    info.taud[0] = 1;

    info.taur[1] = 1;
    info.taud[1] = 2.5;

    info.taur[2] = 0.5;
    info.taud[2] = 1;

    info.taur[3] = 2;
    info.taud[3] = 8;

    info.t_lag = 0.;
    info.const_current = false;

    info.num_ext_types = 2;
    info.nu_ext_multi[0] = d_set[0] * sqrt(pe_f);
    info.nu_ext_multi[1] = d_set[1] * sqrt(pe_s);
    info.w_ext_multi[0] = 0.002;
    info.w_ext_multi[1] = 0.002;

    return info;
}


int flag_eq = 0;
void run(int job_id, void *idxer_void){

    index_t *idxer = (index_t*) idxer_void;
    print_job_start(job_id, idxer->len);
    nn_info_t info = allocate_setting(job_id, idxer);

    char fname_info[100];
    sprintf(fname_info, "id%06d_info.txt", job_id);
    write_info(&info, path_join(fdir, fname_info));

    build_ei_rk4(&info);
    allocate_multiple_ext(&info);

    int nmax = tmax/_dt;
    #ifdef _debug
    extern desyn_t syns[MAX_TYPE];
    print_syn_network(&syns[0], path_join(fdir, "ntk_e_debug.txt"));
    print_syn_network(&syns[1], path_join(fdir, "ntk_i_debug.txt"));
    write_info(&info, path_join(fdir, "info_debug.txt"));

    progbar_t bar;
    init_progressbar(&bar, nmax);
    #endif

    init_measure(info.N, nmax, 4, info.type_range);
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
    nn_info_t info = init_build_info(N, 4);

    // connection strength
    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++){
            info.w[i][j] = 0.;
            info.p_out[i][j] = 0.;
        }
    }

    // synaptic time constant
    info.taur[0] = 0.3;
    info.taur[1] = 0.5;
    info.taur[2] = 1;

    info.taud[0] = 1;
    info.taud[1] = 2.5;
    info.taud[2] = 8;

    info.t_lag = 0.;
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


void allocate_multiple_ext(nn_info_t *info){
    // depends on N
    int nsub = N/2;
    int *target = (int*) malloc(sizeof(int) * nsub);
    
    for (int ntype=0; ntype<2; ntype++){
        for (int i=0; i<nsub; i++){
            target[i] = i + nsub * ntype;
        }
        set_multiple_ext_input(info, ntype, nsub, target);
    }

    free(target);
    check_multiple_input();
}