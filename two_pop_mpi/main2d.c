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

//144, 148
extern double _dt;
extern nnpop_t nnpop;

// #define _debug
#define GetIntSize(arr) sizeof(arr)/sizeof(int)

void run(int job_id, void *idxer_void);
nn_info_t set_info(void);
void set_control_parameters();
static void end_simulation();
nn_info_t allocate_setting(int job_id, index_t *idxer);
static void print_arr(FILE *fp, char *arr_name, int arr_size, double *arr);
void allocate_multiple_ext(nn_info_t *info);


double iapp = 0;
char fdir[100] = "./tmp";

// parameter settinge
int N = 1000;
double taur_set[] = {0.3, 0.5, 0.3, 1}; // E(F), I(F), E(S), I(S)
double taud_set[] = {1, 2.5, 1, 8}; // E(F), I(F), E(S), I(S)
double pe_range[4] = {0.005, 0.3, 0.005, 0.15};
double nu_range[2] = {1000, 8000};

double a_set[2] = {9.4, 4.5};
double b_set[2] = {3, 2.5};
double c = 0.01;

// simulation setting
int ptype = 0; // F(0) or S(1)
int nitr = 15;
int max_len[] = {31, 31, 0}; // pE, nu, nitr
double tmax = 5500;
double teq = 500;

// simulation parameters
index_t idxer;
double *p_exc, *nu_ext;

static void read_args(int argc, char **argv);


extern int world_size, world_rank;

int main(int argc, char **argv){
    #ifndef _debug
    init_mpi(&argc, &argv);
    read_args(argc, argv);
    set_seed(time(NULL) * world_size * 5 + world_rank*2);
    #else
    ignore_exist_file(true);
    set_seed(20000);
    read_args(argc, argv);
    _dt = 0.01;
    tmax = 4000;
    #endif

    if (world_rank == 0) fprintf(stderr, "Start simulation settting...");
    set_control_parameters();

    #ifndef _debug
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
    if (world_rank == 0) fprintf(stderr, "Done, start simulation\n");
    for_mpi(idxer.len, run, &idxer);
    if (world_rank == 0){
        gettimeofday(&toc, NULL);
        printf("Simulation done, total elapsed: %.3f hour\n", get_dt(tic, toc)/3600.);
    }
    #else
    run(101, &idxer);
    #endif

    end_simulation();
}


static void read_args(int argc, char **argv){
    int n = 1;
    while (n < argc){
        if (strcmp(argv[n], "-t") == 0){
            tmax = atof(argv[n+1]); n++;
        } else if (strcmp(argv[n], "--fdir") == 0){
            strcpy(fdir, argv[n+1]); n++;
        } else if (strcmp(argv[n], "--type") == 0){
            ptype = atoi(argv[n+1]); n++;
            if ((ptype != 0) && (ptype != 1)){
                printf("Wrong type: %d\n", ptype);
                exit(-1);
            }
        } else if (strcmp(argv[n], "--nitr") == 0){
            nitr = atoi(argv[n+1]); n++;
        } else {
            printf("Wrong argument typed: %s\n", argv[n]);
            exit(-1);
        }
        n++;
    }
}


void set_control_parameters(){

    max_len[2] = nitr;
    int num_controls = GetIntSize(max_len);
    set_index_obj(&idxer, num_controls, max_len);

    p_exc = linspace(pe_range[2*ptype], pe_range[2*ptype+1], max_len[0]);
    nu_ext = linspace(nu_range[0], nu_range[1], max_len[1]);

    if (world_rank == 0){
        FILE *fp = open_file_wdir(fdir, "control_params.txt", "w");
        for (int n=0; n<num_controls; n++){
            fprintf(fp, "%d,", max_len[n]);
        }
        fprintf(fp, "\n");

        print_arr(fp, "p_exc", max_len[0], p_exc);
        print_arr(fp, "nu_ext", max_len[1], nu_ext);
        fclose(fp);
    }
}


int flag_eq = 0;
void run(int job_id, void *idxer_void){

    nn_info_t info = set_info();
    index_t *idxer = (index_t*) idxer_void;
    update_index(idxer, job_id);
    print_job_start(job_id, idxer->len);

    double pe = p_exc[idxer->id[0]];
    double pi = b_set[ptype] * pe;
    double we = c*sqrt(0.01)/sqrt(pe);
    double wi = a_set[ptype] * we;
    double nu = nu_ext[idxer->id[1]];

    info.p_out[0][0] = pe;
    info.p_out[0][1] = pe;
    info.p_out[1][1] = pi;
    info.p_out[1][0] = pi;

    info.w[0][0] = we;
    info.w[0][1] = we;
    info.w[1][0] = wi;
    info.w[1][1] = wi;

    info.nu_ext_mu[0] = nu;

    info.taur[0] = taur_set[2*ptype];
    info.taud[0] = taud_set[2*ptype];
    info.taur[1] = taur_set[2*ptype+1];
    info.taud[1] = taud_set[2*ptype+1];

    char fname_info[100], fout[200];
    sprintf(fname_info, "id%06d_info.txt", job_id);
    path_join(fout, fdir, fname_info);
    write_info(&info, fout);

    build_ei_rk4(&info);

    int nmax = tmax/_dt;
    #ifdef _debug
    extern desyn_t syns[MAX_TYPE];
    path_join(fout, fdir, "info.txt");
    write_info(&info, fout);
    
    path_join(fout, fdir, "ntk_e.txt");
    print_syn_network(&syns[0], fout);

    path_join(fout, fdir, "ntk_i.txt");
    print_syn_network(&syns[1], fout);

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
        measure(nstep, &nnpop);

        #ifdef _debug
        progressbar(&bar, nstep);
        #endif
    }
    summary_t obj = flush_measure();

    char fname_res[100];
    sprintf(fname_res, "id%06d_result.txt", job_id);
    path_join(fout, fdir, fname_res);
    export_result(&obj, fout);

    char fname_spk[100];
    sprintf(fname_spk, "id%06d_spk.dat", job_id);
    path_join(fout, fdir, fname_spk);
    export_spike(fout);

    char fname_lfp[100];
    sprintf(fname_lfp, "id%06d_lfp.dat", job_id);
    path_join(fout, fdir, fname_lfp);
    export_lfp(fout);

    destroy_neuralnet();
    destroy_measure();
    destroy_rng();

    print_job_end(job_id, idxer->len);
}


nn_info_t set_info(void){
    // nn_info_t info = {0,};
    nn_info_t info = init_build_info();
    info.N = N;
    info.pN = 0;

    int ntype = 2;
    int type_id[2] = {E, IF};
    if (ptype == 1){
        type_id[1] = IS;
    }
    int type_range[] = {800, 200};

    set_type_info(&info, ntype, type_id, type_range);

    info.p_out[0][0] = 0.005;
    info.p_out[0][1] = 0.005;
    info.p_out[1][0] = 0.05;
    info.p_out[1][1] = 0.05;

    info.w[0][0] = 0.05; //
    info.w[0][1] = 0.05; //
    info.w[1][0] = 0.1;
    info.w[1][1] = 0.1;

    info.taur[0] = 0;
    info.taud[0] = 0;
    info.taur[1] = 0;
    info.taud[1] = 0;

    info.t_lag = 0.;
    info.nu_ext_mu[0] = 0;
    info.nu_ext_mu[1] = 0;
    info.nu_ext_sd = 0;
    info.w_ext_mu[0] = 0.002;
    info.w_ext_mu[1] = 0;
    info.w_ext_sd = 0;
    info.const_current = false;

    return info;
}


static void end_simulation(){
    free(p_exc);
    free(nu_ext);

    #ifndef _debug
    end_mpi();
    #endif
}



static void print_arr(FILE *fp, char *arr_name, int arr_size, double *arr){
    fprintf(fp, "%s:", arr_name);
    for (int n=0; n<arr_size; n++){
        fprintf(fp, "%f,", arr[n]);
    }
    fprintf(fp, "\n");
}