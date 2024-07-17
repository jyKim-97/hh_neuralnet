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
#include "unistd.h"


// #define TEST

#ifndef TEST
#include "mpifor.h"
extern int world_size, world_rank;
#else
int world_size, world_rank;
#endif

extern double _dt;
extern nnpop_t nnpop;


#define NTYPE 8

// int N = 2000;
double tmax = 10500;
double teq = 500;
nn_info_t *nn_info_set = NULL;

// global var, can be controllled with input args
int nsamples = 0;
char fname_params[200] = "./";
char fdir_out[100] = "./tmp";


void run(int job_id, void *nullarg);
void read_args(int argc, char **argv);
void read_params(char *fname);
int read_params_line(FILE *fp, nn_info_t *info);
void allocate_multiple_ext(nn_info_t *info);
static FILE *open_params_file(char fname[]);
static void print_help(void);

// double taur[] = {0.3, 0.5, 0.3, 1};
// double taud[] = {  1, 2.5,   1, 8};

double taur[] = {0.3, 0.5, 0.3, 1};
double taud[] = {  1, 2.5,   1, 8};
// double taur[] = {0.3, 0.5, 0.3, 0.5};
// double taud[] = {  1, 2.5,   1, 2.5};

// mpicc 
// mpirun -np 100 --hostfile hostfile ./main.out -t 10500 --fdir_out ./data

/* Note that the time constant for the synapse has been changed */

int main(int argc, char **argv){

    /* read args & init parameters */
    #ifndef TEST
    init_mpi(&argc, &argv);
    set_seed(time(NULL) * world_size * 5 + world_rank*2);
    #else
    world_size = 1; world_rank = 0;
    #endif

    ignore_exist_file(true);

    read_args(argc, argv);
    read_params(fname_params);

    #ifndef TEST
    mpi_barrier();
    #endif

    if (world_rank == 0){
        printf("%d samples are found\n", nsamples);
    }

    /* run simulation */
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);

    #ifndef TEST
    for_mpi(nsamples, run, NULL);
    #else
    for (int n=0; n<nsamples; n++){
        run(n, NULL);
    }
    #endif

    if (world_rank == 0){
        gettimeofday(&toc, NULL);
        printf("Simulation done, total elapsed: %.3f hour\n", get_dt(tic, toc)/3600.);
    }

    /* close simulation */
    free(nn_info_set);

    #ifndef TEST
    end_mpi();
    #endif

    return 0;
}


void run(int job_id, void *nullarg){

    nn_info_t info = nn_info_set[job_id];
    int N = info.N;

    char fname_id[100], prefix[150];
    sprintf(fname_id, "id%06d", job_id);
    path_join(prefix, fdir_out, fname_id);

    /// check if the file is exist
    char fname_tmp[200];
    sprintf(fname_tmp, "%s_lfp.dat", prefix);
    if (access(fname_tmp, F_OK) == 0){
        printf("job %d already done, skip this id\n", job_id);
        return;
    }

    // build population
    build_ei_rk4(&info);
    allocate_multiple_ext(&info);
    int nmax=tmax/_dt, neq=teq/_dt;
    int pop_range[2] = {N/2, N};

    // export information file
    char fname_info[200];
    sprintf(fname_info, "%s_info.txt", prefix);
    write_info(&info, fname_info);

    #ifdef TEST
    // export synaptic network 
    for (int n=0; n<8; n++){
        char fname_syn[200];
        sprintf(fname_syn, "%s_syn_%d.txt", prefix, n);
        print_syn_network(nnpop.syns+n, fname_syn);
    }

    // exit(1);

    progbar_t bar;
    init_progressbar(&bar, nmax);
    #endif

    init_measure(N, nmax, 2, pop_range);
    add_checkpoint(0);

    int flag_eq = 0;
    for (int nstep=0; nstep<nmax; nstep++){
        if ((flag_eq == 0) && (nstep == neq)){
            flush_measure();
            flag_eq = 1;
            
            add_checkpoint(nstep); // for monitoring
            add_checkpoint(nstep);
        } 

        if (nstep == 2*neq){ // export monitoring result
            summary_t obj = flush_measure();
            char fname_res[200];
            sprintf(fname_res, "%s_result(monitor).txt", prefix);
            export_result(&obj, fname_res);
        }

        update_rk4(nstep, 0);
        KEEP_SIMUL();
        measure(nstep, &nnpop);

        #ifdef TEST
        progressbar(&bar, nstep);
        #endif

    }

    // export result    
    summary_t obj = flush_measure();
    export_core_result(prefix, &obj);

    destroy_neuralnet();
    destroy_measure();
}


static void print_help(void){
    printf("-t          Simulation time (s)\n");
    printf("--fdir_out  Output directory\n");
    exit(-1);
}


void read_args(int argc, char **argv){

    int n = 1;
    while (n < argc){
        if (strcmp(argv[n], "-t") == 0){
            tmax = atof(argv[n+1]); n++;
        } else if (strcmp(argv[n], "--fdir_out") == 0){
            strcpy(fdir_out, argv[n+1]); n++;
        } else if (strcmp(argv[n], "-h") == 0){
            print_help();
        } else {
            printf("Wrong argument typed: %s\n", argv[n]);
            print_help();
        }
        n++;
    }

    sprintf(fname_params, "%s/params_to_run.txt", fdir_out);
}


void read_params(char *fname){
    // nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * *nline);

    #ifndef TEST

    if (world_rank == 0){
        FILE *fp = open_params_file(fname);

        nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nsamples);
        for (int n=0; n<nsamples; n++){
            if (read_params_line(fp, &(nn_info_set[n])) == 1){
                fprintf(stderr, "Failed to read parameter in line %d\n", n);
                exit(-1);
            }
        }

        fclose(fp);
    }

    MPI_Bcast(&nsamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0){
        nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nsamples);
    }

    mpi_barrier();
    MPI_Bcast((void*) nn_info_set, nsamples*sizeof(nn_info_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    init_nn(nn_info_set[0].N, nn_info_set[0].num_types);

    #else

    FILE *fp = open_params_file(fname);
    nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nsamples);
    for (int n=0; n<nsamples; n++){
        if (read_params_line(fp, &(nn_info_set[n])) == 1){
            fprintf(stderr, "Failed to read parameter in line %d\n", n);
            exit(-1);
        }
    }
    #endif
}


int read_params_line(FILE *fp, nn_info_t *info){

    /* === Parameter input ===
    EF, IF, TF, RF, ES, IS, TS, RS (T: tranasmission, R: receiver)
      0-  7: number of neurons,
      8- 71: projection probability (p_out),
     72-135: weight
    136-137: nu_bck_f, nu_bck_s
    */
    
    // read number of neurons
    double type_range[NTYPE] = {0,};
    for (int n=0; n<NTYPE; n++){
        if (fscanf(fp, "%lf,", &(type_range[n])) == EOF) return 1;
    }

    int N = 0;
    for (int n=0; n<NTYPE; n++) N += (int) type_range[n];

    // initialize information
    *info = init_build_info(N, NTYPE);

    // read probability
    for (int i=0; i<NTYPE; i++){
        for (int j=0; j<NTYPE; j++){
            if (fscanf(fp, "%lf,", &(info->p_out[i][j])) == EOF) return 1;
            if (info->p_out[i][j] < 0) info->p_out[i][j] = ONE2ONE;
        }
    }

    // read weight
    for (int i=0; i<NTYPE; i++){
        for (int j=0; j<NTYPE; j++){
            if (fscanf(fp, "%lf,", &(info->w[i][j])) == EOF) return 1;
        }
    }

    // read background input
    for (int n=0; n<2; n++){
        if (fscanf(fp, "%lf,", &(info->nu_ext_multi[n])) == EOF) return 1;
    }

    // change line
    fscanf(fp, "\n");

    // set default parameters
    int type_id[NTYPE] = {E, IF, E, E, E, IS, E, E};
    double taur_set[NTYPE] = {taur[0], taur[1], taur[0], taur[0], taur[2], taur[3], taur[2], taur[2]};
    double taud_set[NTYPE] = {taud[0], taud[1], taud[0], taud[0], taud[2], taud[3], taud[2], taud[2]};
    for (int n=0; n<NTYPE; n++){
        info->type_range[n] = type_range[n];
        info->type_id[n] = type_id[n];
        info->taur[n] = taur_set[n];
        info->taud[n] = taud_set[n];
    }
    info->num_ext_types = 2;
    info->w_ext_multi[0] = 0.002;
    info->w_ext_multi[1] = 0.002;
    info->t_lag = 0.;
    info->const_current = false;

    return 0;
}


void allocate_multiple_ext(nn_info_t *info){
    // depends on N
    int nsub = (info->N)/2;
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


static FILE *open_params_file(char fname[]){
    FILE *fp = fopen(fname, "r+");
    if (fp == NULL){
        printf("file %s does not exist\n", fname);
        exit(-1);
    }

    if (fscanf(fp, "%d\n", &nsamples) == EOF){
        printf("Falied to read the number of samples. Check the parameter info file\n");
        exit(-1);
    }
    
    return fp;
}