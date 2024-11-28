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

// compile
// make -C ../include main
// 

#define TEST

#ifndef TEST

#include "mpifor.h"
extern int world_size, world_rank;

#else

int world_size=1, world_rank=0;

#endif

#define NTYPE 3 // E, I, Recv
#define NPOP 2 // F, S
#define NTRANS 1 // Trans

extern double _dt;
extern nnpop_t nnpop;

double tmax = 10500;
double teq  = 500;

int nsamples = 0;
char fname_params[200];
char fdir_out[100] = "./tmp";

nn_info_t *nn_info_set = NULL;

double taur[] = {0.3, 0.5, 0.3, 1};
double taud[] = {  1, 2.5,   1, 8};

static void print_help(void);
static void read_args(int argc, char **argv);
static void check_spike_info_files(void);
static void run(int job_id, void *nullarg);
static void allocate_multiple_ext(nn_info_t *info);
static long read_seed(int job_id);
static void allocate_target_spike(int job_id, int num_pcells);
static void print_syn_ntk(const char *prefix);
static void read_params(char *fname);
static int read_params_line(FILE *fp, nn_info_t *info);
static FILE *open_params_file(const char *fname);
#define REPORT_ERROR(msg) print_error(msg, __FILE__, __LINE__)


int main(int argc, char **argv){
    #ifndef TEST
    init_mpi(&argc, &argv);
    set_seed(time(NULL) * world_size * 5 + world_rank*2);
    set_seed(time(NULL) * world_size * 10 + world_rank*3); // for tP neurons
    #else
    set_seed(42);
    set_seed(2000); // for tP neurons
    #endif

    ignore_exist_file(true);

    // read parameters
    read_args(argc, argv);
    read_params(fname_params);
    check_spike_info_files(); // check FILE exist

    #ifndef TEST
    mpi_barrier();
    #endif

    if (world_rank == 0) printf("%d samples are found\n", nsamples);

    // run simulation
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);

    #ifndef TEST
    for_mpi(nsamples, run, NULL);
    #else
    for (int n=0; n<nsamples; n++) run(n, NULL);
    #endif

    if (world_rank == 0){
        gettimeofday(&toc, NULL);
        printf("Simulation done, total elapsed: %.3f hour\n", get_dt(tic, toc)/3600.);
    }

    // close simulation
    free(nn_info_set);
    #ifndef TEST
    end_mpi();
    #endif

    return 0;
}


static void print_help(void){
    printf("-t      Simulation time (s)\n");
    printf("--fdir  Data directory\n");
    exit(-1);
}


static void read_args(int argc, char **argv){

    int n = 1;
    while (n < argc){
        if (strcmp(argv[n], "-t") == 0){
            tmax = atof(argv[n+1]); n++;
        } else if (strcmp(argv[n], "--fdir") == 0){
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



static void run(int job_id, void *nullarg){
    nn_info_t info = nn_info_set[job_id];
    // int N = info.N;

    char fname_id[100], prefix[150];
    sprintf(fname_id, "id%06d", job_id);
    path_join(prefix, fdir_out, fname_id);

    /// check whether the file exist or not
    char fname_tmp[200];
    sprintf(fname_tmp, "%s_lfp.dat", prefix);
    if (access(fname_tmp, F_OK) == 0){
        printf("job %d already done, skip this id\n", job_id);
        return;
    }

    set_seed_by_id(info.seed, 0); // initialize w random seed
    // print_mt_state(0);

    // export information file
    char fname_info[200];
    sprintf(fname_info, "%s_info.txt", prefix);
    write_info(&info, fname_info);

    // build population
    // long seed = read_seed(job_id);
    build_ei_rk4(&info);
    allocate_multiple_ext(&info);
    allocate_target_spike(job_id, info.pN);
    check_building();

    // setup measurement
    int num_pop = info.N - info.pN;
    int nmax=tmax/_dt, neq=teq/_dt;
    int measure_range[] = {num_pop/2, num_pop};
    init_measure(info.N, nmax, 2, measure_range);
    add_pmeasure(info.pN);

    add_checkpoint(0);
    
    #ifdef TEST
    print_syn_ntk(prefix);
    progbar_t bar;
    init_progressbar(&bar, nmax);
    #endif

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


static void allocate_multiple_ext(nn_info_t *info){
    // depends on N
    int num_recv2 = info->pN/2; // 200
    int num_pop2  = (info->N - info->pN)/2; // 1000
    int N2 = info->N/2;

    int nsub = (info->N - info->pN)/2;

    change_default_rng_id(0); // normal neurons
    for (int nt=0; nt<2; nt++){
        int *target = (int*) malloc(sizeof(int) * nsub);
        for (int i=0; i<nsub; i++){
            target[i] = i + nsub * nt;
        }

        set_multiple_ext_input(info, nt, nsub, target);
        free(target);
    }

    int nrecv = info->pN/2;
    change_default_rng_id(1); // Pseudo neurons
    for (int nt=2; nt<4; nt++){

        int *target = (int*) malloc(sizeof(int) * nrecv);
        for (int i=0; i<nrecv; i++){
            target[i] = i + 2*nsub + nrecv*(nt-2);
        }

        set_multiple_ext_input(info, nt, nrecv, target);
        free(target);
    }

    change_default_rng_id(0);

    // for (int nt=0; nt<NPOP; nt++){
    //     int *target = (int*) calloc(N2, sizeof(int));
    //     for (int i=0; i<num_pop2; i++){
    //         target[i] = nt*num_pop2 + i;
    //     }
    //     for (int i=0; i<num_recv2; i++){
    //         target[i+num_pop2] = 2*num_pop2 + nt*num_recv2 + i;
    //     }

    //     printf("nt: %d, N2: %d, 0: %d\n", nt, N2, target[0]);
    //     set_multiple_ext_input(info, nt, N2, target);
    //     free(target);
    // }
}


static void print_syn_ntk(const char *prefix){
    for (int n=0; n<8; n++){
        char fname_syn[200];
        sprintf(fname_syn, "%s_syn_%d.txt", prefix, n);
        print_syn_network(nnpop.syns+n, fname_syn);
    }
}


static void check_spike_info_files(void){
    for (int n=0; n<nsamples; n++){
        char fname_spk[200];
        sprintf(fname_spk, "%s/spk_info_%06d.txt", fdir_out, n);

        FILE *fp = fopen(fname_spk, "r");
        if (fp == NULL){
            printf("spike info file for job id %d does not exist\n", n);
            exit(-1);
        }
        fclose(fp);
    }
}


static void read_params(char *fname){
    // nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * *nline);

    // #ifndef TEST

    if (world_rank == 0){
        FILE *fp = open_params_file(fname);
        nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nsamples);
        for (int n=0; n<nsamples; n++){
            if (read_params_line(fp, &(nn_info_set[n])) == 1){
                char msg[100];
                sprintf(msg, "Failed to read parameter in line %d", n);
                REPORT_ERROR(msg);
            }
        }
        fclose(fp);
    }

    #ifndef TEST

    MPI_Bcast(&nsamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0){
        nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nsamples);
    }

    mpi_barrier();
    MPI_Bcast((void*) nn_info_set, nsamples*sizeof(nn_info_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    // init_nn(nn_info_set[0].N, nn_info_set[0].num_types);

    #endif

    // #else

    // FILE *fp = open_params_file(fname);
    // nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nsamples);
    // for (int n=0; n<nsamples; n++){
    //     if (read_params_line(fp, &(nn_info_set[n])) == 1){
    //         char msg[100];
    //         sprintf(msg, "Failed to read parameter in line %d", n);
    //         REPORT_ERROR(msg);
    //     }
    // }
    // fclose(fp);
    // #endif

}


static int read_params_line(FILE *fp, nn_info_t *info){

    /* === Parameter input ===
    EF, IF, RF, ES, IS, RS (T: tranasmission, R: receiver)
      0: seed (+1 from next)
      0-  2: number of neurons, 
      2- 38: projection probability (p_out),
     38- 74: weight
     74- 76: weight_tr
     76- 77: Poisson firing rate (deprecated)
     77- 79: nu_bck_f, nu_bck_s
    */
    
    // initialize info
    *info = init_build_info();
    long seed;
    fscanf(fp, "%ld,", &seed);
    info->seed = seed;

    // set type range
    int type_range[NTYPE] = {0,}; // The # of E, I, R/T
    for (int n=0; n<NTYPE; n++){
        double val;
        if (fscanf(fp, "%lf,", &(val)) == EOF) return 1;
        type_range[n] = (int) val;
    }

    int num_neuron = 0;
    for (int n=0; n<NTYPE; n++) num_neuron += (int) type_range[n];
    num_neuron *= NPOP;

    // allocate default number
    info->N = num_neuron; // # of normal neurons + receiver neurons
    info->num_types = NTYPE * NPOP;
    info->pN = (int) type_range[NTYPE-1] * NPOP; // # of transmiter neurons

    // synapse connectivity
    // read probability
    int ntypes = NTYPE * NPOP;
    for (int i=0; i<ntypes; i++){
        for (int j=0; j<ntypes; j++){
            if (fscanf(fp, "%lf,", &(info->p_out[i][j])) == EOF) return 1;
        }
    }
    // read weight
    for (int i=0; i<ntypes; i++){
        for (int j=0; j<ntypes; j++){
            if (fscanf(fp, "%lf,", &(info->w[i][j])) == EOF) return 1;
        }
    }
    info->t_lag = 0.;

    // set synapse parameters
    const int num_total_types = 6;

    int type_range_simul[] = {type_range[0], type_range[1], type_range[0], type_range[1], type_range[2], type_range[2]};
    int type_id[] = {E, IF, E, IS, tP, tP};
    set_type_info(info, num_total_types, type_id, type_range_simul);

    double taur_set[] = {taur[0], taur[1], taur[2], taur[3], taur[0], taur[2]};
    double taud_set[] = {taud[0], taud[1], taud[2], taud[3], taud[0], taud[2]};

    // for (int n=0;n<NTYPE*2; n++){
    for (int n=0; n<num_total_types; n++){
        // info->type_range[n] = (int) type_range_simul[n];
        // info->type_id[n] = type_id[n];
        info->taur[n] = taur_set[n];
        info->taud[n] = taud_set[n];
    }

    // read trans
    // int prange[NPOP][2] = {0,};
    int num_recv = type_range[NTYPE-1];
    for (int n=0; n<NPOP; n++){
        info->prange[n][0] = num_neuron - (NPOP-n)*num_recv;
        info->prange[n][1] = num_neuron - (NPOP-n-1)*num_recv;
    }

    int ntypes_t = NTRANS * NPOP;
    for(int n=0; n<ntypes_t; n++){
        if (fscanf(fp, "%lf,", &(info->pw[n])) == EOF) return 1;
        info->pp_out[n] = ONE2ONE;
    }

    if (fscanf(fp, "%lf,", &(info->pfr)) == EOF) return 1;
    info->ptaur = taur[0];
    info->ptaud = taud[0];

    // read background input
    info->num_ext_types = NPOP*2; 
    info->const_current = false;
    for (int n=0; n<NPOP; n++){
        if (fscanf(fp, "%lf,", &(info->nu_ext_mu[n])) == EOF) return 1;
        info->w_ext_mu[n] = 0.002;
    }
    for (int n=0; n<NPOP; n++){
        info->nu_ext_mu[n+2] = info->nu_ext_mu[n];
        info->w_ext_mu[n+2]  = info->w_ext_mu[n];
    }

    info->w_ext_sd = 0;
    info->nu_ext_sd = 0;

    // change line
    fscanf(fp, "\n");    

    return 0;

}


// static long read_seed(int job_id){
//     // file need to contain seed
//     // seed
//     // %d:%lf,%lf,...
//     char fname_spk[200];
//     sprintf(fname_spk, "%s/spk_info_%06d.txt", fdir_out, job_id);

//     FILE *fp = fopen(fname_spk, "r");
//     long seed;
//     fscanf(fp, "%ld\n", &seed);
//     fclose(fp);

//     return seed;
// }


static void allocate_target_spike(int job_id, int num_pcells){
    // file need to contain seed
    // seed (x)
    // %d:%lf,%lf,...
    char fname_spk[200];
    sprintf(fname_spk, "%s/spk_info_%06d.txt", fdir_out, job_id);

    FILE *fp = fopen(fname_spk, "r");

    // long seed;
    // fscanf(fp, "%ld\n", &seed);
    // printf("total pN: %d, %d\n", nnpop.pneuron->N, num_pcells);

    // read t_spks
    for (int n=0; n<num_pcells; n++){
        int len;
        fscanf(fp, "%d:", &len);
        
        double *t_spk = (double*) malloc(sizeof(double) * len);
        for (int i=0; i<len; i++){
            fscanf(fp, "%lf,", t_spk+i);
        }

        // printf("ID (%d): %d\n",len,  n);
        // for (int i=0; i<len;i++){s[]
        //     printf("%lf,", t_spk[i]);
        // }
        // printf("\n");

        set_pneuron_target_time(nnpop.pneuron, n, len, t_spk);

        // int nmax = nnpop.pneuron->max_steps[n];
        // printf("Id: %d (%d)\n", n, nmax);
        // for (int i=0; i<nmax; i++){
        //     printf("%d,", nnpop.pneuron->target_steps[n][i]);
        // }
        // printf("\n");

        free(t_spk);
        fscanf(fp, "\n");
    }

    fclose(fp);
}




static FILE *open_params_file(const char *fname){
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


static FILE *open_spike_info_file(const char *fname){
    FILE *fp = fopen(fname, "r+");
    if (fp == NULL){
        printf("file %s does not exist\n", fname);
        exit(-1);
    }
}