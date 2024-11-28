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

#define TEST

#ifndef TEST
#include "mpifor.h"
#endif


extern double _dt;
// extern wbneuron_t neuron;
extern nnpop_t nnpop;
extern int world_size, world_rank;

static void print_syn_ntk(const char *prefix);


int N = 2000;
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

// double taur[] = {0.3, 0.5, 0.3, 1};
// double taud[] = {  1, 2.5,   1, 8};

double taur[] = {0.3, 0.5, 0.3, 1};
double taud[] = {  1, 2.5,   1, 8};
// mpirun -np 100 --hostfile hostfile ./main.out -t 10500 --fdir_out ./data

/* Note that the time constant for the synapse has been changed */

int main(int argc, char **argv){

    /* read args & init parameters */
    #ifndef TEST
    init_mpi(&argc, &argv);
    #endif
    // set_seed(time(NULL) * world_size * 5 + world_rank*2);
    ignore_exist_file(true);
    read_args(argc, argv);
    read_params(fname_params);

    #ifndef TEST
    mpi_barrier();
    #endif
    if (world_rank == 0){
        printf("%d samples are found\n", nsamples);
    }

    // end_mpi();
    // exit(1);

    /* run simulation */
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

    /* close simulation */
    free(nn_info_set);
    #ifndef TEST
    end_mpi();
    #endif

    return 0;
}


void run(int job_id, void *nullarg){

    nn_info_t info = nn_info_set[job_id];

    // printf("#%03d started (seed: %8d)\n", job_id, info.seed);

    // check is file exist
    char fname_exist[200];
    sprintf(fname_exist, "%s/id%06d_lfp.dat", fdir_out, job_id);
    int res = access(fname_exist, F_OK);

    if (access(fname_exist, F_OK) == 0){
        printf("job %d already done, skip this id\n", job_id);
        return;
    }

    char fname_info[100], fbuf[200];
    sprintf(fname_info, "id%06d_info.txt", job_id);
    path_join(fbuf, fdir_out, fname_info);
    write_info(&info, fbuf);

    set_seed(info.seed);
    build_ei_rk4(&info);
    allocate_multiple_ext(&info);

    int nmax = tmax/_dt;
    int neq  = teq / _dt;

    int pop_range[2] = {N/2, N};
    init_measure(N, nmax, 2, pop_range);
    add_checkpoint(0);

    // check synaptic network
    #ifdef TEST
    char fname_id[100], prefix[200];
    sprintf(fname_id, "id%06d", job_id);
    path_join(prefix, fdir_out, fname_id);
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

        if (nstep == 2*neq){
            summary_t obj = flush_measure();
            char fname_res[100];
            sprintf(fname_res, "id%06d_result(monitor).txt", job_id);
            path_join(fbuf, fdir_out, fname_res);
            export_result(&obj, fbuf);
        }

        update_rk4(nstep, 0);
        KEEP_SIMUL();
        // measure(nstep, &neuron);
        measure(nstep, &nnpop);

        #ifdef TEST
        progressbar(&bar, nstep);
        #endif
    }

    /* Extract values */
    summary_t obj = flush_measure();
    char fname_res[100];
    sprintf(fname_res, "id%06d_result.txt", job_id);
    path_join(fbuf, fdir_out, fname_res);
    export_result(&obj, fbuf);

    char fname_spk[100];
    sprintf(fname_spk, "id%06d_spk.dat", job_id);
    path_join(fbuf, fdir_out, fname_spk);
    export_spike(fbuf);

    char fname_lfp[100];
    sprintf(fname_lfp, "id%06d_lfp.dat", job_id);
    path_join(fbuf, fdir_out, fname_lfp);
    export_lfp(fbuf);

    destroy_neuralnet();
    destroy_measure();

    // print_job_end(job_id, nsamples);
}


void read_args(int argc, char **argv){
    // get blank typed_args_t & set default values
    // NOTE: hard fix the args

    // strcpy(fname_params, "./data/params_to_run.txt");
    // strcpy(fdir_out, "./data/");

    int n = 1;
    while (n < argc){
        // if (strcmp(argv[n], "-n") == 0){
        //     nsamples = atoi(argv[n+1]); n++;
        // } else if (strcmp(argv[n], "-t") == 0){
        if (strcmp(argv[n], "-t") == 0){
            tmax = atof(argv[n+1]); n++;
        } else if (strcmp(argv[n], "--fdir") == 0){
            strcpy(fdir_out, argv[n+1]); n++;
        } else {
            printf("Wrong argument typed: %s\n", argv[n]);
            exit(-1);
        }
        n++;
    }

    sprintf(fname_params, "%s/params_to_run.txt", fdir_out);

    if (world_rank == 0){
        printf("tmax: %f\nfparams: %s\nfdir_out: %s\n", tmax, fname_params, fdir_out);
    }
}


void read_params(char *fname){

    // nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * *nline);

    if (world_rank == 0){
        FILE *fp;
        if ((fp = fopen(fname, "r+")) == NULL){
            printf("file %s does not exist\n", fname);
            exit(-1);
        }
        
        if (fscanf(fp, "%d\n", &nsamples) < 0){
            printf("Falied to read the number of samples. Check the parameter info file\n");
            exit(-1);
        }

        nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nsamples);
        for (int n=0; n<nsamples; n++){
            if (read_params_line(fp, &(nn_info_set[n])) == 1){
                fprintf(stderr, "Failed to read parameter in line %d\n", n);
                exit(-1);
            }
        }

        fclose(fp);

        // printf("fast->slow: %f\n slow->fast: %f\n", nn_info_set[nsamples-1].p_out[0][2],
        //                                         nn_info_set[nsamples-1].p_out[2][0]);
    }

    #ifndef TEST
    MPI_Bcast(&nsamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0){
        nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nsamples);
    }

    mpi_barrier();
    MPI_Bcast((void*) nn_info_set, nsamples*sizeof(nn_info_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    #endif
    // init_nn(nn_info_set[0].N, nn_info_set[0].num_types);
}


int read_params_line(FILE *fp, nn_info_t *info){

    /* === Parameter input ===
     0: cluster id
     1- 4. w_ef, w_if, w_es, w_is
     5- 8. p_{ef->ef,if,es,is}
     9-12. p_{if->ef,if,es,is}
    13-16. p_{es->ef,if,es,is}
    17-21. p_{is->ef,if,es,is}
    23-24. nu_bck_f, nu_bck_s
    */
    
    *info = init_build_info();
    info->N = N;
    info->pN = 0;

    // set type info
    int ntype = 4;
    int type_id[] = {E, IF, E, IS};
    int type_range[] = {800, 200, 800, 200};
    set_type_info(info, ntype, type_id, type_range);

    // info->type_id[0] = E;
    // info->type_id[1] = IF;
    // info->type_id[2] = E;
    // info->type_id[3] = IS;

    // info->type_range[0] = 800;
    // info->type_range[1] = 200;
    // info->type_range[2] = 800;
    // info->type_range[3] = 200;

    // default parameters
    info->taur[0] = taur[0]; info->taud[0] = taud[0];   // -> E
    info->taur[1] = taur[1]; info->taud[1] = taud[1];   // -> I_F
    info->taur[2] = taur[2]; info->taud[2] = taud[2];   // -> E
    info->taur[3] = taur[3]; info->taud[3] = taud[3];   // -> I_S
    // info->taur[2] = 0.3; info->taud[2] = 1;   // -> E
    // info->taur[3] = 0.5;   info->taud[3] = 2.5;   // -> I_S
    
    info->num_ext_types = 2;
    // info->w_ext_multi[0] = 0.002;
    // info->w_ext_multi[1] = 0.002;
    info->w_ext_mu[0] = 0.002;
    info->w_ext_mu[1] = 0.002;
    info->t_lag = 0.;
    info->const_current = false;

    // set weight
    long seed;
    fscanf(fp, "%ld,", &seed);
    info->seed = seed;

    for (int i=0; i<ntype; i++){
        double w=0;
        if (fscanf(fp, "%lf,", &w) < 1){
            return 1;
        }
        for (int j=0; j<ntype; j++){
            info->w[i][j] = w;
        }
    }

    // set connection probability (out)
    for (int i=0; i<ntype; i++){
        for (int j=0; j<ntype; j++){
            fscanf(fp, "%lf,", &(info->p_out[i][j]));
        }
    }

    // set nu_bck
    for (int i=0; i<2; i++){
        // fscanf(fp, "%lf,", &(info->nu_ext_multi[i]));
        fscanf(fp, "%lf,", &(info->nu_ext_mu[i]));
    }

    // change line
    fscanf(fp, "\n");

    return 0;
}


void allocate_multiple_ext(nn_info_t *info){
    // depends on N
    int nsub = N/2;
    int *target = (int*) malloc(sizeof(int) * nsub);
    
    for (int ntype=0; ntype<2; ntype++){
        for (int i=0; i<nsub; i++){
            target[i] = i + nsub * ntype;
        }

        printf("nt: %d, nsub: %d, 0: %d\n", ntype, nsub, target[0]);
        set_multiple_ext_input(info, ntype, nsub, target);
    }

    free(target);
    // check_multiple_input();
}


static void print_syn_ntk(const char *prefix){
    for (int n=0; n<4; n++){
        char fname_syn[200];
        sprintf(fname_syn, "%s_syn_%d.txt", prefix, n);
        print_syn_network(nnpop.syns+n, fname_syn);
    }
}