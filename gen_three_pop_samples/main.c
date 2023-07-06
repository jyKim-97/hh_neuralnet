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

// 


extern double _dt;
extern wbneuron_t neuron;
extern int world_size, world_rank;


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
void read_params(int nline, char *fname);
void read_params_line(FILE *fp, nn_info_t *info);
void allocate_multiple_ext(nn_info_t *info);


int main(int argc, char **argv){

    init_mpi(&argc, &argv);
    set_seed(time(NULL) * world_size * 5 + world_rank*2);

    /* read args & init parameters */
    read_args(argc, argv);
    read_params(nsamples, fname_params);

    /* run simulation */
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
    for_mpi(nsamples, run, NULL);

    if (world_rank == 0){
        gettimeofday(&toc, NULL);
        printf("Simulation done, total elapsed: %.3f hour\n", get_dt(tic, toc)/3600.);
    }

    /* close simulation */
    free(nn_info_set);
    end_mpi();
    return 0;
}


void run(int job_id, void *nullarg){
    print_job_start(job_id, nsamples);

    nn_info_t info = nn_info_set[job_id];

    char fname_info[100], fbuf[200];
    sprintf(fname_info, "id%06d_info.txt", job_id);
    path_join(fbuf, fdir_out, fname_info);
    write_info(&info, fbuf);

    build_ei_rk4(&info);
    allocate_multiple_ext(&info);

    int nmax = tmax/_dt;
    int neq  = teq / _dt;

    int pop_range[2] = {N/2, N};
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

        if (nstep == 2*neq){
            summary_t obj = flush_measure();
            char fname_res[100];
            sprintf(fname_res, "id%06d_result(monitor).txt", job_id);
            path_join(fbuf, fdir_out, fname_res);
            export_result(&obj, fbuf);
        }

        update_rk4(nstep, 0);
        KEEP_SIMUL();
        measure(nstep, &neuron);
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

    print_job_end(job_id, nsamples);
}


void read_args(int argc, char **argv){
    // get blank typed_args_t & set default values
    // NOTE: hard fix the args

    // strcpy(fname_params, "./data/params_to_run.txt");
    strcpy(fdir_out, "./data/");

    int n = 1;
    while (n < argc){
        if (strcmp(argv[n], "-n") == 0){
            nsamples = atoi(argv[n+1]); n++;
        } else if (strcmp(argv[n], "-t") == 0){
            tmax = atof(argv[n+1]); n++;
        } else if (strcmp(argv[n], "--fparam") == 0){
            strcpy(fname_params, argv[n+1]); n++;
        } else if (strcmp(argv[n], "--fdir_out") == 0){
            strcpy(fdir_out, argv[n+1]); n++;
        } else {
            printf("Wrong argument typed: %s\n", argv[n]);
            exit(-1);
        }
        n++;
    }

    sprintf(fname_params, "%s/params_to_run.txt", fdir_out);
}


void read_params(int nline, char *fname){

    nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nline);

    if (world_rank == 0){
        FILE *fp;
        if ((fp = fopen(fname, "r+")) == NULL){
            printf("file %s does not exist\n", fname);
            exit(-1);
        }

        for (int n=0; n<nline; n++){
            read_params_line(fp, &(nn_info_set[n]));
        }

        fclose(fp);
    }

    MPI_Bcast((void*) nn_info_set, nline*sizeof(nn_info_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    init_nn(nn_info_set[0].N, nn_info_set[0].num_types);
}


void read_params_line(FILE *fp, nn_info_t *info){

    /* === Parameter input ===
     0: cluster id
     1- 4. w_ef, w_if, w_es, w_is
     5- 8. p_{ef->ef,if,es,is}
     9-12. p_{if->ef,if,es,is}
    13-16. p_{es->ef,if,es,is}
    17-21. p_{is->ef,if,es,is}
    23-24. nu_bck_f, nu_bck_s
    */
    
    int ntype = 4;
    *info = init_build_info(N, ntype);

    // default parameters
    info->taur[0] = 0.3; info->taud[0] = 1;   // -> E
    info->taur[1] = 0.5; info->taud[1] = 2.5; // -> I_F
    info->taur[2] = 0.3; info->taud[2] = 1;   // -> E
    info->taur[3] = 1;   info->taud[3] = 8;   // -> I_S
    
    info->num_ext_types = 2;
    info->w_ext_multi[0] = 0.002;
    info->w_ext_multi[1] = 0.002;
    info->t_lag = 0.;
    info->const_current = false;

    // set weight
    for (int i=0; i<ntype; i++){
        double w=0;
        fscanf(fp, "%lf,", &w);
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
        fscanf(fp, "%lf,", &(info->nu_ext_multi[i]));
    }

    // change line
    fscanf(fp, "\n");
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
