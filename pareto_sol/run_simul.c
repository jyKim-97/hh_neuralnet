#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "storage.h"
#include "model2.h"
#include "neuralnet.h"
#include "measurement2.h"


#define _debug

#ifdef _mpi
#include "mpifor.h"
extern int world_rank, world_size;
#endif


int N = 1000;
nn_info_t allocate_simulation_param(const char *fname);
void run(const char *out_name, nn_info_t *nn_info);

double teq = 500;
double tmax = 1500;
int flag_eq = 0;
// char fdir[] = "/home/jungyoung/Project/hh_neuralnet/pareto_sol";
char fdir[20] = "./";

int main(int argc, char **argv){

    #ifdef _mpi
    init_mpi(&argc, &argv);
    #else
    int world_rank=0, world_size=1;
    #endif
    
    int start_id=0, len=1, arg_stack=0;
    // argc needs to contain "start_id" and "len"
    for (int n=1; n<argc; n++){
        if (strcmp("--start_id", argv[n]) == 0){
            start_id = atoi(argv[n+1]);
            arg_stack++;
        } else if (strcmp("--len", argv[n]) == 0){
            len = atoi(argv[n+1]);
            arg_stack++;
        } else if (strcmp("--tmax", argv[n]) == 0){
            double tmax_tmp = atof(argv[n+1]);
            tmax = tmax_tmp>teq? tmax_tmp: teq+500;
        }
    }

    if (arg_stack != 2){
        printf("Given arguments are not enough, need to contain 'start_id' and 'len'");
        exit(-1);
    }

    int job_id = start_id + world_rank;
    char finfo[100];
    sprintf(finfo, "%s/params/param_%04d.txt", fdir, job_id);
    nn_info_t info = allocate_simulation_param(finfo);

    char fout[100];
    sprintf(fout, "%s/results/summary_%04d.txt", fdir, job_id);
    run(fout, &info);

    #ifdef _mpi
    end_mpi();
    #endif

    return 0;
}


void run(const char *out_name, nn_info_t *nn_info){

    extern wbneuron_t neuron;
    build_ei_rk4(nn_info);

    int nmax = tmax/_dt;
    init_measure(N, nmax, 2, NULL);
    add_checkpoint(0);

    #ifdef _debug
    progbar_t bar = {0,};
    init_progressbar(&bar, nmax);
    #endif


    for (int nstep=0; nstep<nmax; nstep++){

        if ((flag_eq == 0) && (nstep * _dt >= teq)){
            flush_measure();
            add_checkpoint(nstep);
            flag_eq = 1;
        }

        update_rk4(nstep, 0);
        measure(nstep, &neuron);

        #ifdef _debug
        progressbar(&bar, nstep);
       #endif
    }

    summary_t obj = flush_measure();
    export_result(&obj, out_name);
    destroy_neuralnet();
    destroy_measure();
}



/* 
Parameter input

 1. w_e->e
 2. w_e->i
 3. w_i->e
 4. w_i->i

 5. p_e->e
 6. p_e->i
 7. p_i->e
 8. p_i->i
 
 9. tlag
10. taur
11. taud
12. nu_ext

*/


nn_info_t allocate_simulation_param(const char *fname){

    FILE *fp = fopen(fname, "r");
    if (fp == NULL){
        printf("file %s does not exist\n", fname);
        exit(-1);
    }

    nn_info_t nn_info = init_build_info(N, 2);

    fscanf(fp, "%lf,", &(nn_info.w[0][0]));
    fscanf(fp, "%lf,", &(nn_info.w[0][1]));
    fscanf(fp, "%lf,", &(nn_info.w[1][0]));
    fscanf(fp, "%lf,", &(nn_info.w[1][1]));

    fscanf(fp, "%lf,", &(nn_info.p_out[0][0]));
    fscanf(fp, "%lf,", &(nn_info.p_out[0][1]));
    fscanf(fp, "%lf,", &(nn_info.p_out[1][0]));
    fscanf(fp, "%lf,", &(nn_info.p_out[1][1]));

    fscanf(fp, "%lf,", &(nn_info.t_lag));
    fscanf(fp, "%lf,", &(nn_info.taur[1]));
    fscanf(fp, "%lf,", &(nn_info.taud[1]));
    fscanf(fp, "%lf,", &(nn_info.nu_ext_mu));
    
    nn_info.taur[0] = 0.3;
    nn_info.taud[0] = 1.;
    nn_info.w_ext_mu = 0.002;
    nn_info.w_ext_sd = 0;
    nn_info.nu_ext_sd = 0;
    nn_info.const_current = false;

    fclose(fp);

    return nn_info;
}
