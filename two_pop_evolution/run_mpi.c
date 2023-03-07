#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "storage.h"
#include "model2.h"
#include "neuralnet.h"
#include "measurement2.h"
#include "mpifor.h"


int N = 1000;
extern int world_rank, world_size;
nn_info_t allocate_simulation_param(const char *fname);
void run(const char *out_name, nn_info_t *nn_info);

double teq = 500;
double tmax = 1500;
int flag_eq = 0;
const char *fdir_abs = "/home/jungyoung/Project/hh_neuralnet/two_pop_evolution";

int main(int argc, char **argv){

    init_mpi(&argc, &argv);

    // need to read job index [0, offpsring number)
    if (argc != 2){
        printf("Need to get directory number\n");
        exit(-1);
    }

    int id = atoi(argv[1]);

    char finfo[100];
    // sprintf(finfo, "./data/process%d/param%d.txt", id, world_rank);
    sprintf(finfo, "%s/data/process%d/param%d.txt", fdir_abs, id, world_rank);
    nn_info_t info = allocate_simulation_param(finfo);

    // write_info(&info, "./test_input.txt");

    char fout[100];
    sprintf(fout, "%s/data/process%d/result%d.txt", fdir_abs, id, world_rank);
    run(fout, &info);

    end_mpi();

    return 0;
}


void run(const char *out_name, nn_info_t *nn_info){

    extern wbneuron_t neuron;
    build_ei_rk4(nn_info);

    int nmax = tmax/_dt;
    init_measure(N, nmax, 2, NULL);
    add_checkpoint(0);
    for (int nstep=0; nstep<nmax; nstep++){

        if ((flag_eq == 0) && (nstep * _dt >= teq)){
            flush_measure();
            add_checkpoint(nstep);
            flag_eq = 1;
        }

        update_rk4(nstep, 0);
        measure(nstep, &neuron);
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