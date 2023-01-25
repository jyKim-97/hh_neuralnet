#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "storage.h"
#include "model2.h"
#include "neuralnet.h"
#include "measurement2.h"

/*
gcc -g -Wall -O2 -std=c11 -I../include -o main.out main.c -L../lib -lhhnet -lm
*/

extern double _dt;
extern wbneuron_t neuron;

void init();
void run(double tmax);

nn_info_t set_info(void);
nn_info_t info = {0,};
void allocate_multiple_ext(nn_info_t *info);

char fdir[100] = "./tmp";
FILE *fp_v = NULL;
int N = 2000;
double iapp = 0;
double tmax = 2500;



int main(int argc, char **argv){

    for (int n=1; n<argc; n++){
        if (strcmp(argv[n], "-fdir") == 0){
            sprintf(fdir, "%s", argv[n+1]);
        }

        if (strcmp(argv[n], "-tmax") == 0){
            tmax = atof(argv[n+1]);
        }
    }

    _dt = 0.01;
    change_sampling_rate(2000);
    set_seed(1000);

    init();
    run(tmax);
    destroy_neuralnet();
}


void init(){
    info = set_info();
    build_ei_rk4(&info);
    allocate_multiple_ext(&info);

    extern desyn_t syns[MAX_TYPE];
    write_info(&info, path_join(fdir, "info.txt"));
    print_syn_network(&syns[0], path_join(fdir, "ntk_e.txt"));
    print_syn_network(&syns[1], path_join(fdir, "ntk_i.txt"));
}


void allocate_multiple_ext(nn_info_t *info){
    // depends on N
    int nsub = N/2;
    int *target = (int*) malloc(sizeof(int) * nsub);
    
    for (int ntype=0; ntype<2; ntype++){
        // for excitatory population
        int num_e = nsub * 0.8;
        for (int i=0; i<num_e; i++){
            target[i] = num_e * ntype + i;
        }
        // for inhibitory population
        int n0 = N * 0.8;
        int num_i = nsub * 0.2;
        for (int i=0; i<num_i; i++){
            target[num_e+i] = n0 + num_i * ntype + i;
        }

        set_multiple_ext_input(info, ntype, nsub, target);
    }

    check_multiple_input();
}


double teq = 500;
int flag_eq = 0;
void run(double tmax){
    int nmax = tmax/_dt;

    init_measure(N, nmax, 3, info.type_range);
    fp_v = fopen(path_join(fdir, "v_out.dat"), "wb");

    progbar_t bar;
    init_progressbar(&bar, nmax);
    for (int nstep=0; nstep<nmax; nstep++){

        if ((flag_eq == 0) && (nstep * _dt >= teq)){
            flush_measure();
            flag_eq = 1;
        }

        update_rk4(nstep, iapp);
        measure(nstep, &neuron);
        save(N, nstep, neuron.vs, fp_v);
        progressbar(&bar, nstep);
    }
    printf("\n");

    export_spike(path_join(fdir, "spk.dat"));
    export_lfp(path_join(fdir, "lfp.dat"));

    fclose(fp_v);
    summary_t obj = flush_measure();
    export_result(&obj, path_join(fdir, "result.txt"));
}


nn_info_t set_info(void){
    // nn_info_t info = {0,};
    nn_info_t info = init_build_info(N, 3);

    // connection probability
    info.p_out[0][0] = 0.01;
    info.p_out[0][1] = 0.01;
    info.p_out[0][2] = 0.01;

    info.p_out[1][0] = 0.1;
    info.p_out[1][1] = 0.1;
    info.p_out[1][2] = 0.1;

    info.p_out[2][0] = 0.1;
    info.p_out[2][1] = 0.1;
    info.p_out[2][2] = 0.1;

    // connection strength
    info.w[0][0] = 0.01;
    info.w[0][1] = 0.01;
    info.w[0][2] = 0.01;

    info.w[1][0] = 0.01;
    info.w[1][1] = 0.01;
    info.w[1][2] = 0.01;

    info.w[2][0] = 0.01;
    info.w[2][1] = 0.1;
    info.w[2][2] = 0.1;

    info.taur[0] = 0.5;
    info.taur[1] = 1;
    info.taur[2] = 2;

    info.taud[0] = 1;
    info.taud[1] = 2.5;
    info.taud[2] = 8;

    info.t_lag = 0.;
    info.const_current = false;

    info.num_ext_types = 2;
    info.nu_ext_multi[0] = 2000;
    info.nu_ext_multi[1] = 3000;
    info.w_ext_multi[0] = 0.002;
    info.w_ext_multi[1] = 0.002;

    return info;
}


