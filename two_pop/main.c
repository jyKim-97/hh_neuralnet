#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "storage.h"
#include "model2.h"
#include "neuralnet.h"
#include "measurement2.h"

/*
gcc -g -Wall -O2 -std=c11 -I../include -o main.out main.c -L../lib -lhhnet
*/

extern double _dt;
extern wbneuron_t neuron;

void init();
void run(double tmax);
nn_info_t set_info(void);

char fdir[100] = "./tmp";
FILE *fp_v = NULL;
int N = 1000;
double iapp = 0;
double tmax = 2500;

// #define SST

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

    init();
    run(tmax);
    destroy_neuralnet();
}


void init(){
    nn_info_t info = set_info();
    build_ei_rk4(&info);

    #ifdef SST
    for (int n=N*0.8; n<N; n++){
        neuron.params[n].cm  = 2;
        neuron.params[n].gna = 60;
        neuron.params[n].gk  = 20;
        neuron.params[n].gl  = 0.1;
        neuron.params[n].phi = 5;
    }
    #endif

    extern desyn_t syns[MAX_TYPE];
    write_info(&info, path_join(fdir, "info.txt"));
    print_syn_network(&syns[0], path_join(fdir, "ntk_e.txt"));
    print_syn_network(&syns[1], path_join(fdir, "ntk_i.txt"));
}

double teq = 500;
int flag_eq = 0;
void run(double tmax){
    int nmax = tmax/_dt;

    init_measure(N, nmax, 2, NULL);
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

        // fwrite(neuron.vs, sizeof())
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
    nn_info_t info = get_empty_info();

    info.N = N;
    info.num_types = 2;
    info.type_range[0] = info.N * 0.8;
    info.type_range[1] = info.N;

    info.p_out[0][0] = 0.;
    info.p_out[0][1] = 0.;
    info.p_out[1][0] = 0.01;
    info.p_out[1][1] = 0.01;

    info.w[0][0] = 0.;
    info.w[0][1] = 0.;
    info.w[1][0] = 0.001;
    info.w[1][1] = 0.001;

    info.t_lag = 0.5;
    info.nu_ext = 500;
    info.w_ext  = 0.001;
    info.const_current = false;

    return info;
}


