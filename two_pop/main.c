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
void save_current(int nstep);

char fdir[100] = "./tmp";
FILE *fp_v = NULL, *fp_syn_e=NULL, *fp_syn_i=NULL;
int N = 1000;
double iapp = 0;
double tmax = 5000;

int main(int argc, char **argv){

    for (int n=1; n<argc; n++){
        if (strcmp(argv[n], "--fdir") == 0){
            sprintf(fdir, "%s", argv[n+1]);
        }

        if (strcmp(argv[n], "--tmax") == 0){
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

    extern desyn_t syns[MAX_TYPE];
    write_info(&info, path_join(fdir, "info.txt"));
    print_syn_network(&syns[0], path_join(fdir, "ntk_e.txt"));
    print_syn_network(&syns[1], path_join(fdir, "ntk_i.txt"));
}

double teq = 500;
int flag_eq = 0;
void run(double tmax){

    int nmax = tmax/_dt;
    int neq  = teq / _dt;
    int nmove = 200 / _dt;

    init_measure(N, nmax, 2, NULL);
    add_checkpoint(0);

    fp_v = fopen(path_join(fdir, "v_out.dat"), "wb");
    fp_syn_e = fopen(path_join(fdir, "syn_e.dat"), "wb");
    fp_syn_i = fopen(path_join(fdir, "syn_i.dat"), "wb");


    progbar_t bar;
    init_progressbar(&bar, nmax);
    for (int nstep=0; nstep<nmax; nstep++){

        if ((flag_eq == 0) && (nstep == neq)){
            flush_measure();
            add_checkpoint(nstep);
            flag_eq = 1;
        }

        // if ((flag_eq == 1) && ((nstep-neq)%nmove == 0)){
        //     add_checkpoint(nstep);
        //     print_num_check();

        //     if (nstep >= neq + 5*nmove){
        //         summary_t obj = flush_measure();
        //     }
        // }

        update_rk4(nstep, iapp);
        measure(nstep, &neuron);
        save(N, nstep, neuron.vs, fp_v);
        save_current(nstep);
        progressbar(&bar, nstep);

    }
    printf("\n");

    export_spike(path_join(fdir, "spk.dat"));
    export_lfp(path_join(fdir, "lfp.dat"));

    fclose(fp_v);
    fclose(fp_syn_e);
    fclose(fp_syn_i);
    summary_t obj = flush_measure();
    export_result(&obj, path_join(fdir, "result.txt"));
}


nn_info_t set_info(void){
    // nn_info_t info = {0,};
    nn_info_t info = init_build_info(N, 2);

    info.p_out[0][0] = 0.234024;
    info.p_out[0][1] = 0.234024;
    info.p_out[1][0] = 0.702073;
    info.p_out[1][1] = 0.702073;

    info.w[0][0] = 0.002067;
    info.w[0][1] = 0.002067;
    info.w[1][0] = 0.019431;
    info.w[1][1] = 0.019431;

    info.taur[0] = 0.3;
    info.taud[0] = 1;
    info.taur[1] = 0.5;
    info.taud[1] = 2.5;

    info.t_lag = 0.;
    info.nu_ext_mu = 6841.410000;
    info.nu_ext_sd = 0;
    info.w_ext_mu  = 0.002;
    info.w_ext_sd  = 0.0;
    info.const_current = false;

    return info;
}


extern int _nstep_save;

void save_current(int nstep){
    if (nstep % _nstep_save != 0) return;

    extern desyn_t syns[MAX_TYPE];

    double *ic_e = (double*) malloc(sizeof(double) * N);
    double *ic_i = (double*) malloc(sizeof(double) * N);
    
    for (int n=0; n<N; n++){
        ic_e[n] = get_current(&(syns[0]), n, neuron.vs[n]);
        ic_i[n] = get_current(&(syns[1]), n, neuron.vs[n]);
    }

    write_signal_d(N, ic_e, fp_syn_e);
    write_signal_d(N, ic_i, fp_syn_i);
    
    free(ic_e);
    free(ic_i);
}
