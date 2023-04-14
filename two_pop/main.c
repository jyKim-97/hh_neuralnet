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
int N = 500;
double iapp = 0;
double tmax = 4000; // ~220 s (with N=1000)

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

    char fout[200];
    path_join(fout, fdir, "info.txt");
    write_info(&info, fout);
    
    path_join(fout, fdir, "ntk_e.txt");
    print_syn_network(&syns[0], fout);

    path_join(fout, fdir, "ntk_i.txt");
    print_syn_network(&syns[1], fout);
}

double teq = 500;
int flag_eq = 0;
void run(double tmax){

    int nmax = tmax/_dt;
    int neq  = teq / _dt;
    // int nmove = 200 / _dt;

    init_measure(N, nmax, 2, NULL);
    add_checkpoint(0);

    fp_v     = open_file_wdir(fdir, "v_out.dat", "wb");
    fp_syn_e = open_file_wdir(fdir, "syn_e.dat", "wb");
    fp_syn_i = open_file_wdir(fdir, "syn_i.dat", "wb");

    progbar_t bar;
    init_progressbar(&bar, nmax);
    for (int nstep=0; nstep<nmax; nstep++){

        if ((flag_eq == 0) && (nstep == neq)){
            flush_measure();
            add_checkpoint(nstep);
            flag_eq = 1;
        }

        update_rk4(nstep, iapp);
        measure(nstep, &neuron);
        save(N, nstep, neuron.vs, fp_v);
        save_current(nstep);
        progressbar(&bar, nstep);

    }
    printf("\n");

    char fbuf[200];
    path_join(fbuf, fdir, "spk.dat");
    export_spike(fbuf);

    path_join(fbuf, fdir, "lfp.dat");
    export_lfp(fbuf);

    fclose(fp_v);
    fclose(fp_syn_e);
    fclose(fp_syn_i);
    summary_t obj = flush_measure();

    path_join(fbuf, fdir, "result.txt");
    export_result(&obj, fbuf);
}


nn_info_t set_info(void){
    // nn_info_t info = {0,};
    nn_info_t info = init_build_info(N, 2);

    // info.p_out[0][0] = 0.21;
    // info.p_out[0][1] = 0.23706;
    // info.p_out[1][0] = 0.93227;
    // info.p_out[1][1] = 0.63310;

    // info.w[0][0] = 0.01537;
    // info.w[0][1] = 0.10873;
    // info.w[1][0] = 0.13037;
    // info.w[1][1] = 0.22721;

    // info.taur[0] = 0.3;
    // info.taud[0] = 1;
    // info.taur[1] = 0.5;
    // info.taud[1] = 2.5;
    // // info.taur[1] = 1;
    // // info.taud[1] = 10;

    // info.t_lag = 0.78073;
    // info.nu_ext_mu = 6298.24245;
    // info.nu_ext_sd = 0;
    // info.w_ext_mu  = 0.002;
    // info.w_ext_sd  = 0.0;
    // info.const_current = false;

    info.p_out[0][0] = 0.21 * 2;
    info.p_out[0][1] = 0.01129 * 2;
    info.p_out[1][0] = 0.04439 * 2;
    info.p_out[1][1] = 0.03015 * 2;

    info.w[0][0] = 0.07044;
    info.w[0][1] = 0.49824;
    info.w[1][0] = 0.59741;
    info.w[1][1] = 1.04121;

    info.taur[0] = 0.3;
    info.taud[0] = 1;
    info.taur[1] = 0.5;
    info.taud[1] = 2.5;
    // info.taur[1] = 1;
    // info.taud[1] = 10;

    info.t_lag = 0.78073;
    info.nu_ext_mu = 1374.38918;
    info.nu_ext_sd = 0;
    info.w_ext_mu  = 0.002;
    info.w_ext_sd  = 0.0;
    info.const_current = false;

    return info;
}


void save_current(int nstep){
    if (is_save_step(nstep) == 0) return;

    extern desyn_t syns[MAX_TYPE];
    float *ic_e = (float*) malloc(sizeof(float) * N);
    float *ic_i = (float*) malloc(sizeof(float) * N);
    
    for (int n=0; n<N; n++){
        ic_e[n] = (float) get_current(&(syns[0]), n, neuron.vs[n]);
        ic_i[n] = (float) get_current(&(syns[1]), n, neuron.vs[n]);
    }

    fwrite(ic_e, sizeof(float), N, fp_syn_e);
    fwrite(ic_i, sizeof(float), N, fp_syn_i);
    
    free(ic_e);
    free(ic_i);
}
