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

typedef struct _simul_info_t{
    
    double p_out[2][2];
    double w[2][2];
    double t_lag;
    double nu_ext_mu;
    double w_ext_mu;

} simul_info_t;


nn_info_t set_simulation(simul_info_t info);
int run(int simul_id, simul_info_t info);
void set_parent_dir(char *parent_dir);
void set_taue(double tr, double td);
void set_taui(double tr, double td);

char fdir[100] = "./tmp";
int N = 1000;
double tmax = 1000;
double nu_ext_sd = 0;
double w_ext_sd = 0;
double taur_e=0.3, taud_e=1;
double taur_i=0.5, taud_i=2;
// 0.5, 2 - 1, 5

extern wbneuron_t neuron;
double teq = 500;
int flag_eq = 0;


int main(){
    simul_info_t info = {0,};
    run(1, info);
}


int run(int simul_id, simul_info_t info){
    nn_info_t nn_info = set_simulation(info);
    build_ei_rk4(&nn_info);
    char fname_info[100];
    sprintf(fname_info, "id%06d_info.txt", simul_id);
    write_info(&nn_info, path_join(fdir, fname_info)); // for verification, save enviornment

    int nmax = tmax/_dt;
    init_measure(N, nmax, 2, NULL);
    for (int nstep=0; nstep<nmax; nstep++){

        if ((flag_eq == 0) && (nstep * _dt >= teq)){
            flush_measure();
            flag_eq = 1;
        }

        update_rk4(nstep, 0);
        measure(nstep, &neuron);
    }

    summary_t obj = flush_measure();

    char fname[100];
    sprintf(fname, "id%06d_result.txt", simul_id);
    export_result(&obj, path_join(fdir, fname));
    destroy_neuralnet();

    return 0;
}


void set_parent_dir(char *parent_dir){
    strcpy(fdir, parent_dir);
}


void set_tmax(double _tmax){
    tmax = _tmax;
}

void set_taue(double tr, double td){
    taur_e = tr;
    taud_e = td;
}

void set_taui(double tr, double td){
    taur_i = tr;
    taud_i = td;
}

nn_info_t set_simulation(simul_info_t info){
    // nn_info_t info = {0,};
    nn_info_t nn_info = get_empty_info();

    nn_info.N = N;
    nn_info.num_types = 2;
    nn_info.type_range[0] = N * 0.8;
    nn_info.type_range[1] = N;

    for (int i=0; i<2; i++){
        for (int j=0; j<2; j++){
            nn_info.p_out[i][j] = info.p_out[i][j];
            nn_info.w[i][j] = info.w[i][j];
        }
    }

    nn_info.t_lag = info.t_lag;
    nn_info.nu_ext_mu = info.nu_ext_mu;
    nn_info.w_ext_mu  = info.w_ext_mu;
    nn_info.nu_ext_sd = nu_ext_sd;
    nn_info.w_ext_sd  = w_ext_sd;
    nn_info.const_current = false;
    
    nn_info.taur[0] = taur_e;
    nn_info.taud[0] = taud_e;
    nn_info.taur[1] = taur_i;
    nn_info.taud[1] = taud_i;

    return nn_info;
}
