#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "build.h"
#include "mt64.h"


neuron_t neuron;
syn_t syn_i;
int N = 2;
double w = 0.1;
double t_lag = 2;

void init_pop(void);
void destroy_pop(void);
void update_pop(int nstep, double *iapp);
void run(double tmax);

// utilities
double *gen_current(void);
double *get_vcopy(void);

// FILE *fp_v, *fp_ic;

int main(){
    run(100);
}


void run(double tmax){
    int nmax = tmax / _dt;

    FILE *fp_v = fopen("./v_out.dat", "wb");
    FILE *fp_i = fopen("./i_app.dat", "wb");
    FILE *fp_synr = fopen("./syn_r.dat", "wb");
    FILE *fp_synd = fopen("./syn_d.dat", "wb");

    init_pop();
    for (int n=0; n<nmax; n++){
        double *iapp = gen_current();
        update_pop(n, iapp);

        fwrite(neuron.v, sizeof(double), N, fp_v);
        fwrite(iapp, sizeof(double), N, fp_i);
        fwrite(syn_i.expr, sizeof(double), N, fp_synr);
        fwrite(syn_i.expd, sizeof(double), N, fp_synd);
        

        free(iapp);
    }
    destroy_pop();

    fclose(fp_v);
    fclose(fp_i);
    fclose(fp_synr);
    fclose(fp_synd);
}


double *gen_current(void){
    double *iapp = (double*) malloc(sizeof(double) * N);
    // for (int n=0; n<N; n++){
    //     iapp[n] = genrand64_normal(1, 0.1);
    // }
    iapp[0] = 1;
    iapp[1] = 0;
    return iapp;
}


void init_pop(void){
    build_wb_ipop(N, &neuron, &syn_i, w, t_lag, Euler);
    for (int n=0; n<N; n++){
        neuron.v[n] = genrand64_normal(-70, 10);
    }

    for (int npre=0; npre<N; npre++){
        for (int id=0; id<N-1; id++){
            printf("%3d", syn_i.ntk.adj_list[npre][id]);
        }
        printf("\n");
    }
}

void destroy_pop(void){
    destroy_wbNeuron(&neuron);
    destroy_deSyn(&syn_i);
}


void update_pop(int nstep, double *iapp){
    // save vpop
    double *v_prev = get_vcopy();

    // add spike to syn_t    
    add_spike_deSyn(&syn_i, nstep, &(neuron.buf));

    // update 
    double *ptr_v = neuron.v;
    double *ptr_h = neuron.h_ion;
    double *ptr_n = neuron.n_ion;

    for (int n=0; n<N; n++){
        double isyn = get_current_deSyn(&syn_i, n, *ptr_v);
        double dv = solve_wb_v(*ptr_v, iapp[n]-isyn, *ptr_h, *ptr_n);
        double dh = solve_wb_h(*ptr_h, *ptr_v);
        double dn = solve_wb_n(*ptr_n, *ptr_v);
        update_deSyn(&syn_i, n);

        // if (n == 1){
        //     printf("isyn = %5.2f\n", isyn);
        // }

        *ptr_v += dv;
        *ptr_h += dh;
        *ptr_n += dn;

        ptr_v++; ptr_h++; ptr_n++;
    }
     
    update_spkBuf(nstep, &(neuron.buf), v_prev, neuron.v);
    free(v_prev);
}


double *get_vcopy(void){
    double *v_prev = (double*) malloc(sizeof(double) * N);
    memcpy(v_prev, neuron.v, sizeof(double) * N);
    return v_prev;
}

// excitatory pop 추가