#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "build.h"
#include "mt64.h"
#include "utils.h"
#include "storage.h"


#define _print_params
#define fname_param(fname) "./check_params/"fname


// neuron_t neuron;
// syn_t syn_i={0,}, syn_e={0,};
extern neuron_t neuron;
extern syn_t syn[MAX_TYPE];

#define _test

int N = 100;
double w = 0.1;
int n_lag = 1/_dt;

void init_pop(void);
void destroy_pop(void);
void update_pop(int nstep, double *iapp);
void run(double tmax);

// utilities
double get_syn_current(int nid, double v);
double *get_vcopy(void);

// dsave
void save_network(void);
float *cast2float(int N, double *arr);

// for testing
void test_synfire(void);

int main(){
    init_genrand64(1000);
    run(500);
    return 0;
}


void run(double tmax){
    init_pop();

    #ifdef _test
    test_synfire(); // test
    FILE *fv = fopen("./v_out_synfire.dat", "wb");
    #else
    FILE *fv = fopen("./v_out.dat", "wb");
    #endif
    // t_delay 넣어주기

    double *iapp = (double*) calloc(N, sizeof(double));
    iapp[0] = 1;
    // for (int n=0; n<N; n++) iapp[n] = 0.5;
    
    int nmax = tmax/_dt;
    progbar_t bar;
    init_progressbar(&bar, nmax);

    for (int n=0; n<nmax; n++){
        update_pop(n, iapp);
        // save parameters
        // fwrite(neuron.v, sizeof(double), N, fv);
        // save(N, n, neuron.v, fv);

        progressbar(&bar, n);
    }
    fprintf(stderr, "\n");
    
    fclose(fv);
    free(iapp);
    destroy_pop();
}


buildInfo init_info(void){
    buildInfo info = {0,};
    info.N = N;
    info.buf_size = n_lag;
    info.ode_method = RK4;

    info.num_types[0] = info.N * 0.8;
    info.num_types[1] = info.N * 0.2;

    for (int i=0; i<2; i++){
        for (int j=0; j<2; j++){
            info.mean_outdeg[i][j] = 10;
            info.w[i][j] = w;
        }
    }

    return info;
}

void init_pop(void){
    buildInfo info = init_info();
    build_eipop(&info);

    #ifdef _print_params
    save_network();
    #endif    
}


void save_network(void){
    print_network(fname_param("net_syn_e.txt"), &(syn[0].ntk));
    print_network(fname_param("net_syn_i.txt"), &(syn[1].ntk));
}


void destroy_pop(void){
    destroy_wbNeuron(&neuron);
    destroy_deSyn(syn);
    destroy_deSyn(syn+1);
    // destroy_deSyn(&syn_i);
    // destroy_deSyn(&syn_e);
}


void update_pop(int nstep, double *iapp){
    // save vpop
    double *v_prev = get_vcopy();

    // add spike to syn_t    
    add_spike_deSyn(&(syn[0]), nstep, &(neuron.buf));
    add_spike_deSyn(&(syn[1]), nstep, &(neuron.buf));

    // update 
    double *ptr_v = neuron.v;
    double *ptr_h = neuron.h_ion;
    double *ptr_n = neuron.n_ion;

    for (int n=0; n<N; n++){
        // RK4 method
        // 1st step
        // double isyn = get_current_deSyn(syn, n, *ptr_v);
        double isyn = get_syn_current(n, *ptr_v);
        double dv1 = solve_wb_v(*ptr_v, iapp[n]-isyn, *ptr_h, *ptr_n);
        double dh1 = solve_wb_h(*ptr_h, *ptr_v);
        double dn1 = solve_wb_n(*ptr_n, *ptr_v);
        
        // 2nd step
        update_deSyn(syn, n);
        update_deSyn(syn+1, n);
        isyn = get_syn_current(n, *ptr_v);

        double dv2 = solve_wb_v(*ptr_v+dv1*0.5, iapp[n]-isyn, *ptr_h+dh1*0.5, *ptr_n+dn1*0.5);
        double dh2 = solve_wb_h(*ptr_h+dh1*0.5, *ptr_v+dv1*0.5);
        double dn2 = solve_wb_n(*ptr_n+dn1*0.5, *ptr_v+dv1*0.5);

        // 3rd step
        isyn = get_syn_current(n, *ptr_v);
        double dv3 = solve_wb_v(*ptr_v+dv2*0.5, iapp[n]-isyn, *ptr_h+dh2*0.5, *ptr_n+dn2*0.5);
        double dh3 = solve_wb_h(*ptr_h+dh2*0.5, *ptr_v+dv2*0.5);
        double dn3 = solve_wb_n(*ptr_n+dn2*0.5, *ptr_v+dv2*0.5);

        // 4th step
        update_deSyn(syn, n);
        update_deSyn(syn+1, n);
        isyn = get_syn_current(n, *ptr_v);
        double dv4 = solve_wb_v(*ptr_v+dv3, iapp[n]-isyn, *ptr_h+dh3, *ptr_n+dn3);
        double dh4 = solve_wb_h(*ptr_h+dh3, *ptr_v+dv3);
        double dn4 = solve_wb_n(*ptr_n+dn3, *ptr_v+dv3);
        

        *ptr_v += (dv1 + 2*dv2 + 2*dv3 + dv4)/6.;
        *ptr_h += (dh1 + 2*dh2 + 2*dh3 + dh4)/6.;
        *ptr_n += (dn1 + 2*dn2 + 2*dn3 + dn4)/6.;

        ptr_v++; ptr_h++; ptr_n++;
    }
     
    update_spkBuf(nstep, &(neuron.buf), v_prev, neuron.v);
    free(v_prev);
}


double get_syn_current(int nid, double v){
    double isyn = get_current_deSyn(syn, nid, v);
    isyn += get_current_deSyn(syn+1, nid, v);
    return isyn;
}


double *get_vcopy(void){
    double *v_prev = (double*) malloc(sizeof(double) * N);
    memcpy(v_prev, neuron.v, sizeof(double) * N);
    return v_prev;
}


float *cast2float(int N, double *arr){
    float *float_arr = (float*) malloc(sizeof(float) * N);
    for (int n=0; n<N; n++){
        float_arr[n] = (float) arr[n];
    }
    return float_arr;
}


void test_synfire(void){
    for (int npre=0; npre<N; npre++){
        int npost_e = npre < N? npre+1: 0;
        int npost_i = npre > 0? npre-1: N-1;

        syn[0].ntk.adj_list[npre][0] = npost_e;
        syn[1].ntk.adj_list[npre][0] = npost_i;

        for (int i=0; i<2; i++){
            syn[i].ntk.num_edges[i] = 1;
            syn[i].ntk.weight_list[npre][0] = w;
            syn[i].ntk.n_delay[npre][0] = 1/_dt;
        }
    }
}