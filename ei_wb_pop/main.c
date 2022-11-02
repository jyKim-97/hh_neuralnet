#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "build.h"
#include "rng.h"
#include "utils.h"
#include "storage.h"
#include "mt64.h"
#include "measurement.h"

extern neuron_t neuron;
extern syn_t syn[MAX_TYPE];

#define PRINT_ALL_VAR

int N = 1000;
double w = 0.002;
const double iapp = 0;

double w_ext = 0.001;
double nu_ext = 2000; // 2000 Hz
double *lambda_ext = NULL;

void run(double tmax);
void init_simulation(void);
void update_pop(int nstep);
double get_syn_current(int nid, double v);
void end_pop();
void write_reading(reading_t obj_r);


// NOTE: monitoring 시스템 따로 빼기
void init_check();
void write_data_for_check(int nstep);
void end_check();

// module간 dependency 없애기

// write down data
FILE *fp_v;
FILE *fp_syn_e;
FILE *fp_syn_i;
FILE *fp_syn_ext;

#ifdef PRINT_ALL_VAR
FILE *fp_v0, *fp_n0, *fp_h0, *fp_i0;
const int target_id = 0;
#endif

int main(){
    set_seed(1000);
    run(60000);
}


void run(double tmax){
    int nmax = tmax / _dt;

    int *types = (int*) calloc(N, sizeof(int));
    for (int n=(int) N*0.8; n<N; n++) types[n] = 1;
    init_measure(N, nmax, 2, types);
    init_simulation();
    init_check();

    progbar_t bar;
    init_progressbar(&bar, nmax);
    for (int n=0; n<nmax; n++){
        update_pop(n);
        measure(n, &neuron);
        write_data_for_check(n);

        #ifdef PRINT_ALL_VAR
        fprintf(fp_v0, "%f,", neuron.v[target_id]);
        fprintf(fp_n0, "%f,", neuron.n_ion[target_id]);
        fprintf(fp_h0, "%f,", neuron.h_ion[target_id]);
        #endif

        progressbar(&bar, n);
    }
    printf("\nEnd\n");

    reading_t obj_read = flush_measure();
    write_reading(obj_read);
    free_reading(&obj_read);
    // printf("%f\n", obj_read.frs_m[1]);

    // extern int **step_spk;
    // printf("first t: %d\n", step_spk[0][0]);
    export_spike("./spike");

    free(types);
    end_pop();
}


void write_reading(reading_t obj_r){
    char fname[] = "./result.txt";
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "chi,frs_m,frs_s\n");
    for (int n=0; n<2; n++){
        fprintf(fp, "%f,%f,%f,\n", obj_r.chi[n], obj_r.frs_m[n], obj_r.frs_s[n]);
    }
    fclose(fp);
}


void init_simulation(void){
    buildInfo info = {0,};
    info.N = N;
    info.buf_size = 1/_dt; // 1 ms
    info.ode_method = RK4;

    info.num_types[0] = info.N * 0.8;
    info.num_types[1] = info.N * 0.2;

    for (int i=0; i<2; i++){
        for (int j=0; j<2; j++){
            info.mean_outdeg[i][j] = 10;
            info.w[i][j] = w;
        }
    }

    build_eipop(&info);

    // set external input info
    lambda_ext = (double*) malloc(sizeof(double) * N);
    for (int n=0; n<N; n++){
        lambda_ext[n] = _dt/1000.*nu_ext;
    }

    init_deSyn(N, 0, _dt/2., &ext_syn);
}


void update_pop(int nstep){

    double *v_prev = copy_array(N, neuron.v);

    // add spike to syn_t    
    add_spike_deSyn(&(syn[0]), nstep, &(neuron.buf));
    add_spike_deSyn(&(syn[1]), nstep, &(neuron.buf));

    // update 
    double *ptr_v = neuron.v;
    double *ptr_h = neuron.h_ion;
    double *ptr_n = neuron.n_ion;

    // get poisson input
    int *num_ext = get_poisson_array(N, lambda_ext);

    for (int n=0; n<N; n++){

        // if (*ptr_v == nan){
        if (isnan(*ptr_v)){
            printf("\nERROR: nan detected in Neuron %d in step %d\n", n, nstep);
            end_check();
            exit(1);
        }

        ext_syn.expr[n] += w_ext * ext_syn.A * num_ext[n];
        ext_syn.expd[n] += w_ext * ext_syn.A * num_ext[n];

        // RK4 method
        // 1st step
        // double isyn = get_current_deSyn(syn, n, *ptr_v);
        double isyn = get_syn_current(n, *ptr_v);
        // double isyn = 0;
        double dv1 = solve_wb_v(*ptr_v, iapp-isyn, *ptr_h, *ptr_n);
        double dh1 = solve_wb_h(*ptr_h, *ptr_v);
        double dn1 = solve_wb_n(*ptr_n, *ptr_v);
        
        // 2nd step
        update_deSyn(syn, n);
        update_deSyn(syn+1, n);
        update_deSyn(&ext_syn, n);
        isyn = get_syn_current(n, *ptr_v);

        double dv2 = solve_wb_v(*ptr_v+dv1*0.5, iapp-isyn, *ptr_h+dh1*0.5, *ptr_n+dn1*0.5);
        double dh2 = solve_wb_h(*ptr_h+dh1*0.5, *ptr_v+dv1*0.5);
        double dn2 = solve_wb_n(*ptr_n+dn1*0.5, *ptr_v+dv1*0.5);

        // 3rd step
        isyn = get_syn_current(n, *ptr_v);
        // isyn = 0;
        double dv3 = solve_wb_v(*ptr_v+dv2*0.5, iapp-isyn, *ptr_h+dh2*0.5, *ptr_n+dn2*0.5);
        double dh3 = solve_wb_h(*ptr_h+dh2*0.5, *ptr_v+dv2*0.5);
        double dn3 = solve_wb_n(*ptr_n+dn2*0.5, *ptr_v+dv2*0.5);

        // 4th step
        update_deSyn(syn, n);
        update_deSyn(syn+1, n);
        update_deSyn(&ext_syn, n);
        isyn = get_syn_current(n, *ptr_v);
        // isyn = 0;
        double dv4 = solve_wb_v(*ptr_v+dv3, iapp-isyn, *ptr_h+dh3, *ptr_n+dn3);
        double dh4 = solve_wb_h(*ptr_h+dh3, *ptr_v+dv3);
        double dn4 = solve_wb_n(*ptr_n+dn3, *ptr_v+dv3);

        #ifdef PRINT_ALL_VAR
        if (n == target_id) fprintf(fp_i0, "%f,", iapp-isyn);
        #endif
        
        *ptr_v += (dv1 + 2*dv2 + 2*dv3 + dv4)/6.;
        *ptr_h += (dh1 + 2*dh2 + 2*dh3 + dh4)/6.;
        *ptr_n += (dn1 + 2*dn2 + 2*dn3 + dn4)/6.;

        ptr_v++; ptr_h++; ptr_n++;
    }

    update_spkBuf(nstep, &(neuron.buf), v_prev, neuron.v);
    free(v_prev);
    free(num_ext);
}


double get_syn_current(int nid, double v){
    double isyn = get_current_deSyn(syn, nid, v);
    isyn += get_current_deSyn(syn+1, nid, v);
    isyn += get_current_deSyn(&ext_syn, nid, v);
    return isyn;
}


void end_pop(){
    free(lambda_ext);
    destroy_wbNeuron(&neuron);
    destroy_deSyn(syn);
    destroy_deSyn(syn+1);
    free_measure();
    end_check();
}


void init_check(){
    fp_v = fopen("./check_v.dat", "wb");
    fp_syn_e = fopen("./check_syn_e.dat", "wb");
    fp_syn_i = fopen("./check_syn_i.dat", "wb");
    fp_syn_ext = fopen("./check_syn_ext.dat", "wb");

    #ifdef PRINT_ALL_VAR
    fp_v0 = fopen("./single_v.txt", "w");
    fp_i0 = fopen("./single_i.txt", "w");
    fp_n0 = fopen("./single_n.txt", "w");
    fp_h0 = fopen("./single_h.txt", "w");
    #endif
}


void write_data_for_check(int nstep){
    // write r
    double *r_syn_e = (double*) malloc(sizeof(double) * N);
    double *r_syn_i = (double*) malloc(sizeof(double) * N);
    double *r_syn_ext = (double*) malloc(sizeof(double) * N);
    for (int n=0; n<N; n++){
        r_syn_e[n] = syn[0].expr[n] - syn[0].expd[n];
        r_syn_i[n] = syn[1].expr[n] - syn[1].expd[n];
        r_syn_ext[n] = ext_syn.expr[n] - ext_syn.expd[n];
    }

    save(N, nstep, neuron.v, fp_v);
    save(N, nstep, r_syn_e, fp_syn_e);
    save(N, nstep, r_syn_i, fp_syn_i);
    save(N, nstep, r_syn_ext, fp_syn_ext);
}


void end_check(){
    fclose(fp_v);
    fclose(fp_syn_e);
    fclose(fp_syn_i);
    fclose(fp_syn_ext);

    #ifdef PRINT_ALL_VAR
    fclose(fp_v0);
    fclose(fp_i0);
    fclose(fp_n0);
    fclose(fp_h0);
    #endif
}
