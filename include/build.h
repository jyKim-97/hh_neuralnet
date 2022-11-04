#ifndef _BUILD
#define _BUILD

#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "mt64.h"
#include "ntk.h"

#define MAX_TYPE 4

extern neuron_t neuron;
extern syn_t syn[MAX_TYPE];
extern syn_t ext_syn;


enum odeType {
    Euler = 0,
    RK4
};


typedef struct _buildInfo{
    int N;
    int num_types[MAX_TYPE];
    int buf_size;
    double mdeg_out[MAX_TYPE][MAX_TYPE];
    int n_lag[MAX_TYPE][MAX_TYPE];
    // double prob[MAX_TYPE][MAX_TYPE]; // pre -> post
    double w[MAX_TYPE][MAX_TYPE]; // pre -> post
    enum odeType ode_method;

} buildInfo;


void build_wb_ipop(int N, neuron_t *neuron, syn_t *syn_i, double w, double t_lag, enum odeType type);
void build_homogen_net(netsyn_t *ntk, double w, int n_lag);

void build_eipop(buildInfo *info);
void build_randomnet(netsyn_t *ntk, double mean_outdeg, double w, int n_lag, int pre_range[2], int post_range[2]);
void print_syn(char fname[], netsyn_t *ntk);

#endif