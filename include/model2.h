#ifndef _model
#define _model

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "ntk.h"
#include "rng.h"
// #include "mt64.h"

extern double _dt;
#define _spk_buf_size 500

#define wb_phi 5
#define wb_el -65
#define wb_ena 55
#define wb_ek -90

typedef struct _wbparams_t{
    double phi, cm, gl, gna, gk;
} wbparams_t;


typedef struct _wbneuron_t{
    // 3-dimensional model (WB)
    int N;
    double *vs, *hs, *ns;
    wbparams_t *params;
    // spike
    int spk_count;
    bool *is_spk;
    int *spk_buf[_spk_buf_size]; // buffer size X (number of the neurons)

} wbneuron_t;


typedef struct _desyn_t{
    int N;
    // network
    int *num_indeg;
    int **indeg_list;
    int pre_range[2], post_range[2];
    // coupling strength
    bool is_const_w;
    double w, **w_list;
    // delay
    bool is_const_delay;
    int n_delay, **n_delays;
    // check point
    bool load_ntk, load_w, load_delay, load_attrib;
    // parameters
    double *expr, *expd;
    double ev, taur, taud;
    double A, mul_expr, mul_expd;
    // for external poisson input
    double nu, expl;
    #ifdef USE_MKL
    double *lambda;
    #endif
} desyn_t;

#define THRESHOLD 0
#define FIRE(vold, vnew) ((vold-THRESHOLD < 0) && (vnew-THRESHOLD > 0))

// Neurons
void init_wbneuron(int N, wbneuron_t *neuron);
void destroy_wbneuron(wbneuron_t *neuron);
void set_default_wbparams(wbparams_t *params);
double solve_wb_v(wbparams_t *params, double v, double h, double n, double iapp);
double solve_wb_h(wbparams_t *params, double h, double v);
double solve_wb_n(wbparams_t *params, double n, double v);
void check_fire(wbneuron_t *neuron, double *v_prev);

// Synapse
void init_desyn(int N, desyn_t *syn);
void destroy_desyn(desyn_t *syn);
void set_attrib(desyn_t *syn, double ev, double taur, double taud, double ode_factor);
void set_network(desyn_t *syn, ntk_t *ntk);
void set_const_coupling(desyn_t *syn, double w);
void set_coupling(desyn_t *syn, int pre_range[2], int post_range[2], double target_w);
void set_const_delay(desyn_t *syn, double td);
void set_delay(desyn_t *syn, int pre_range[2], int post_range[2], double target_td);
void add_spike(int nstep, desyn_t *syn, wbneuron_t *neuron);
void update_desyn(desyn_t *syn, int nid);
double get_current(desyn_t *syn, int nid, double vpost);

// Pos synapse
void init_extsyn(int N, desyn_t *syn);
void set_poisson(desyn_t *ext_syn, double nu, double w);
void add_ext_spike(desyn_t *ext_syn);
void print_syn_network(desyn_t *syn, char *fname);

// void check_fire(wbneuron_t *neuron, double *prev_vs);


// void add(int N, double *x, double b, double *dx);
// double solve_wb_v(double v, double h_ion, double n_ion, double I);
// double solve_wb_h(double h_ion, double v);
// double solve_wb_n(double n_ion, double v);


#endif