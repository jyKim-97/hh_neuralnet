#ifndef _model
#define _model

#include "stdlib.h"
#include "math.h"

#define _dt 0.01


typedef struct _netSyn{
    int N;
    int *num_edges;
    int **adj_list;
    double **weight_list;
    int **n_delay;
    int pre_range[2];
    int post_range[2];
} netsyn_t;


// #define THRESHOLD 20

typedef struct _spkBuf{
    int N; // (number of the neuron)
    int buf_size;
    int **spk_buf; // (number fo the neuron) X (buffer size)
} spkbuf_t;


// Parameters for neuron_t
#define wb_cm 1 // ms
#define wb_phi 5
#define wb_gl 0.1
#define wb_el -65
#define wb_gna 35
#define wb_ena 55
#define wb_gk 9
#define wb_ek -90


typedef struct _neuron_t{
    int N;
    int *types;
    // 3-dimensional model (WB)
    double *v;
    double *h_ion;
    double *n_ion;

    spkbuf_t buf;

} neuron_t;


typedef struct _deSyn{
    int N;
    double *expr; // rising part
    double *expd; // decaying part
    double ev;
    double A; // normalization constant
    double mul_expr;
    double mul_expd;
    char tag[100];
    netsyn_t ntk;

} syn_t;


// double *solve_wbNeuron(double wb_v, double wb_h, double wb_n, double isyn, double iapp);

// Neurons
void init_wbNeuron(int N, int buf_size, neuron_t *neuron);
void destroy_wbNeuron(neuron_t *neuron);
double solve_wb_v(double v, double h_ion, double n_ion, double I);
static inline double get_minf(double v);
double solve_wb_h(double h_ion, double v);
double solve_wb_n(double n_ion, double v);

// spike buffer 
void init_spkBuf(int N, double nd_max, spkbuf_t *buffer);
void destroy_spkBuf(spkbuf_t *buf);
void update_spkBuf(int nstep, spkbuf_t *buf, double *v_old, double *v_new);

// Synapse
void init_deSyn(int N, double ev, double dt, syn_t *syn);
void destroy_deSyn(syn_t *syn);
void add_spike_deSyn(syn_t *syn, int nstep, spkbuf_t *buf);
void update_deSyn(syn_t *syn, int id);
double get_current_deSyn(syn_t *syn, int id, double vpost);

// network
void init_netSyn(int N, netsyn_t *ntk);
void destroy_netSyn(netsyn_t *ntk);

#endif
