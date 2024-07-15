#ifndef _neuralnet
#define _neuralnet
#include <stdbool.h>
#include <stdio.h>
#include "model2.h"
#include "storage.h"
#include "utils.h"

#define MAX_TYPE 8

#define E 0
#define IF 1
#define IS 2

typedef struct _nnpop_t
{
    wbneuron_t *neuron;
    desyn_t *syns;
    desyn_t *ext_syn;
    int N;
    int num_types;

} nnpop_t;



typedef struct _neuralnet_info_t{
    int N;
    int num_types;
    int type_range[MAX_TYPE];
    int type_id[MAX_TYPE]; // 0 (E), 1 (IF), 2(IS)
    double mdeg_in[MAX_TYPE][MAX_TYPE];
    double p_out[MAX_TYPE][MAX_TYPE];
    double w[MAX_TYPE][MAX_TYPE]; // pre -> post
    double taur[MAX_TYPE];
    double taud[MAX_TYPE];
    double t_lag;
    double nu_ext_mu, nu_ext_sd;
    double w_ext_mu, w_ext_sd;

    double nu_ext_multi[MAX_TYPE];
    double w_ext_multi[MAX_TYPE];

    bool const_current;
    int num_ext_types;

} nn_info_t;

void init_nn(int N, int _num_types);
// void init_nn(nn_info_t *info);
nn_info_t init_build_info(int N, int _num_types);
void build_ei_rk4(nn_info_t *info);
void write_info(nn_info_t *info, char *fname);
void update_rk4(int nstep, double iapp);
void destroy_neuralnet(void);
void write_all_vars(int nstep, FILE *fp); // -> for debugging

void set_multiple_ext_input(nn_info_t *info, int type_id, int num_targets, int *target_id);
void check_multiple_input();

#endif