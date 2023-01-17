#ifndef _neuralnet
#define _neuralnet
#include <stdbool.h>
#include <stdio.h>
#include "model2.h"
#include "storage.h"

#define MAX_TYPE 4


typedef struct _neuralnet_info_t{
    int N;
    int num_types;
    int type_range[MAX_TYPE];
    double mdeg_in[MAX_TYPE][MAX_TYPE];
    double p_out[MAX_TYPE][MAX_TYPE];
    double w[MAX_TYPE][MAX_TYPE]; // pre -> post
    double taur[MAX_TYPE];
    double taud[MAX_TYPE];
    double t_lag;
    double nu_ext_mu, nu_ext_sd;
    double w_ext_mu, w_ext_sd;
    bool const_current;
} nn_info_t;

// nn_info_t get_empty_info(int N, int _num_types);
nn_info_t init_build_info(int N, int _num_types);
void build_ei_rk4(nn_info_t *info);
void write_info(nn_info_t *info, char *fname);
void update_rk4(int nstep, double iapp);
void destroy_neuralnet(void);
void write_all_vars(int nstep, FILE *fp); // -> for debugging
#endif