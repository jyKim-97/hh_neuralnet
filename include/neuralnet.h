#ifndef _neuralnet
#define _neuralnet
#include <stdbool.h>
#include <stdio.h>
#include "model2.h"

#define MAX_TYPE 4


typedef struct _neuralnet_info_t{
    int N;
    int num_types;
    int type_range[MAX_TYPE];
    double mdeg_in[MAX_TYPE][MAX_TYPE];
    double w[MAX_TYPE][MAX_TYPE]; // pre -> post
    double t_lag;
    double nu_ext, w_ext;
    int const_current;
} nn_info_t;


void build_rk4(nn_info_t *info);
void write_info(nn_info_t *info, char *fname);
void update_rk4(int nstep, double iapp);
void destroy_neuralnet(void);
#endif