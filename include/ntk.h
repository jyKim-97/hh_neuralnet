#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "mt64.h"

#ifndef _NTK
#define _NTK

#ifdef __cplusplus
extern "C" {
#endif

#define ONE2ONE -2
#define NULL_CON -3

enum binet_type {
    outdeg = 0,
    indeg = 1
};


typedef struct _ntk_t{
    int N;
    int *num_edges; 
    int **adj_list;
    enum binet_type edge_dir;
} ntk_t;


ntk_t get_empty_net(int N);
void free_network(ntk_t *ntk);
void gen_er_pout(ntk_t *ntk, double p_out, int pre_range[2], int post_range[2]);
void gen_er_pin(ntk_t *ntk, double p_in, int pre_range[2], int post_range[2]);
void gen_er_mdout(ntk_t *ntk, double mdeg_out, int pre_range[2], int post_range[2]);
double cvt_mdeg_out2in(double mdeg_out, int num_pre, int num_post);
void gen_er_mdin(ntk_t *ntk, double mdeg_in, int pre_range[2], int post_range[2]);
void print_network(const char *fname, ntk_t *ntk);

#ifdef __cplusplus
}
#endif

#endif