#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "model2.h"
#include "neuralnet.h"
#include "utils.h"
#include "storage.h"

#define MAX_CLASS_M 5
#define MAX_CHECK_M 10

typedef struct _summary_t {
    // 0 index is for total neurons
    int num_types;
    float chi[MAX_CLASS_M];
    float frs_m[MAX_CLASS_M];
    float frs_s[MAX_CLASS_M];
    float cv_isi[MAX_CLASS_M];
    // float spk_sync[MAX_CLASS_M][MAX_CLASS_M];
} summary_t;


void init_measure(int N, int num_steps, int _num_class_types, int *_type_range);
void reset();
void add_checkpoint(int nstep);
void add_pmeasure(int NP);
// void measure(int nstep, wbneuron_t *neuron);
void measure(int nstep, nnpop_t *nnpop); 
summary_t flush_measure(void);
void destroy_measure(void);
void print_num_check();
void export_core_result(const char *fname, summary_t *obj);
void export_spike(const char *fname);
void export_lfp(const char *fname);
void export_lfp_syn(const char *fname);
void export_result(summary_t *obj, const char *fname);
void test_print(summary_t *obj);