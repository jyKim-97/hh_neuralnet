#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "model2.h"
#include "utils.h"
#include "storage.h"

#define LEN 5

typedef struct _summary_t {
    // 0 index is for total neurons
    int num_types;
    float chi[LEN];
    float frs_m[LEN];
    float frs_s[LEN];
    float cv_isi[LEN];
    float spk_sync[LEN][LEN];
} summary_t;


void init_measure(int N, int num_steps, int _n_class, int *_id_class);
void set_class(int _n_class, int *_id_class);
void init_spike();
void init_flct();
void free_spike();
void free_flct();
void reset();
void measure(int nstep, wbneuron_t *neuron);
float average(float *x);
summary_t flush_measure(void);
void destroy_measure(void);

void calculate_cv_isi(summary_t *obj);
void calculate_spike_sync(summary_t *obj);
void calculate_flct(summary_t *obj);
void calculate_firing_rate(summary_t *obj);

void export_spike(const char *fname);
void export_lfp(const char *fname);
void export_result(summary_t *obj, const char *fname);
void test_print(summary_t *obj);