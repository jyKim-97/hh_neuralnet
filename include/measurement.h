/*
Source code for measurement

Accessible options
- SPK
- LFP
- V_FLUCT
- FIRING_RATE
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "model.h"
#include "utils.h"

typedef struct _reading_t {
    int num_types;
    double *chi; // the coherence of fluctuation (num_types, )
    double *frs_m; // the firing rate
    double *frs_s; // the firing rate -> firing irregularity
} reading_t;

#define SPK
#define FIRING_RATE
#define LFP
#define V_FLUCT

static void init_spk(void);
static void init_fluct(void);
static void init_lfp(int total_step);
void init_measure(int N, int total_step, int n_class, int *id_class);
void measure(int nstep, neuron_t *neuron);

reading_t init_reading();
reading_t flush_measure();
void calculate_fluct(reading_t *obj_r);
void calculate_firing_rate(reading_t *obj_r);

void reset_fluct();
void reset_firing_rate();

static void free_spk(void);
static void free_fluct();
static void free_lfp(void);
void free_measure();
void free_reading(reading_t *obj_r);