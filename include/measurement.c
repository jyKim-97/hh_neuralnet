/*
Source code for measurement
*/

#include "measurement.h"


int cum_steps = 0;

// population id
int size_pops = 0;
int num_pop_types = 1;
int *id_pops = NULL;
int *num_pops = NULL;

// SPIKE recording
int **step_spk = NULL; // (N, ?)
int *num_spk = NULL; // (N, )
int *cum_spk = NULL; // measure the firing rate

double **v_lfp; // (num_times, num_pop_types)
double *v_m1, *v_m2; // (num_popos)
double *v_fluct_m1, *v_fluct_m2; // (N,)


void init_measure(int N, int total_step, int n_class, int *id_class){
    size_pops = N;
    num_pop_types = n_class;
    id_pops = id_class;
    num_pops = (int*) calloc(num_pop_types, sizeof(int));
    for (int n=0; n<N; n++){
        int id = id_pops[n];
        num_pops[id]++;
    }

    init_spk();
    cum_spk = (int*) calloc(N, sizeof(int));
    init_fluct();
    init_lfp(total_step);
}


void measure(int nstep, neuron_t *neuron){

    int n_buf = nstep % (neuron->buf.buf_size);
    double *ptr_v = neuron->v;
    for (int n=0; n<size_pops; n++){
        int id = id_pops[n];
        double v1 = ptr_v[n];

        v_lfp[id][nstep]  += v1;
        v_fluct_m1[n] += v1;
        v_fluct_m2[n] += v1 * v1;

        // spike recording
        if (neuron->buf.spk_buf[n][n_buf] == 0) continue;
        append_int(step_spk+n, num_spk[n], nstep);
        num_spk[n]++;
        cum_spk[n]++;
    }

    for (int id=0; id<num_pop_types; id++){
        v_lfp[id][nstep] /= num_pops[id];
        double V = v_lfp[id][nstep];
        v_m1[id] += V;
        v_m2[id] += V*V;
    }

    cum_steps++;
}


reading_t flush_measure(){
    reading_t obj_r = init_reading();
    calculate_fluct(&obj_r);
    calculate_firing_rate(&obj_r);
    // reset values
    reset_fluct();
    reset_firing_rate();
    cum_steps = 0;

    return obj_r;
}


void calculate_fluct(reading_t *obj_r){
    // calculate fluctuation
    double *var_indiv = (double*) calloc(num_pop_types, sizeof(double));
    for (int n=0; n<size_pops; n++){
        double vm = v_fluct_m1[n]/(double) cum_steps;
        double var = v_fluct_m2[n]/(double) cum_steps - vm*vm;
        int id = id_pops[n];
        var_indiv[id] += var;
    }

    for (int id=0; id<num_pop_types; id++){
        double vm = v_m1[id]/(double) cum_steps;
        double var = v_m2[id]/(double) cum_steps - vm*vm;
        obj_r->chi[id] = sqrt(var * num_pop_types / var_indiv[id]);
    }
    free(var_indiv);
}


void reset_fluct(){
    for (int id=0; id<num_pop_types; id++){
        v_m1[id] = 0;
        v_m2[id] = 0;
        v_fluct_m1[id] = 0;
        v_fluct_m2[id] = 0;
    }
}


void reset_firing_rate(){
    for (int n=0; n<size_pops; n++){
        cum_spk[n] = 0;
    }
}


void calculate_firing_rate(reading_t *obj_r){
    // firing rate (Hz)
    double *fr1 = (double*) calloc(num_pop_types, sizeof(double));
    double *fr2 = (double*) calloc(num_pop_types, sizeof(double));

    double div = cum_steps / 1000.;
    for (int n=0; n<size_pops; n++){
        int id = id_pops[n];
        fr1[id] += cum_spk[n]/div/num_pops[id];
        fr2[id] += cum_spk[n]*cum_spk[n]/div/num_pops[id];
    }

    for (int id=0; id<num_pop_types; id++){
        obj_r->frs_m[id] = fr1[id];
        obj_r->frs_s[id] = sqrt(fr2[id] - fr1[id]*fr1[id]);
    }
    free(fr1);
    free(fr2);
}


reading_t init_reading(){
    reading_t obj_r = {0,};
    obj_r.num_types = num_pop_types;
    obj_r.chi = (double*) calloc(num_pop_types, sizeof(double));
    obj_r.frs_m = (double*) calloc(num_pop_types, sizeof(double));
    obj_r.frs_s = (double*) calloc(num_pop_types, sizeof(double));
    return obj_r;
}


void free_reading(reading_t *obj_r){
    free(obj_r->chi);
    free(obj_r->frs_m);
    free(obj_r->frs_s);
}


static void init_spk(void){
    step_spk = (int**) malloc(size_pops * sizeof(int*));
    for (int n=0; n<size_pops; n++) step_spk[n] = (int*) calloc(_block_size, sizeof(int));
    num_spk = (int*) calloc(size_pops, sizeof(int));
}


static void free_spk(void){
    for (int n=0; n<size_pops; n++){
        free(step_spk[n]);
    }
    free(step_spk);
    free(num_spk);
}


static void init_fluct(void){
    v_m1 = (double*) calloc(num_pop_types, sizeof(double));
    v_m2 = (double*) calloc(num_pop_types, sizeof(double));
    v_fluct_m1 = (double*) calloc(size_pops, sizeof(double));
    v_fluct_m2 = (double*) calloc(size_pops, sizeof(double));
}


void free_measure(){
    free_spk();
    free(cum_spk);
    free_fluct();
    free_lfp();
}


static void free_fluct(){
    free(v_m1);
    free(v_m2);
    free(v_fluct_m1);
    free(v_fluct_m2);
}


static void init_lfp(int total_step){
    v_lfp = (double**) malloc(sizeof(double*) * num_pop_types);
    for (int n=0; n<num_pop_types; n++){
        v_lfp[n] = (double*) calloc(total_step, sizeof(double));
    }
}


static void free_lfp(void){
    for (int n=0; n<num_pop_types; n++){
        free(v_lfp[n]);
    }
    free(v_lfp);
}


/* Save part */
void export_spike(const char *tag){

    char fname[200];
    sprintf(fname, "%s_info.txt", tag);
    FILE *fp_spk_info = fopen(fname, "w");
    sprintf(fname, "%s.dat", tag);
    FILE *fp_spk_time = fopen(fname, "wb");

    int num_tot_spk = 0;
    for (int n=0; n<size_pops; n++){
        fprintf(fp_spk_info, "%d,", num_spk[n]);
        num_tot_spk += num_spk[n];
    }
    fclose(fp_spk_info);

    int *tot_spk = (int*) malloc(sizeof(int) * num_tot_spk);
    int cum = 0;
    for (int n=0; n<size_pops; n++){
        for (int i=0; i<num_spk[n]; i++){
            tot_spk[i+cum] = step_spk[n][i];
        }
        cum += num_spk[n];
    }

    fwrite(tot_spk, sizeof(int), num_tot_spk, fp_spk_time);
    fclose(fp_spk_time);

}