/*
Source code for measurement
*/
#include "measurement.h"

int max_step  = -1;
int cum_step  = 0;
int prev_step = 0;

// population 
int ntk_size  = -1; // total # of neurons
int num_class_types = 2;
int *id_class = NULL; // class id for each neurons
int num_class[LEN]; // The number of the cells belongs to the class

float *vlfp[LEN]; // LFP recording
float v_avg1[LEN]; // average voltage avg
float v_avg2[LEN]; // average v^2 avg
double v_tot1, v_tot2;
float *v1, *v2; // each neurons membrane potential (avg, ^2)

// SPIKE recording
int *num_spk   = NULL; // (N, )
int *cum_spk   = NULL; // the number of cumulated spikes
int **step_spk = NULL; // (N, ?)

const float t_spike_bin = 2.; // 5 ms


void init_measure(int N, int num_steps, int _n_class, int *_id_class){
    ntk_size = N;
    max_step = num_steps;
    set_class(_n_class, _id_class);
    
    init_spike();
    init_flct();
    reset();
}


void destroy_measure(void){
    free_spike();
    free_flct();
}


void set_class(int _n_class, int *_id_class){
    id_class = (int*) malloc(sizeof(int) * ntk_size);
    if ((_id_class == NULL) || (_n_class == -1)){
        num_class_types = 2;
        for (int n=0; n<ntk_size; n++) id_class[n] = (n < ntk_size*0.8)? 0: 1;
    } else {
        num_class_types = _n_class;
        memcpy(id_class, _id_class, sizeof(int)*ntk_size);
    }

    // get num_class
    for (int n=0; n<LEN; n++) num_class[n] = 0;

    for (int n=0; n<ntk_size; n++){
        if ((id_class[n] < 0) || (id_class[n] >= LEN-1)){
            printf("Neuron ID is wrong: %d, resize the LEN or check class\n", id_class[n]);
            exit(1);
        }

        if (id_class[n] >= _n_class){
            printf("The class id (%d) is different with expected (%d)\n", id_class[n], num_class_types);
        }

        int id = id_class[n];
        num_class[id]++;
    }
}


void init_spike(){
    cum_spk  = (int*) calloc(ntk_size, sizeof(int));
    num_spk  = (int*) calloc(ntk_size, sizeof(int));
    step_spk = (int**) malloc(ntk_size * sizeof(int*));
    for (int n=0; n<ntk_size; n++){
        step_spk[n] = (int*) calloc(_block_size, sizeof(int));
    }
}


void free_spike(){
    free(cum_spk);
    free(num_spk);
    for (int n=0; n<ntk_size; n++) free(step_spk[n]);
    free(step_spk);
}


void init_flct(){
    v1 = (float*) calloc(ntk_size, sizeof(float));
    v2 = (float*) calloc(ntk_size, sizeof(float));

    for (int id=0; id<num_class_types; id++){
        vlfp[id] = (float*) calloc(max_step, sizeof(float));
    }
}


void free_flct(){
    free(v1);
    free(v2);
    for (int n=0; n<num_class_types; n++) free(vlfp[n]);
}


void reset(){
    for (int n=0; n<num_class_types; n++){
        v_avg1[n] = 0;
        v_avg2[n] = 0;
    }
    v_tot1 = 0;
    v_tot2 = 0;

    for (int n=0; n<ntk_size; n++){
        v1[n] = 0;
        v2[n] = 0;
        cum_spk[n] = 0;
    }

    prev_step += cum_step;
    cum_step = 0;
}


void measure(int nstep, wbneuron_t *neuron){
    
    double v_tmp=0;
    for (int n=0; n<ntk_size; n++){
        double v = neuron->vs[n];
        long double v_pow = v*v;

        // for each id
        int id = id_class[n];
        vlfp[id][nstep] += v;
        v1[n] += v;
        v2[n] += v_pow;
        v_tmp += v;

        // add spike
        if (neuron->is_spk[n]){
            append_int(step_spk+n, num_spk[n], nstep);
            num_spk[n]++;
            cum_spk[n]++;
        }
    }
    v_tmp /= ntk_size;
    v_tot1 += v_tmp;
    v_tot2 += v_tmp*v_tmp;

    for (int id=0; id<num_class_types; id++){
        vlfp[id][nstep] /= num_class[id];
        v_avg1[id] += vlfp[id][nstep];
        v_avg2[id] += vlfp[id][nstep]*vlfp[id][nstep];
    }

    cum_step++;
}


summary_t flush_measure(void){
    summary_t obj = {0,};
    calculate_cv_isi(&obj);
    calculate_firing_rate(&obj);
    calculate_flct(&obj);
    calculate_spike_sync(&obj);
    reset();
    return obj;
}


void calculate_cv_isi(summary_t *obj){
    float *cv = (float*) calloc(ntk_size, sizeof(float));

    for (int n=0; n<ntk_size; n++){
        // ignore bef prev_step
        int nstep = -1, stack=0;
        while (nstep > prev_step){
            if (stack == num_spk[n]){
                break;
            }
            nstep = step_spk[n][stack];
            stack++;
        }
        int num = num_spk[n] - stack;

        if (num < 3){
            cv[n] = -1;
            continue;
        }

        int n_prev = nstep;
        float dn1=0, dn2=0;
        for (int i=stack; i<num_spk[n]; i++){
            nstep = step_spk[n][i];

            float dn = nstep - n_prev;
            dn1 += dn/(float)num;
            dn2 += dn*dn/(float)num;
            n_prev = nstep;
        }

        // double mu = (double) dn1 / (double) num;
        float mu = dn1;
        float s = sqrt(dn2 - dn1*dn1);
        if (isnan(s)){
            printf("nan detected: mu: %5.2f, s: %5.2f\n", mu, s);
        }

        cv[n] = s/mu;
    }

    // total summary
    obj->cv_isi[0] = average(cv);

    float cv_isi_tmp[LEN] = {0,};
    int stack[LEN] = {0,};
    for (int n=0; n<ntk_size; n++){
        if (cv[n] == -1) continue;

        int id = id_class[n];
        cv_isi_tmp[id] += cv[n];
        stack[id]++;
    }

    for (int id=0; id<num_class_types; id++){
        if (stack[id] == 0){
            obj->cv_isi[id+1] = -1;
        } else {
            obj->cv_isi[id+1] = cv_isi_tmp[id]/stack[id];
        }
    }

    free(cv);
}


void calculate_spike_sync(summary_t *obj){
    // for (int n=0; n<)
    int nbin = t_spike_bin / _dt;
    int len  = cum_step / nbin + 1;

    // convert to vector form
    int **t_vec = (int**) malloc(sizeof(int*) * ntk_size);
    int *sum_vec = (int*) calloc(ntk_size, sizeof(int));

    for (int n=0; n<ntk_size; n++){
        t_vec[n] = (int*) calloc(len, sizeof(int));

        for (int i=0; i<num_spk[n]; i++){
            int nstep = step_spk[n][i];
            if (nstep < prev_step) continue;

            int id = (nstep-prev_step) / nbin;
            t_vec[n][id] = 1;
        }
    }

    // calculate spike vector sync (calculate only upper triangle part)
    double *cij = (double*) calloc(ntk_size*ntk_size, sizeof(double));
    for (int i=0; i<ntk_size; i++){
        for (int j=i; j<ntk_size; j++){
            double mul = sum_vec[i] * sum_vec[j];
            if (mul == 0) continue;

            for (int n=0; n<len; n++){
                cij[ntk_size*i+j] += t_vec[i][n] * t_vec[j][n];
            }

            cij[ntk_size*i+j] /= sqrt(mul);
        }
        free(t_vec[i]);
    }
    free(sum_vec);

    // average for each type
    for (int i=0; i<ntk_size; i++){
        int id1 = id_class[i];
        for (int j=i; j<ntk_size; j++){
            int id2 = id_class[j];
            obj->spk_sync[id1][id2] += cij[ntk_size*i+j]/(num_class[id1]*num_class[id2]/2);
        }
    }
    free(cij);
}


void calculate_flct(summary_t *obj){
    float var_indiv[LEN] = {0,};

    for (int n=0; n<ntk_size; n++){
        float vm  = v1[n]/cum_step;
        float var = v2[n]/cum_step - vm*vm;
        int id = id_class[n];
        var_indiv[id] += var;
    }

    double var_indiv_tot=0;
    for (int id=0; id<num_class_types; id++){
        var_indiv_tot += var_indiv[id];
    }

    v_tot2 /= cum_step;
    v_tot1 /= cum_step;
    double var_tot = v_tot2 - v_tot1 * v_tot1;
    obj->chi[0] = sqrt(var_tot * ntk_size/var_indiv_tot);

    // each class
    for (int id=0; id<num_class_types; id++){
        float vm  = v_avg1[id]/cum_step;
        float var = v_avg2[id]/cum_step - vm*vm;
        obj->chi[id+1] = sqrt(var*num_class[id]/var_indiv[id]);
    }
}


void calculate_firing_rate(summary_t *obj){
    float fr1[LEN] = {0,};
    float fr2[LEN] = {0,};

    float m=0, var=0; // for total summary
    float div = cum_step * _dt / 1000.;
    for (int n=0; n<ntk_size; n++){
        int id = id_class[n];
        float x = cum_spk[n]/div;

        m   += x/ntk_size;
        var += x*x/ntk_size;
        fr1[id] += x/num_class[id];
        fr2[id] += x*x/num_class[id];
    }

    // total steps
    obj->frs_m[0] = m;
    obj->frs_s[0] = sqrt(var - m*m);

    for (int id=0; id<num_class_types; id++){    
        obj->frs_m[id] = fr1[id];
        obj->frs_s[id] = sqrt(fr2[id] - fr1[id]*fr1[id]);
    }
}


float average(float *x){
    float xsum= 0;
    int stack = 0;
    for (int n=0; n<ntk_size; n++){
        if (x[n] == -1){
            continue;
        }
        xsum += x[n];
        stack++;
    }
    if (stack == 0){
        return -1;
    } else {
        return xsum / stack;
    }
}


void export_lfp(const char *fname){
    FILE *fp = fopen(fname, "wb");

    float info[2] = {(float) num_class_types, (float) max_step};
    fwrite(info, sizeof(float), 2, fp);

    for (int id=0; id<num_class_types; id++){
        fwrite(vlfp[id], sizeof(float), num_class[id], fp);
    }
    fclose(fp);
}


void export_spike(const char *fname){
    FILE *fp = fopen(fname, "wb");
    int info[2] = {ntk_size, max_step};
    fwrite(info, sizeof(int), 2, fp);
    fwrite(num_spk, sizeof(int), ntk_size, fp);
    for (int n=0; n<ntk_size; n++){
        fwrite(step_spk[n], sizeof(int), num_spk[n], fp);
    }
}


void test_print(summary_t *obj){
    printf("cv: ");
    for (int n=0; n<num_class_types+1; n++){
        printf("%5.3f,", obj->cv_isi[n]);
    }; printf("\n");

    printf("chi: ");
    for (int n=0; n<num_class_types+1; n++){
        printf("%5.3f,", obj->chi[n]);
    }; printf("\n");

    printf("frs_m: ");
    for (int n=0; n<num_class_types+1; n++){
        printf("%5.3f,", obj->frs_m[n]);
    }; printf("\n");

    printf("frs_s: ");
    for (int n=0; n<num_class_types+1; n++){
        printf("%5.3f,", obj->frs_s[n]);
    }; printf("\n");

    printf("spike syn:\n");
    for (int i=0; i<num_class_types+1; i++){
        for (int j=0; j<num_class_types+1; j++){
            printf("%5.3f,", obj->spk_sync[i][j]);
        }
        printf("\n");
    }
}


// void export_spike(const char *tag){

//     char fname[200];
//     sprintf(fname, "%s_info.txt", tag);
//     FILE *fp_spk_info = fopen(fname, "w");
//     sprintf(fname, "%s.dat", tag);
//     FILE *fp_spk_time = fopen(fname, "wb");

//     int num_tot_spk = 0;
//     for (int n=0; n<size_pops; n++){
//         fprintf(fp_spk_info, "%d,", num_spk[n]);
//         num_tot_spk += num_spk[n];
//     }
//     fclose(fp_spk_info);

//     int *tot_spk = (int*) malloc(sizeof(int) * num_tot_spk);
//     int cum = 0;
//     for (int n=0; n<size_pops; n++){
//         for (int i=0; i<num_spk[n]; i++){
//             tot_spk[i+cum] = step_spk[n][i];
//         }
//         cum += num_spk[n];
//     }

//     fwrite(tot_spk, sizeof(int), num_tot_spk, fp_spk_time);
//     fclose(fp_spk_time);
// }


/*

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

    int buf_size = neuron->buf.buf_size;
    int n_buf = (buf_size == 0)? 0: nstep % buf_size;

    double *ptr_v = neuron->v;
    for (int n=0; n<size_pops; n++){
        int id = id_pops[n];
        double v1 = ptr_v[n];

        // printf("v1: %5.2f\n", v1);

        v_lfp[id][nstep] += v1;
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

// spike synchrony measure하는 코드도 필요
void calculate_spike_sync(reading_t *obj){
    // for (int n=0; n<)
    int nbin = t_spike_bin / _dt;
    int len = cum_steps / nbin + 1;

    int **t_vec = (int**) malloc(sizeof(int*) * size_pops);
    for (int n=0; n<size_pops; n++){
        t_vec[n] = (int*) calloc(len, sizeof(int));
    }

    // calculate spike vector
    for (int n=0; n<size_pops; n++){
        for (int i=0; i<num_spk[n]; i++){
            int nstep = step_spk[n][i];
            if (nstep < flushed_steps) continue;

            int id = (nstep-flushed_steps) / nbin;
            t_vec[n][id] = 1;
        }
    }

    int *sum_vec = (int*) calloc(size_pops, sizeof(int));
    for (int n=0; n<size_pops; n++){
        for (int i=0; i<len; i++){
            sum_vec[n] += t_vec[n][i];
        }
    }

    // calculate spike vector sync (calculate only upper triangle part)
    double *cij = (double*) calloc(size_pops*size_pops, sizeof(double));
    for (int i=0; i<size_pops; i++){
        for (int j=i; j<size_pops; j++){
            for (int n=0; n<len; n++){
                cij[size_pops*i+j] += t_vec[i][n] * t_vec[j][n];
            }
            double mul = sum_vec[i] * sum_vec[j];
            if (mul == 0) continue;

            cij[size_pops*i+j] /= sqrt(mul);
        }
    }

    // average for each type
    int *cum = (int*) calloc(num_pop_types*num_pop_types, sizeof(int));
    for (int i=0; i<size_pops; i++){
        int id1 = id_pops[i];
        for (int j=i; j<size_pops; j++){
            int id2 = id_pops[j];
            obj->spk_sync[id1][id2] += cij[size_pops*i+j];
            cum[id1*num_pop_types+id2] += 1;
        }
    }

    for (int i=0; i<num_pop_types; i++){
        for (int j=i; j<num_pop_types; j++){
            obj->spk_sync[i][j] /= cum[i*num_pop_types+j];
        }
    }

    // free(t_vec);
    free(sum_vec);
    free(cij);
    free(cum);
    for (int n=0; n<size_pops; n++){
        free(t_vec[n]);
    }
    free(t_vec);
}


void calculate_cv_isi(reading_t *obj){
    double *cv = (double*) calloc(size_pops, sizeof(double));
    for (int n=0; n<size_pops; n++){

        int nstep = -1, stack=0;
        while (nstep > flushed_steps){
            if (stack == num_spk[n]){
                break;
            }
            nstep = step_spk[n][stack];
            stack++;
        }
        int num = num_spk[n] - stack;

        if (num < 3){
            cv[n] = -1;
            continue;
        }

        int n_prev = nstep;
        double dn1=0, dn2=0;
        for (int i=stack; i<num_spk[n]; i++){
            nstep = step_spk[n][i];

            double dn = nstep - n_prev;
            dn1 += dn/(double)num;
            dn2 += dn*dn/(double)num;
            n_prev = nstep;
        }

        // double mu = (double) dn1 / (double) num;
        double mu = dn1;
        double s = sqrt(dn2 - dn1*dn1);
        if (isnan(s)){
            printf("nan detected: mu: %5.2f, s: %5.2f\n", mu, s);
        }

        cv[n] = s/mu;
    }

    // average
    int *nums = (int*) calloc(num_pop_types, sizeof(int));
    for (int n=0; n<size_pops; n++){
        if (cv[n] == -1) continue;

        int id = id_pops[n];
        obj->cv_isi[id] += cv[n];
        nums[id]++;
    }

    for (int id=0; id<num_pop_types; id++){
        if (nums[id] == 0){
            obj->cv_isi[id] = -1;
        } else {
            obj->cv_isi[id] = obj->cv_isi[id]/nums[id];
        }
    }

    free(nums);
    free(cv);
}

#define DEBUG

#ifdef DEBUG
#define PRINT_DEBUG(x) fprintf(stderr, x);
#else
#define PRINT_DEBUG(x) fprintf(stderr, "");
#endif


reading_t flush_measure(){
    reading_t obj_r = init_reading();
    calculate_fluct(&obj_r);
    calculate_firing_rate(&obj_r);
    calculate_cv_isi(&obj_r);
    calculate_spike_sync(&obj_r);
    // reset values
    reset_fluct();
    reset_firing_rate();
    flushed_steps += cum_steps;
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

        // if ((n == 0) || (n==size_pops-1)){
        // printf("cum_steps: %d, v_fluct_m1: %.2f, v_fluct_m2: %.2f, var: %.2f\n", cum_steps, v_fluct_m1[n], v_fluct_m2[n], var);
        // printf("vm: %.2f, v2: %.2f\n", vm, v_fluct_m2[n]/(double)cum_steps);
        // }
    }
    // printf("fluct1\n");

    for (int id=0; id<num_pop_types; id++){
        double vm = v_m1[id]/(double) cum_steps;
        double var = v_m2[id]/(double) cum_steps - vm*vm;
        // printf("var_tot: %f, var_indiv: %f, num_pops: %d\n", var, var_indiv[id], num_pop_types);
        obj_r->chi[id] = sqrt(var * num_pops[id] / var_indiv[id]);
    }
    free(var_indiv);
}


void reset_fluct(){
    for (int n=0; n<size_pops; n++){
        v_fluct_m1[n] = 0;
        v_fluct_m2[n] = 0;
    }

    for (int id=0; id<num_pop_types; id++){
        v_m1[id] = 0;
        v_m2[id] = 0;
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

    double div = cum_steps * _dt / 1000.;
    // double div = cum_steps / 1000.;
    for (int n=0; n<size_pops; n++){
        int id = id_pops[n];
        double x = cum_spk[n]/div;
        fr1[id] += x/num_pops[id];
        fr2[id] += x*x/num_pops[id];
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
    obj_r.cv_isi = (double*) calloc(num_pop_types, sizeof(double));
    obj_r.spk_sync = (double**) malloc(sizeof(double*) * num_pop_types);
    for (int n=0; n<num_pop_types; n++){
        obj_r.spk_sync[n] = (double*) calloc(num_pop_types, sizeof(double));
    }

    return obj_r;
}


void free_reading(reading_t *obj_r){
    free(obj_r->chi);
    free(obj_r->frs_m);
    free(obj_r->frs_s);
    free(obj_r->cv_isi);
    for (int n=0; n<num_pop_types; n++){
        free(obj_r->spk_sync[n]);
    }
    free(obj_r->spk_sync);
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
    len_vlfp = total_step;
}


static void free_lfp(void){
    for (int n=0; n<num_pop_types; n++){
        free(v_lfp[n]);
    }
    free(v_lfp);
}

*/
// /* Save part */
// void export_spike(const char *tag){

//     char fname[200];
//     sprintf(fname, "%s_info.txt", tag);
//     FILE *fp_spk_info = fopen(fname, "w");
//     sprintf(fname, "%s.dat", tag);
//     FILE *fp_spk_time = fopen(fname, "wb");

//     int num_tot_spk = 0;
//     for (int n=0; n<size_pops; n++){
//         fprintf(fp_spk_info, "%d,", num_spk[n]);
//         num_tot_spk += num_spk[n];
//     }
//     fclose(fp_spk_info);

//     int *tot_spk = (int*) malloc(sizeof(int) * num_tot_spk);
//     int cum = 0;
//     for (int n=0; n<size_pops; n++){
//         for (int i=0; i<num_spk[n]; i++){
//             tot_spk[i+cum] = step_spk[n][i];
//         }
//         cum += num_spk[n];
//     }

//     fwrite(tot_spk, sizeof(int), num_tot_spk, fp_spk_time);
//     fclose(fp_spk_time);
// }

