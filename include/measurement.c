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

const float t_spike_bin = 5.;


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
    free(id_class);
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
            obj->spk_sync[id1][id2] += (float) cij[ntk_size*i+j]/(num_class[id1]*num_class[id2]/2);
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
        obj->frs_m[id+1] = fr1[id];
        obj->frs_s[id+1] = sqrt(fr2[id] - fr1[id]*fr1[id]);
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
    FILE *fp = open_file(fname, "wb");

    float info[2] = {(float) num_class_types, (float) max_step};
    fwrite(info, sizeof(float), 2, fp);
    
    float *vlfp_tot = (float*) calloc(max_step, sizeof(float));
    for (int id=0; id<num_class_types; id++){
        float factor = (float) num_class[id] / ntk_size;
        for (int n=0; n<max_step; n++){
            vlfp_tot[n] += vlfp[id][n] * factor;
        }
    }
    
    write_signal_f(max_step, vlfp_tot, fp);
    for (int id=0; id<num_class_types; id++){
        write_signal_f(max_step, vlfp[id], fp);
    }

    free(vlfp_tot);
    fclose(fp);
}


void export_spike(const char *fname){
    FILE *fp = open_file(fname, "wb");
    int info[2] = {ntk_size, max_step};
    fwrite(info, sizeof(int), 2, fp);
    fwrite(num_spk, sizeof(int), ntk_size, fp);
    for (int n=0; n<ntk_size; n++){
        fwrite(step_spk[n], sizeof(int), num_spk[n], fp);
    }
}


void export_result(summary_t *obj, const char *fname){
    FILE *fp = open_file(fname, "w");
    fprintf(fp, "num_types:%d\n", num_class_types);
    
    fprintf(fp, "chi:");
    for (int n=0; n<num_class_types+1; n++){
        fprintf(fp, "%f,", obj->chi[n]);
    }; fprintf(fp, "\n");

    fprintf(fp, "cv:");
    for (int n=0; n<num_class_types+1; n++){
        fprintf(fp, "%f,", obj->cv_isi[n]);
    }; fprintf(fp, "\n");

    fprintf(fp, "frs_m:");
    for (int n=0; n<num_class_types+1; n++){
        fprintf(fp, "%f,", obj->frs_m[n]);
    }; fprintf(fp, "\n");

    fprintf(fp, "frs_s:");
    for (int n=0; n<num_class_types+1; n++){
        fprintf(fp, "%f,", obj->frs_s[n]);
    }; fprintf(fp, "\n");

    fprintf(fp, "spike_syn:\n");
    for (int i=0; i<num_class_types+1; i++){
        for (int j=0; j<num_class_types+1; j++){
            fprintf(fp, "%f,", obj->spk_sync[i][j]);
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
}


void test_print(summary_t *obj){
    printf("chi: ");
    for (int n=0; n<num_class_types+1; n++){
        printf("%5.3f,", obj->chi[n]);
    }; printf("\n");

    printf("cv: ");
    for (int n=0; n<num_class_types+1; n++){
        printf("%5.3f,", obj->cv_isi[n]);
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
