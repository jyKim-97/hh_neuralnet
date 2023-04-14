/*
Source code for measurement
*/
#include "measurement2.h"


// population parameters
static int ntk_size = -1; // total size of the network
static int num_class_types = 0; // total number of different types to measure
static int *id_class = NULL; // 
static int num_class[MAX_CLASS_M]; // The maximal number of types
extern double _dt; // ms

// simulation step
int max_step  = -1;
static int cum_steps[MAX_CHECK_M];
static int check_steps[MAX_CHECK_M];
static int num_check = 0;

// fluctuation recording
// - save for global average
static double v_tot1[MAX_CHECK_M];
static double v_tot2[MAX_CHECK_M];
// - save for local populations
static double *v1[MAX_CHECK_M]; 
static double *v2[MAX_CHECK_M];
static double v_avg1[MAX_CHECK_M][MAX_CLASS_M];
static double v_avg2[MAX_CHECK_M][MAX_CLASS_M];
// - lfp recording
static float *vlfp[MAX_CLASS_M];

// spike recording
// - measure firing rate
static int *cum_spk[MAX_CHECK_M];
// recording for save
static int *num_spk   = NULL; // (N, )
static int **step_spk = NULL; // (N, ?)

static void set_class(int _num_class_types, int *type_range);
static void init_flct();
static void push_flct();
static void free_flct();
static void add_checkpoint_flct();

static void init_spike();
static void push_spike();
static void free_spike();
static void add_checkpoint_spike();

static void calculate_cv_isi(summary_t *obj);
static void calculate_flct(summary_t *obj);
static void calculate_firing_rate(summary_t *obj);

// static void calculate_cv_isi(summary_t *obj);
static float average(float *x);




void init_measure(int N, int num_steps, int _num_class_types, int *_type_range){
    ntk_size = N;
    max_step = num_steps;
    set_class(_num_class_types, _type_range);
    
    init_spike();
    init_flct();
}


void destroy_measure(void){

    for (int n=0; n<num_check; n++){
        free(v1[n]);
        free(v2[n]);
        free(cum_spk[n]);
    }
    num_check = 0;

    for (int n=0; n<MAX_CHECK_M; n++){
        cum_steps[n] = 0;
        check_steps[n] = 0;
    }

    free_spike();
    free_flct();
    free(id_class);
}


static void set_class(int _num_class_types, int *type_range){
    id_class = (int*) malloc(sizeof(int) * ntk_size);
    
    if (type_range == NULL){
        num_class_types = 2;
        for (int n=0; n<ntk_size; n++) id_class[n] = (n < ntk_size*0.8)? 0: 1;
    } else {
        num_class_types = _num_class_types;
        int tp = 0;
        for (int n=0; n<ntk_size; n++){
            if (n == type_range[tp]) tp ++;
            id_class[n] = tp;
        }
    }

    // get num_class
    for (int n=0; n<MAX_CLASS_M; n++) num_class[n] = 0;

    for (int n=0; n<ntk_size; n++){
        if ((id_class[n] < 0) || (id_class[n] >= MAX_CLASS_M-1)){
            printf("Neuron ID is wrong: %d, resize the LEN or check class\n", id_class[n]);
            exit(1);
        }

        if (id_class[n] >= _num_class_types){
            printf("The class id (%d) exceeds with expected (%d) (measurement2.c: set_class)\n", id_class[n], num_class_types);
        }

        int id = id_class[n];
        num_class[id]++;
    }
}


void print_num_check(){
    printf("num_check: %d\n", num_check);
}


static void init_spike(){
    num_spk  = (int*) calloc(ntk_size, sizeof(int));
    step_spk = (int**) malloc(ntk_size * sizeof(int*));
    for (int n=0; n<ntk_size; n++){
        step_spk[n] = (int*) calloc(_block_size, sizeof(int));
    }
}


static void add_checkpoint_spike(){
    cum_spk[num_check] = (int*) calloc(ntk_size, sizeof(int));
}



static void free_spike(){
    free(num_spk);
    for (int n=0; n<num_check; n++) free(cum_spk[n]);
    for (int n=0; n<ntk_size; n++) free(step_spk[n]);
    free(step_spk);
}


static void init_flct(){

    for (int n=0; n<MAX_CHECK_M; n++){
        v1[n] = NULL;
        v2[n] = NULL;

        v_tot1[n] = 0;
        v_tot2[n] = 0;

        for (int id=0; id<MAX_CLASS_M; id++){
            v_avg1[n][id] = 0;
            v_avg2[n][id] = 0;
        }
    }

    for (int id=0; id<num_class_types; id++){
        vlfp[id] = (float*) calloc(max_step, sizeof(float));
    }
}


static void add_checkpoint_flct(){

    v1[num_check] = (double*) calloc(ntk_size, sizeof(double));
    v2[num_check] = (double*) calloc(ntk_size, sizeof(double));

    v_tot1[num_check] = 0;
    v_tot2[num_check] = 0;
    for (int n=0; n<MAX_CLASS_M; n++){
        v_avg1[num_check][n] = 0;
        v_avg2[num_check][n] = 0;
    }
}


static void free_flct(){

    for (int n=0; n<num_check-1; n++){
        free(v1[n]);
        free(v2[n]);
    }
    for (int n=0; n<num_class_types; n++) free(vlfp[n]);
}


void add_checkpoint(int nstep){

    if (num_check == MAX_CHECK_M){
        printf("The number of checkpoint exceeds maximum number. Cannot add checkpoint.\n");
        return ;
    }

    add_checkpoint_flct();
    add_checkpoint_spike();
    check_steps[num_check] = nstep;
    num_check++;
}


void measure(int nstep, wbneuron_t *neuron){

    if (num_check == 0){
        printf("Warning: there is no checkpoint to measure\n");
    }
    
    double v_tmp=0;
    for (int n=0; n<ntk_size; n++){
        double v = neuron->vs[n];
        long double v_pow = v*v;
        int id = id_class[n];

        v_tmp += v;
        vlfp[id][nstep] += v;
        for (int i=0; i<num_check; i++){
            v1[i][n] += v;
            v2[i][n] += v_pow;
        }

        // check spike
        if (neuron->is_spk[n]){
            append_int(step_spk+n, num_spk[n], nstep);
            num_spk[n]++;

            for (int i=0; i<num_check; i++){
                cum_spk[i][n]++;
            }
        }
    }
    v_tmp /= ntk_size;

    for (int i=0; i<num_check; i++){
        v_tot1[i] += v_tmp;
        v_tot2[i] += v_tmp*v_tmp;
    }

    for (int id=0; id<num_class_types; id++){
        vlfp[id][nstep] /= num_class[id];
        for (int i=0; i<num_check; i++){
            v_avg1[i][id] += vlfp[id][nstep];
            v_avg2[i][id] += vlfp[id][nstep]*vlfp[id][nstep];
        }
    }

    for (int i=0; i<num_check; i++){
        cum_steps[i]++;
    }
}


summary_t flush_measure(void){

    summary_t obj = {0,};

    if (num_check == 0){
        printf("There is no checkpoint to flush\n");
        return obj;
    }

    calculate_cv_isi(&obj);
    calculate_firing_rate(&obj);
    calculate_flct(&obj);
    
    push_flct();
    push_spike();
    for (int n=1; n<num_check; n++){
        cum_steps[n-1] = cum_steps[n];
        check_steps[n-1] = check_steps[n];
    }
    cum_steps[num_check-1] = 0;
    check_steps[num_check-1] = 0;

    num_check--;
    // printf("num_check: %d\n", num_check);

    return obj;
}


static void push_flct(){
    free(v1[0]);
    free(v2[0]);
    int sz = sizeof(double) * ntk_size;

    if (num_check > 1){
        v1[0] = (double*) malloc(sz);
        v2[0] = (double*) malloc(sz);
    }

    for (int n=1; n<num_check; n++){
        memcpy(v1[n-1], v1[n], sz);
        memcpy(v2[n-1], v2[n], sz);

        for (int id=0; id<MAX_CLASS_M; id++){
            v_avg1[n-1][id] = v_avg1[n][id];
            v_avg2[n-1][id] = v_avg2[n][id];
        }

    }

    if (num_check > 1){
        free(v1[num_check-1]);
        free(v2[num_check-1]);
    }
}


static void push_spike(){
    free(cum_spk[0]);
    int sz = sizeof(int) * ntk_size;

    if (num_check > 1){
        cum_spk[0] = (int*) malloc(sz);
    }
    
    for (int n=1; n<num_check; n++){
        memcpy(cum_spk[n-1], cum_spk[n], sz);
    }

    if (num_check > 1){
        free(cum_spk[num_check-1]);
    }
}


static void calculate_cv_isi(summary_t *obj){
    float *cv = (float*) calloc(ntk_size, sizeof(float));
    int num_init = check_steps[0];

    for (int n=0; n<ntk_size; n++){
        // ignore bef prev_step
        int nstep = -1, stack=0;
        while (nstep > num_init){
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

        float mu = dn1;
        float s = sqrt(dn2 - dn1*dn1);
        if (isnan(s)){
            printf("nan detected: mu: %5.2f, s: %5.2f\n", mu, s);
        }

        cv[n] = s/mu;
    }

    // total summary
    obj->cv_isi[0] = average(cv);

    float cv_isi_tmp[MAX_CLASS_M] = {0,};
    int stack[MAX_CLASS_M] = {0,};
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


static void calculate_flct(summary_t *obj){
    double var_indiv[MAX_CLASS_M] = {0,};

    for (int n=0; n<ntk_size; n++){
        double vm  = v1[0][n]/cum_steps[0];
        double var = v2[0][n]/cum_steps[0] - vm*vm;
        int id = id_class[n];
        var_indiv[id] += var;
    }

    double var_indiv_tot=0;
    for (int id=0; id<num_class_types; id++){
        var_indiv_tot += var_indiv[id];
    }

    v_tot2[0] /= cum_steps[0];
    v_tot1[0] /= cum_steps[0];
    double var_tot = v_tot2[0] - v_tot1[0] * v_tot1[0];
    obj->chi[0] = sqrt(var_tot * ntk_size/var_indiv_tot);

    // printf("cum steps: %d, var_tot: %f, var_indiv_tot: %f\n", cum_steps[0], var_tot, var_indiv_tot/ntk_size);

    // each class
    for (int id=0; id<num_class_types; id++){
        double vm  = v_avg1[0][id]/cum_steps[0];
        double var = v_avg2[0][id]/cum_steps[0] - vm*vm;
        obj->chi[id+1] = sqrt(var*num_class[id]/var_indiv[id]);
    }
}


static void calculate_firing_rate(summary_t *obj){
    float fr1[MAX_CLASS_M] = {0,};
    float fr2[MAX_CLASS_M] = {0,};

    float m=0, var=0; // for total summary
    float div = cum_steps[0] * _dt / 1000.;
    for (int n=0; n<ntk_size; n++){
        int id = id_class[n];
        float x = cum_spk[0][n]/div;

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


static float average(float *x){
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

    extern double _fs_save;
    float info[2] = {(float) num_class_types, (float) _fs_save};
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
    fclose(fp);
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

    // fprintf(fp, "spike_syn:\n");
    // for (int i=0; i<num_class_types+1; i++){
    //     for (int j=0; j<num_class_types+1; j++){
    //         fprintf(fp, "%f,", obj->spk_sync[i][j]);
    //     }
    //     fprintf(fp, "\n");
    // }
    
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
}

/* Deprecated functions

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
*/
