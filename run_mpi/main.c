/** Source code for mpirun **/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include "model.h"
#include "build.h"
#include "rng.h"
#include "utils.h"
#include "storage.h"
#include "mt64.h"
#include "measurement.h"

#define NCTRL 4

// #define DEBUG_MOD

extern neuron_t neuron;
extern syn_t syn[MAX_TYPE];
extern syn_t ext_syn;


int N = 2000;
double tmax = 12000;

double t_eq = 1000; // measure after 1s
// double t_flush = 10000;
double t_flush = 2000;
int n_eq=-1, n_flush=-1;
int world_rank, world_size;
double delay=0;
double iapp=0;
double *lambda_ext=NULL;
double w_ext=0;
char fdir[] = "./data";

void run(int run_id, buildInfo *info);
buildInfo set_default(void);
void update_pop(int nstep);
void write_reading(const char *fname, reading_t obj_r);
double *linspace(double x0, double x1, int len_x);
void end_pop();
double get_syn_current(int nid, double v);
void debug_file_open(int run_id, int flush_id);
void print_arr(FILE *fp, char *arr_name, int arr_size, double *arr);


int main(int argc, char **argv){
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    n_eq = t_eq / _dt;
    n_flush = t_flush / _dt;

    // set parameters
    int nitr = 1;
    int max_len[NCTRL] = {8, 5, 3, nitr};

    /*** #################################### ***/
    index_t idxer;
    set_index_obj(&idxer, NCTRL, max_len);

    /*** Set control parameter & write ***/

    double *mdeg_out_inh = linspace(300, 1200, max_len[0]);
    double *g_inh = linspace(0.01, 0.1, max_len[1]);
    double *nu_ext = linspace(2000, 8000, max_len[2]);

    char fname[200];
    sprintf(fname, "%s/control_params.txt", fdir);
    FILE *fp = fopen(fname, "w");
    print_arr(fp, "mdeg_out_inh", max_len[0], mdeg_out_inh);
    print_arr(fp, "g_inh", max_len[1], g_inh);
    print_arr(fp, "nu_ext", max_len[2], nu_ext);
    fclose(fp);
    
    for (int n=world_rank; n<idxer.len; n+=world_size){
        update_index(&idxer, n);

        /* Set parameter */
        buildInfo info = set_default();
        info.mdeg_out[1][0] = mdeg_out_inh[idxer.id[0]]/5.*4;
        info.mdeg_out[1][1] = mdeg_out_inh[idxer.id[0]]/5.;
        info.w[1][0] = g_inh[idxer.id[1]];
        info.w[1][1] = g_inh[idxer.id[1]];
        info.nu_ext  = nu_ext[idxer.id[2]];

        // run
        run(n, &info);
    }

    MPI_Finalize();
}


void print_arr(FILE *fp, char *arr_name, int arr_size, double *arr){
    fprintf(fp, "%s:", arr_name);
    for (int n=0; n<arr_size; n++){
        fprintf(fp, "%f,", arr[n]);
    }
    fprintf(fp, "\n");
}


FILE *fv=NULL;
void run(int run_id, buildInfo *info){

    // report information
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
    time_t t_tmp = time(NULL);
    struct tm t = *localtime(&t_tmp);
    printf("[%d-%02d-%02d %02d:%02d:%02d] ", t.tm_year+1900, t.tm_mon+1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
    printf("Node%3d Start job%5d in pid %d\n", world_rank, run_id, getpid()); 

    int nmax = tmax / _dt;
    int *types = (int*) calloc(N, sizeof(int));
    // init_simulation(); -> 수정
    for (int n=(int) N*0.8; n<N; n++) types[n] = 1;
    init_measure(N, nmax, 2, types);

    #ifdef DEBUG_MOD
    debug_file_open(run_id, 0);
    #endif

    /* Build information */
    build_eipop(info);
    lambda_ext = (double*) malloc(sizeof(double) * N);
    for (int n=0; n<N; n++){
        lambda_ext[n] = _dt/1000.*info->nu_ext;
    }
    w_ext = info->w_ext;
    init_deSyn(N, 0, _dt/2., &ext_syn);

    int stack = 0;
    int flush_id = 0;

    for (int n=0; n<nmax; n++){
        update_pop(n);
        measure(n, &neuron);

        #ifdef DEBUG_MOD
        save(N, n, neuron.v, fv);
        #endif

        /* Flush 파트 추가 */
        if (n == n_eq){
            reading_t obj_read = flush_measure();
            free_reading(&obj_read);
            stack = 0;
        } else if (stack == n_flush){
            reading_t obj_read = flush_measure();
            // save 
            char fname[100];
            sprintf(fname, "%s/id%06d_%02d_result.txt", fdir, run_id, flush_id);
            write_reading(fname, obj_read);
            free_reading(&obj_read);
            flush_id++;
            stack = 0;

            #ifdef DEBUG_MOD
            char fspk[100];
            sprintf(fspk, "./debug/id%06d_%02d", run_id, flush_id);
            export_spike(fspk);
            fclose(fv);
            debug_file_open(run_id, flush_id);
            printf("flush!\n");
            #endif
        }
        stack ++;
    }

    free(lambda_ext);
    end_pop();

    gettimeofday(&toc, NULL);
    double dt = get_dt(tic, toc);
    printf("#%03d-%05d job Done, elapsed=%4.2fm\n", world_rank, run_id, dt/60.);
}


void debug_file_open(int run_id, int flush_id){
    char fname[100];
    sprintf(fname, "./debug/id%06d_%02d_v.txt", run_id, flush_id);
    fv = fopen(fname, "wb");
}


buildInfo set_default(void){
    buildInfo info = {0,};

    info.N = N;
    info.buf_size = delay/_dt+5; // 1 ms
    info.ode_method = RK4;

    info.num_types[0] = info.N * 0.8;
    info.num_types[1] = info.N * 0.2;

    info.mdeg_out[0][0] = 200/5*4; //40/5*4;
    info.mdeg_out[0][1] = 200/5; //40/5;
    info.mdeg_out[1][0] = 600/5*4;
    info.mdeg_out[1][1] = 600/5; //400/5;
    
    info.w[0][0] = 0.01;
    info.w[0][1] = 0.01;
    info.w[1][0] = 0.1;
    info.w[1][1] = 0.1;

    info.n_lag[0][0] = delay/_dt;
    info.n_lag[0][1] = delay/_dt;
    info.n_lag[1][0] = delay/_dt;
    info.n_lag[1][1] = delay/_dt;

    info.nu_ext = 2000;
    info.w_ext  = 0.0003;

    return info;
}


void write_reading(const char *fname, reading_t obj_r){
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "chi,frs_m,frs_s,cv_isi\n");
    for (int n=0; n<2; n++){
        fprintf(fp, "%f,%f,%f,%f,\n", obj_r.chi[n], obj_r.frs_m[n], obj_r.frs_s[n], obj_r.cv_isi[n]);
    }

    fprintf(fp, "cij\n");
    for (int i=0; i<2; i++){
        for (int j=0; j<2; j++){
            fprintf(fp, "%f,", obj_r.spk_sync[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}


double *linspace(double x0, double x1, int len_x){
    double *x = (double*) malloc(sizeof(double) * len_x);
    if (len_x == 1){
        // printf("Too few length selected. x is set to %.2f\n", x0);
        x[0] = x0;
        return x;
    }
    for (int n=0; n<len_x; n++){
        x[n] = n*(x1-x0)/(len_x-1)+x0;
    }
    return x;
}


void update_pop(int nstep){

    double *v_prev = copy_array(N, neuron.v);

    // update 
    double *ptr_v = neuron.v;
    double *ptr_h = neuron.h_ion;
    double *ptr_n = neuron.n_ion;

    // get poisson input
    int *num_ext = get_poisson_array(N, lambda_ext);
    // printf("lambda ext: %f\n", lambda_ext[0]);

    // 이부분 parallelize 가능할듯?
    // #pragma omp parallel
    for (int n=0; n<N; n++){

        // if (*ptr_v == nan){
        if (isnan(*ptr_v)){
            fprintf(stderr, "\nERROR: nan detected in Neuron %d in step %d\n", n, nstep);
            exit(1);
        }
        ext_syn.expr[n] += w_ext * ext_syn.A * num_ext[n];
        ext_syn.expd[n] += w_ext * ext_syn.A * num_ext[n];
        // printf("num: %d, r: %f\n", num_ext[n], ext_syn.expr[n]);

        add_spike_syn(&(syn[0]), n, nstep, &(neuron.buf));
        add_spike_syn(&(syn[1]), n, nstep, &(neuron.buf));

        // RK4 method
        // 1st step
        // double isyn = get_current_deSyn(syn, n, *ptr_v);
        double isyn = get_syn_current(n, *ptr_v);
        // double isyn = 0;
        double dv1 = solve_wb_v(*ptr_v, iapp-isyn, *ptr_h, *ptr_n);
        double dh1 = solve_wb_h(*ptr_h, *ptr_v);
        double dn1 = solve_wb_n(*ptr_n, *ptr_v);
        
        // 2nd step
        update_deSyn(syn, n);
        update_deSyn(syn+1, n);
        update_deSyn(&ext_syn, n);
        isyn = get_syn_current(n, *ptr_v);

        double dv2 = solve_wb_v(*ptr_v+dv1*0.5, iapp-isyn, *ptr_h+dh1*0.5, *ptr_n+dn1*0.5);
        double dh2 = solve_wb_h(*ptr_h+dh1*0.5, *ptr_v+dv1*0.5);
        double dn2 = solve_wb_n(*ptr_n+dn1*0.5, *ptr_v+dv1*0.5);

        // 3rd step
        isyn = get_syn_current(n, *ptr_v);
        // isyn = 0;
        double dv3 = solve_wb_v(*ptr_v+dv2*0.5, iapp-isyn, *ptr_h+dh2*0.5, *ptr_n+dn2*0.5);
        double dh3 = solve_wb_h(*ptr_h+dh2*0.5, *ptr_v+dv2*0.5);
        double dn3 = solve_wb_n(*ptr_n+dn2*0.5, *ptr_v+dv2*0.5);

        // 4th step
        update_deSyn(syn, n);
        update_deSyn(syn+1, n);
        update_deSyn(&ext_syn, n);
        isyn = get_syn_current(n, *ptr_v);
        // isyn = 0;
        double dv4 = solve_wb_v(*ptr_v+dv3, iapp-isyn, *ptr_h+dh3, *ptr_n+dn3);
        double dh4 = solve_wb_h(*ptr_h+dh3, *ptr_v+dv3);
        double dn4 = solve_wb_n(*ptr_n+dn3, *ptr_v+dv3);
        
        *ptr_v += (dv1 + 2*dv2 + 2*dv3 + dv4)/6.;
        *ptr_h += (dh1 + 2*dh2 + 2*dh3 + dh4)/6.;
        *ptr_n += (dn1 + 2*dn2 + 2*dn3 + dn4)/6.;

        ptr_v++; ptr_h++; ptr_n++;
    }

    update_spkBuf(nstep, &(neuron.buf), v_prev, neuron.v);
    free(v_prev);
    free(num_ext);
}


double get_syn_current(int nid, double v){
    double isyn = get_current_deSyn(syn, nid, v);
    isyn += get_current_deSyn(syn+1, nid, v);
    isyn += get_current_deSyn(&ext_syn, nid, v);
    return isyn;
}


void end_pop(){
    destroy_wbNeuron(&neuron);
    destroy_deSyn(syn);
    destroy_deSyn(syn+1);
    destroy_deSyn(&ext_syn);
    free_measure();
}
