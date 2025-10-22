#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "utils.h"
#include "storage.h"
#include "model2.h"
#include "neuralnet.h"
#include "measurement2.h"
#include "unistd.h"

// #define TEST

#ifndef TEST
#include "mpifor.h"
#endif


extern double _dt;
// extern wbneuron_t neuron;
extern nnpop_t nnpop;
extern int world_size, world_rank;

static void print_syn_ntk(const char *prefix);
static void print_syn_ntk2(const char *prefix);

#define REPORT_ERROR(msg) print_error(msg, __FILE__, __LINE__)

int N = 2000;
double tmax = 10500;
double teq = 500;

nn_info_t *nn_info_set = NULL;

// Parameters for transmitters
int num_trans = 10; // for each population (excitatory neuron)
// int w_ratio = 100; // input strength ratio
double w_ratio = 0;
int nd_trans = 0;

// global var, can be controllled with input args
int nsamples = 0;
char fname_params[200] = "./";
char fdir_out[100] = "./tmp";

void run(int job_id, void *nullarg);
void read_args(int argc, char **argv);
void read_params(char *fname);
int read_params_line(FILE *fp, nn_info_t *info);
void allocate_multiple_ext(nn_info_t *info);
static void reorganize_syn_network(const char *prefix);

// double taur[] = {0.3, 0.5, 0.3, 1};
// double taud[] = {  1, 2.5,   1, 8};

// mpirun -np 100 --hostfile hostfile ./main.out -t 10500 --fdir_out ./data
// #define PTYPE "normal"
double taur[] = {0.3, 0.5, 0.3, 1};
double taud[] = {  1, 2.5,   1, 8};


/* Note that the time constant for the synapse has been changed */
int main(int argc, char **argv){

    /* read args & init parameters */
    #ifndef TEST
    init_mpi(&argc, &argv);
    #endif
    // set_seed(time(NULL) * world_size * 5 + world_rank*2);
    ignore_exist_file(true);
    read_args(argc, argv);
    read_params(fname_params);

    #ifdef PTYPE
    if (strcmp(PTYPE, "_normal") == 0) {
        if (world_rank == 0){
            printf("Test with normal population\n");
        }
    } else if (strcmp(PTYPE, "_mfast") == 0) {
        if (world_rank == 0) printf("Run with fast-fast population\n");
        for (int i=0; i<2; i++){
            taur[i+2] = taur[i];
            taud[i+2] = taud[i];
        }
    } else if (strcmp(PTYPE, "_mslow") == 0) {
        if (world_rank == 0) printf("Run with slow-slow population\n");
        for (int i=0; i<2; i++){
            taur[i] = taur[i+2];
            taud[i] = taud[i+2];
        }
    } else {
        if (world_rank == 0) printf("Wrong population: %s\n", PTYPE);
        
        return 0;
    }
    #endif

    #ifndef TEST
    mpi_barrier();
    #endif
    if (world_rank == 0){
        printf("%d samples are observed\n", nsamples);
    }

    /* Temporarily added */
    ignore_exist_file(true);

    /* run simulation */
    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
    #ifndef TEST
    for_mpi(nsamples, run, NULL);
    #else
    // teq = 0;
    tmax = 3000;
    _dt = 0.1;
    // run(123, NULL);
    for (int n=23; n<500; n++){
        fprintf(stderr, "========\nsample %d check\n\n=============", n);
        run(n, NULL);
    }

    // run(3019, NULL);
    // for (int n=0; n<nsamples; n++) run(n, NULL);
    #endif

    if (world_rank == 0){ 
        gettimeofday(&toc, NULL);
        printf("Simulation done, total elapsed: %.3f hour\n", get_dt(tic, toc)/3600.);
    }

    /* close simulation */
    free(nn_info_set);
    #ifndef TEST
    end_mpi();
    #endif

    return 0;
}


void run(int job_id, void *nullarg){

    nn_info_t info = nn_info_set[job_id];
    w_ratio  = info.wratio_trans;
    nd_trans = info.tdelay_trans / _dt;
    // nd_trans = info.tdelay_trans / _dt;
    // printf("nd_trans: %d\n", nd_trans);

    // check is file exist
    char fname_exist[200];
    sprintf(fname_exist, "%s/id%06d_lfp.dat", fdir_out, job_id);
    // int res = access(fname_exist, F_OK);

    if (access(fname_exist, F_OK) == 0){
        printf("job %d already done, skip this id\n", job_id);
        return;
    }

    char fname_info[100], fbuf[200];
    sprintf(fname_info, "id%06d_info.txt", job_id);
    path_join(fbuf, fdir_out, fname_info);
    write_info(&info, fbuf);

    // set_seed(info.seed);
    set_seed_by_id(info.seed, 0);
    // printf("seed: %d\nstate: %d\n", info.seed, );
    build_ei_rk4(&info);
    allocate_multiple_ext(&info);

    char fname_tr[100];
    sprintf(fname_tr, "id%06d_trinfo.txt", job_id);
    path_join(fbuf, fdir_out, fname_tr);
    reorganize_syn_network(fbuf);

    int nmax = tmax/_dt;
    int neq  = teq / _dt;    

    int pop_range[2] = {N/2, N};
    init_measure(N, nmax, 2, pop_range);
    add_checkpoint(0);

    char fname_id[100], prefix[200];
    sprintf(fname_id, "id%06d", job_id);
    path_join(prefix, fdir_out, fname_id);
    print_syn_ntk2(prefix);

#ifdef TEST
    destroy_neuralnet();
    destroy_measure();
    destroy_rng();
    return;
#endif

    // check detailed synaptic network
    #ifdef TEST
    print_syn_ntk(prefix);
    progbar_t bar;
    init_progressbar(&bar, nmax);
    #endif

    int flag_eq = 0;
    // int n0 = 0;
    // int nw = 1000 / _dt; // 1 s
    // int nm = 800 / _dt;
    // int nstack = 0;
    for (int nstep=0; nstep<nmax; nstep++){
        if ((flag_eq==0) && (nstep == neq)){
            flush_measure();
            flag_eq = 1;

            add_checkpoint(nstep);
            // n0 = neq;
        }

        // if ((flag_eq==1) && (nstep==n0+nm)){
        //     // extern int num_check;
        //     add_checkpoint(nstep);
        //     // printf("Check added (%d): %.1f\n", num_check, nstep * _dt);
        // }

        update_rk4(nstep, 0);
        KEEP_SIMUL();
        measure(nstep, &nnpop);

        #ifdef TEST
        progressbar(&bar, nstep);
        #endif
    }

    /* Extract values */
    summary_t obj = flush_measure();
    char fname_res[100];
    sprintf(fname_res, "id%06d_result.txt", job_id);
    path_join(fbuf, fdir_out, fname_res);
    export_result(&obj, fbuf);

    char fname_spk[100];
    sprintf(fname_spk, "id%06d_spk.dat", job_id);
    path_join(fbuf, fdir_out, fname_spk);
    export_spike(fbuf);

    char fname_lfp[100];
    sprintf(fname_lfp, "id%06d_lfp.dat", job_id);
    path_join(fbuf, fdir_out, fname_lfp);
    export_lfp(fbuf);

    destroy_neuralnet();
    destroy_measure();
    destroy_rng();

    // print_job_end(job_id, nsamples);
}


void read_args(int argc, char **argv){
    // get blank typed_args_t & set default values
    // NOTE: hard fix the args

    // strcpy(fname_params, "./data/params_to_run.txt");
    // strcpy(fdir_out, "./data/");

    int n = 1;
    while (n < argc){
        // if (strcmp(argv[n], "-n") == 0){
        //     nsamples = atoi(argv[n+1]); n++;
        // } else if (strcmp(argv[n], "-t") == 0){
        if (strcmp(argv[n], "-t") == 0){
            tmax = atof(argv[n+1]); n++;
        } else if (strcmp(argv[n], "--fdir") == 0){
            strcpy(fdir_out, argv[n+1]); n++;
        } else {
            printf("Wrong argument typed: %s\n", argv[n]);
            exit(-1);
        }
        n++;
    }

    sprintf(fname_params, "%s/params_to_run.txt", fdir_out);

    if (world_rank == 0){
        printf("tmax: %f\nfparams: %s\nfdir_out: %s\n", tmax, fname_params, fdir_out);
    }
}


void read_params(char *fname){

    // nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * *nline);

    if (world_rank == 0){
        FILE *fp;
        if ((fp = fopen(fname, "r+")) == NULL){
            printf("file %s does not exist\n", fname);
            exit(-1);
        }
        
        if (fscanf(fp, "%d\n", &nsamples) < 0){
            printf("Falied to read the number of samples. Check the parameter info file\n");
            exit(-1);
        }

        nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nsamples);
        for (int n=0; n<nsamples; n++){
            if (read_params_line(fp, &(nn_info_set[n])) == 1){
                fprintf(stderr, "Failed to read parameter in line %d\n", n);
                exit(-1);
            }
        }

        fclose(fp);
        // printf("fast->slow: %f\n slow->fast: %f\n", nn_info_set[nsamples-1].p_out[0][2],
        //                                         nn_info_set[nsamples-1].p_out[2][0]);
    }

    #ifndef TEST
    MPI_Bcast(&nsamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0){
        nn_info_set = (nn_info_t*) malloc(sizeof(nn_info_t) * nsamples);
    }

    mpi_barrier();
    MPI_Bcast((void*) nn_info_set, nsamples*sizeof(nn_info_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    #endif
    // init_nn(nn_info_set[0].N, nn_info_set[0].num_types);
}


int read_params_line(FILE *fp, nn_info_t *info){

    /* === Parameter input ===
     0: seed
     0: cluster id 
     1- 4. w_ef, w_if, w_es, w_is
     5- 8. p_{ef->ef,if,es,is}
     9-12. p_{if->ef,if,es,is}
    13-16. p_{es->ef,if,es,is}
    17-21. p_{is->ef,if,es,is}
    23-24. nu_bck_f, nu_bck_s
    */
    
    *info = init_build_info();
    info->N = N;
    info->pN = 0;

    // set type info
    int ntype = 4;
    int type_id[] = {E, IF, E, IS};
    int type_range[] = {800, 200, 800, 200};
    set_type_info(info, ntype, type_id, type_range);

    // default parameters
    info->taur[0] = taur[0]; info->taud[0] = taud[0];   // -> E
    info->taur[1] = taur[1]; info->taud[1] = taud[1];   // -> I_F
    info->taur[2] = taur[2]; info->taud[2] = taud[2];   // -> E
    info->taur[3] = taur[3]; info->taud[3] = taud[3];   // -> I_S
    // info->taur[4] = taur[0]; info->taud[4] = taud[0];   // -> E
    // info->taur[5] = taur[0]; info->taud[5] = taud[0];   // -> E
    // info->taur[2] = 0.3; info->taud[2] = 1;   // -> E
    // info->taur[3] = 0.5;   info->taud[3] = 2.5;   // -> I_S
    
    info->num_ext_types = 2;
    // info->w_ext_multi[0] = 0.002;
    // info->w_ext_multi[1] = 0.002;
    info->w_ext_mu[0] = 0.002;
    info->w_ext_mu[1] = 0.002;
    info->t_lag = 0.;
    info->const_current = false;

    // set weight
    long seed;
    fscanf(fp, "%ld,", &seed);
    info->seed = seed;

    for (int i=0; i<ntype; i++){
        double w=0;
        if (fscanf(fp, "%lf,", &w) < 1){
            return 1;
        }
        for (int j=0; j<ntype; j++){
            info->w[i][j] = w;
        }
    }

    // set connection probability (out)
    for (int i=0; i<ntype; i++){
        for (int j=0; j<ntype; j++){
            fscanf(fp, "%lf,", &(info->p_out[i][j]));
        }
    }

    for (int i=0; i<ntype+2; i++){
        for (int j=0; j<ntype+2; j++){
            if ((i >= ntype) || (j >= ntype)){
                info->p_out[i][j] = 0;
                info->w[i][j] = 0;
            }
        }
    }

    // set nu_bck
    for (int i=0; i<2; i++){
        // fscanf(fp, "%lf,", &(info->nu_ext_multi[i]));
        fscanf(fp, "%lf,", &(info->nu_ext_mu[i]));
    }

    fscanf(fp, "%lf,", &(info->wratio_trans));
    fscanf(fp, "%lf,", &(info->tdelay_trans));

    // change line
    fscanf(fp, "\n");

    return 0;
}


void allocate_multiple_ext(nn_info_t *info){
    // depends on N
    int nsub = N/2;
    int *target = (int*) malloc(sizeof(int) * nsub);
    
    for (int ntype=0; ntype<2; ntype++){
        for (int i=0; i<nsub; i++){
            target[i] = i + nsub * ntype;
        }
        set_multiple_ext_input(info, ntype, nsub, target);
    }

    free(target);
    // check_multiple_input();
}


static void print_syn_ntk2(const char *prefix){
    // print only adjacency list

    char fname_syn[200];
    sprintf(fname_syn, "%s_adj.txt", prefix);
    FILE *fp = fopen(fname_syn, "w");

    for (int n=0; n<N; n++){
        fprintf(fp, "%d<-", n);
        int is_not_zero = 0;
        for (int ntp=0; ntp<4; ntp++){
            desyn_t syn = nnpop.syns[ntp];
            int num_pre = syn.num_indeg[n];
            for (int i=0; i<num_pre; i++){
                int npre = syn.indeg_list[n][i];
                if (npre == -1) continue;

                is_not_zero = 1;
                fprintf(fp, "%d,", npre);
            }
        }
        if (is_not_zero == 0){
            fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}


static void print_syn_ntk(const char *prefix){
    for (int n=0; n<4; n++){
        char fname_syn[200];
        sprintf(fname_syn, "%s_syn_%d.txt", prefix, n);
        print_syn_network(nnpop.syns+n, fname_syn);
    }
}


const int tdelay_max = 30;
const int tdelay_itv = 2;

static void reorganize_syn_network(const char *fname){

    int nhalf = nnpop.N/2;
    int* sel_tneuron = (int*) calloc(2*num_trans, sizeof(int));
    int* sel_rneuron = (int*) calloc(2*num_trans, sizeof(int)); // selected receving neuron
    
    extern int cell_range[MAX_TYPE][2];
    int Ne = cell_range[0][1] - cell_range[0][0];
    for (int ntp=0; ntp<2; ntp++){
        for (int n=0; n<num_trans; n++){
            sel_tneuron[ntp*num_trans+n] = ntp*nhalf+n;
        }
    }


    for (int ntp=0; ntp<2; ntp++){ // type of pre- pop
        for (int n=0; n<num_trans; n++){
            
            // list up the post- ID
            int *id_target = (int*) malloc(sizeof(int)*Ne);
            desyn_t syn = nnpop.syns[2*ntp];
            
            int n0 = ntp*nhalf+n;
            bool is_update = false;
            int nstack = 0;
            while (nstack < 5){ // select T neurons which has more than 5 post-syn neurons
                if (is_update){
                    int nmax = 0;
                    for (int i=ntp*num_trans; i<(ntp+1)*num_trans; i++){
                        if (nmax < sel_tneuron[i]){
                            nmax = sel_tneuron[i];
                        }
                    }
                    n0 = nmax+1;
                }
                for (int n=0; n<Ne; n++) id_target[n] = -1;

                nstack = 0;
                int ntp_r = 2*(1-ntp);
                for (int i=cell_range[ntp_r][0]; i<cell_range[ntp_r][1]; i++){
                    if (i < cell_range[ntp_r][0]+num_trans){
                        continue;
                    }
                    for (int j=0; j<syn.num_indeg[i]; j++){
                        if (syn.indeg_list[i][j] == n0){
                            id_target[nstack++] = i;
                            break;
                        }
                    }
                }
                id_target[nstack] = -1;
                
                for (int i=0; i<ntp*num_trans+n; i++){
                    if ((n0 == sel_tneuron[i]) || (n0 == sel_rneuron[i])){
                        nstack = 0;
                    }
                }

                sel_tneuron[ntp*num_trans+n] = n0;
                is_update = true;
            }

            // select target neuron (receiver neuron)
            int nitr = 0;
            int idsel;
            int flag_in = 1;
            while (flag_in == 1){
                double p = genrand64_real2();
                int idp = (int) (p*nstack);
                idsel = id_target[idp];
                // confirm that idsel is not in target
                flag_in = 0;
                for (int i=0; i<ntp*num_trans+n; i++){
                    if ((idsel==sel_tneuron[i]) || (idsel == sel_rneuron[i])){
                        flag_in = 1;
                        break;
                    }
                }
                nitr += 1;

                if (nitr > 1000){
                    fprintf(stderr, "Cannot find the target neuron for %d (%d)\n", n0, nstack);
                    exit(1);
                }
            }

            sel_rneuron[ntp*num_trans+n] = idsel;
            free(id_target);            
        }
    }

    // disconnect inter-population connection
    /*
    Transmitting neuron (T neuron) projects single dendrite to the target neuron in different population 
        T neurons receive the same input as the other neurons in same population, but the only difference is in projection
    Receiver neuron (R neuron) do not project its dendrite to the opposite population. 
        They only receive inter-population input from a corresponding T neuron, and from within population neurons
    */

    // initialize different delays
    for (int i=0; i<2; i++){
        int ntp_t = 2*i;
        desyn_t *syn = nnpop.syns+ntp_t;
        syn->is_const_delay = false;

        syn->n_delays = (int**) malloc(N * sizeof(int*));
        for (int n=0; n<N; n++){
            syn->n_delays[n] = (int*) calloc(syn->num_indeg[n], sizeof(int));
        }
    }

    int pop_range[4][2] = {0,};
    pop_range[0][0] = cell_range[0][0];
    pop_range[0][1] = cell_range[1][1];
    pop_range[1][0] = pop_range[0][0];
    pop_range[1][1] = pop_range[0][1];
    pop_range[2][0] = cell_range[2][0];
    pop_range[2][1] = cell_range[3][1];
    pop_range[3][0] = pop_range[2][0];
    pop_range[3][1] = pop_range[2][1];
    

    for (int n=0; n<2*num_trans; n++){
        int ntp_t = 2*(sel_tneuron[n]/nhalf); // 0, 2
        int ntp_r = 2-ntp_t;                  // 2, 0

        // disconnect inter-population connection from T
        int flag = 0;
        desyn_t syn = nnpop.syns[ntp_t];

        // syn.is_const_delay = false;
        // syn.n_delays = (int**) malloc(N * sizeof(int*));
        // for (int n=0; n<N; n++){
        //     syn.n_delays[n] = (int*) calloc(syn.num_indeg[n], sizeof(int));
        // }

        // disconnect inter-population projection from T neurons
        for (int npost=pop_range[ntp_r][0]; npost<pop_range[ntp_r][1]; npost++){
            for (int i=0; i<syn.num_indeg[npost]; i++){
                int npre = syn.indeg_list[npost][i];
                if (npre == sel_tneuron[n]){ // from T neuron
                    if (npost == sel_rneuron[n]){
                        double w = syn.w_list[npost][i];

                        #ifdef TEST
                        syn.w_list[npost][i] = w * 50;
                        syn.n_delays[npost][i] = nd_trans;
                        // syn.n_delays[npost][i] = sel_tdelay[n]/_dt;
                        // syn.n_delays[npost][i] = 50;
                        #else
                        syn.w_list[npost][i] = w * w_ratio;
                        syn.n_delays[npost][i] = nd_trans;
                        #endif

                        flag = 1;
                    } else {
                        syn.indeg_list[npost][i] = -1;
                        syn.w_list[npost][i] = 0;
                    }
                    break; // originally, break is in here, which would not disconnect the transmitter neuron input to the others
                }
            }
        }

        if (flag == 0){
            fprintf(stderr, "Transmitter %d has not connected (recv: %d)\n", sel_tneuron[n], sel_rneuron[n]);
            fprintf(stderr, "Trange:\n");
            for (int i=0; i<2*num_trans; i++){
                fprintf(stderr, "%d, ", sel_tneuron[i]);
            }
            fprintf(stderr, "Rrange:\n");
            for (int i=0; i<2*num_trans; i++){
                fprintf(stderr, "%d, ", sel_rneuron[i]);
            }

            REPORT_ERROR("Connection Error\n");
        }
        
        // disconnect inter-population projection to R

        int id_r = sel_rneuron[n];
        
        for (int m=0; m<2; m++){
            syn = nnpop.syns[ntp_t+m];

            for (int i=0; i<syn.num_indeg[id_r]; i++){
                int npre = syn.indeg_list[id_r][i];
                if (npre == sel_tneuron[n]) continue;
                if ((npre >= pop_range[ntp_t][0]) && (npre < pop_range[ntp_t][1])){
                    
                    syn.indeg_list[id_r][i] = -1;
                    syn.w_list[id_r][i] = 0;
                }
            }
        }
        
        // disconnect inter-population connection from R
        syn = nnpop.syns[ntp_r];
        for (int npost=pop_range[ntp_t][0]; npost<pop_range[ntp_t][1]; npost++){
            for (int i=0; i<syn.num_indeg[npost]; i++){
                int npre = syn.indeg_list[npost][i];
                if (npre == sel_rneuron[n]){
                    syn.indeg_list[npost][i] = -1;
                    syn.w_list[npost][i] = 0;
                    break;
                }
            }
        }
    }

    #ifdef TEST
    for (int n=0; n<2*num_trans; n++){
        int id_pre=sel_tneuron[n], id_post=sel_rneuron[n];

        int ntp_t = 2*(id_pre/nhalf); // 0, 2
        desyn_t syn = nnpop.syns[ntp_t];

        // fprintf(stderr, "ntp_t: %d, ntp_r: %d, address: %x\n", ntp_t, ntp_r, syn.n_delays);
        
        int npost = -1;
        int is_find = 0;
        for (int i=0; i<syn.num_indeg[id_post]; i++){
            if (syn.indeg_list[id_post][i] == id_pre){
                is_find = 1;
                npost = i;
                break;
            }
        }
        
        if (is_find == 0){
            fprintf(stderr, "Cannot find matched neurons\n");
            exit(1);
        }
        
        int nd = syn.n_delays[id_post][npost];
        double w = syn.w_list[id_post][npost];
        // printf("%4d -selected (delay: %2d, w: %.2f)-> %4d\n", id_pre, nd, w, id_post);
    }
    #endif

    // #ifdef TEST
    // for (int ntp=0; ntp<2; ntp++){
    //     for (int n=0; n<N; n++){
    //         desyn_t syn = nnpop.syns[NSYN+ntp];
    //         if (syn.num_indeg[n] == 0) continue;
    //         printf("%4d <- %4d\n", n, syn.indeg_list[n][0]);
    //     }
    // }

    // for (int n=0; n<2*num_trans; n++){
    //     printf("%4d -selected-> %4d\n", sel_tneuron[n], sel_rneuron[n]);
    // }
    // #endif


    // write result
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "ID(T),ID(R)\n");
    for (int n=0; n<2*num_trans; n++){
        // fprintf(fp,"%d,%d,%d\n", sel_tneuron[n], sel_rneuron[n], sel_tdelay[n]);
        fprintf(fp,"%d,%d,%d\n", sel_tneuron[n], sel_rneuron[n], nd_trans);
    }
    fclose(fp);

    free(sel_tneuron);
    free(sel_rneuron);
    // free(sel_tdelay);

    // REPORT_ERROR("Test");
}
