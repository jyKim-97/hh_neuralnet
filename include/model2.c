#include "model2.h"


#define IN(x, range) ((x >= range[0]) && (x < range[1]))
#define _inf 10000000
#define REPORT_ERROR(msg) print_error(msg, __FILE__, __LINE__)

static inline double get_minf(double v);

double _dt = 0.01;
int flag_nan = 0;


void init_wbneuron(int N, wbneuron_t *neuron){
    neuron->N = N;
    neuron->vs   = (double*) malloc(sizeof(double) * N);
    neuron->hs   = (double*) malloc(sizeof(double) * N);
    neuron->ns   = (double*) malloc(sizeof(double) * N);
    neuron->is_spk = (int*) malloc(sizeof(int) * N);
    neuron->params = (wbparams_t*) malloc(sizeof(wbparams_t) * N);

    for (int n=0; n<N; n++){
        neuron->vs[n] = -70; // initializing with constant value
        neuron->hs[n] = 0;
        neuron->ns[n] = 0;
        neuron->is_spk[n] = 0;
        set_default_wbparams(&neuron->params[n]);
    }
    
    for (int n=0; n<_spk_buf_size; n++){
        neuron->spk_buf[n] = (int*) calloc(N, sizeof(int));
    }
    neuron->spk_count = 0;
}


void set_default_wbparams(wbparams_t *params){
    params->phi = 5;
    params->cm  = 1;
    params->gl  = 0.1;
    params->gna = 35;
    params->gk  = 9;
}


void destroy_wbneuron(wbneuron_t *neuron){
    free(neuron->vs);
    free(neuron->hs);
    free(neuron->ns);
    free(neuron->params);
    free(neuron->is_spk);
    for (int n=0; n<_spk_buf_size; n++){
        free(neuron->spk_buf[n]);
    }
}


static inline double get_minf(double v){
    double am = -0.1 * (v+35) / (exp(-0.1 * (v + 35)) - 1);
    double bm = 4 * exp(-(v + 60)/18);
    return am / (am + bm);
}

int print_id = 0;
double solve_wb_v(wbparams_t *params, double v, double h, double n, double iapp){
    double m = get_minf(v);
    double ina = params->gna*m*m*m*h*(v - ena);
    double ik  = params->gk*n*n*n*n*(v - ek);
    double il  = params->gl*(v - el);
    double dv = (-ina-ik-il+iapp) / params->cm;
    if (isnan(dv)){
        flag_nan = 1;
    }

    return _dt * dv;
}


double solve_wb_h(wbparams_t *params, double h, double v){
    double ah = 0.07 * exp(-(v + 58)/20);
    double bh = 1 / (exp(-0.1 * (v + 28)) + 1);
    return _dt * params->phi * (ah * (1-h) - bh * h);
}


double solve_wb_n(wbparams_t *params, double n, double v){
    double an = -0.01 * (v + 34) / (exp(-0.1 * (v + 34)) - 1);
    double bn = 0.125 * exp(-(v + 44)/80);
    return _dt * params->phi * (an * (1-n) - bn * n);
}


void check_fire(wbneuron_t *neuron, double *v_prev){
    // 시간 맞는지 체크하기
    int nbuf = neuron->spk_count;
    for (int n=0; n<neuron->N; n++){
        if FIRE(v_prev[n], neuron->vs[n]){
            neuron->spk_buf[nbuf][n] = 1;
            neuron->is_spk[n] = 1;
        } else {
            neuron->spk_buf[nbuf][n] = 0;
            neuron->is_spk[n] = 0;
        }
    }
    neuron->spk_count = nbuf==_spk_buf_size-1? 0: nbuf+1;
}


void init_desyn(int N, desyn_t *syn){
    syn->N = N;
    syn->num_indeg  = NULL; //(int*) calloc(N, sizeof(int));
    syn->indeg_list = NULL; //(int**) malloc(N * sizeof(int*));
    syn->expr = (double*) calloc(N, sizeof(double));
    syn->expd = (double*) calloc(N, sizeof(double));
    syn->w_list = NULL;
    syn->w_ext  = NULL;
    syn->expl   = NULL;
    syn->nu_ext = NULL;

    syn->is_ext = false;
    syn->load_ntk    = false;
    syn->load_delay  = false;
    syn->load_w      = false;
    syn->load_attrib = false;
}


void set_attrib(desyn_t *syn, double ev, double taur, double taud, double ode_factor){
    syn->ev   = ev;
    syn->taur = taur;
    syn->taud = taud;
    syn->mul_expr = exp(-_dt*ode_factor / taur);
    syn->mul_expd = exp(-_dt*ode_factor / taud);

    double tp = taur * taud / (taud - taur) * log(taud/taur);
    syn->A = 1 / (exp(-tp/taur) - exp(-tp/taud));
    syn->load_attrib = true;
}


void set_network(desyn_t *syn, ntk_t *ntk){
    /*
    Set synaptic network (in-) to the given network structure (ntk)
    */
   
    if (ntk->edge_dir != indeg) REPORT_ERROR("Network data type must be indegree! Cannot set synapse");
    if (syn->N != ntk->N) REPORT_ERROR("Synapse size and Network size are different");

    int N = syn->N;
    syn->num_indeg = (int*) calloc(N, sizeof(int));
    syn->indeg_list = (int**) malloc(N * sizeof(int*));
    for (int npost=0; npost<N; npost++){
        int num_pre = ntk->num_edges[npost];
        syn->num_indeg[npost] = num_pre;
        syn->indeg_list[npost] = (int*) malloc(num_pre * sizeof(int));
        for (int i=0; i<num_pre; i++){
            syn->indeg_list[npost][i] = ntk->adj_list[npost][i];
        }
    }

    syn->load_ntk = true;
}


void set_const_coupling(desyn_t *syn, double w){
    int N = syn->N;

    syn->w_ext = (double*) malloc(sizeof(double) * N);
    for (int n=0; n<N; n++){
        syn->w_ext[n] = w;
    }

    syn->is_ext = true;
    syn->load_w = true;
}


void set_gaussian_coupling(desyn_t *syn, double w_mu, double w_sd){
    int N = syn->N;

    syn->w_ext = (double*) malloc(sizeof(double) * N);
    for (int n=0; n<N; n++){
        syn->w_ext[n] = genrand64_normal(w_mu, w_sd);
    }

    syn->is_ext = true;
    syn->load_w = true;
}



void set_coupling(desyn_t *syn, int pre_range[2], int post_range[2], double target_w){
    if (!syn->load_ntk){
        printf("Network is not loaded, coupling strength cannot be updated\n");
    }

    int N = syn->N;
    int gen_flag = 0;
    if (syn->w_list == NULL){
        gen_flag = 1;
        syn->w_list = (double**) malloc(sizeof(double*) * N);
        // w_list is 2-dim,: (npost, npre index
        // printf("%d, %d - %d, %d\n", pre_range[0], pre_range[1], post_range[0], post_range[1]);
    }

    for (int id_post=0; id_post<N; id_post++){
        int npre = syn->num_indeg[id_post];
        
        if (gen_flag == 1){
            // printf("%d, %d - %d, %d, %d\n", pre_range[0], pre_range[1], post_range[0], post_range[1], id_post);
            syn->w_list[id_post] = (double*) calloc(npre, sizeof(double));
            for (int n=0; n<npre; n++) syn->w_list[id_post][n] = _inf;
        }

        for (int n=0; n<npre; n++){
            int id_pre = syn->indeg_list[id_post][n];
            if (IN(id_pre, pre_range) && IN(id_post, post_range)){
                syn->w_list[id_post][n] = target_w;
            }
        }
    }

    syn->is_ext = false;
    syn->load_w = true;
}


void check_coupling(desyn_t *syn){
    int N = syn->N;
    if (syn->is_ext) return;

    for (int id_post=0; id_post<N; id_post++){
        int num_pre = syn->num_indeg[id_post];
        for (int n=0; n<num_pre; n++){
            if (syn->w_list[id_post][n] > _inf/2.){
                int id_pre = syn->indeg_list[id_post][n];
                printf("coupling %d<-%d have no value\n", id_post, id_pre);
            }
        }
    }
}


void set_const_delay(desyn_t *syn, double td){
    syn->n_delay = td / _dt;
    syn->is_const_delay = true;
    syn->load_delay = true;

    if (syn->n_delay >= _spk_buf_size){
        printf("Line 230 in model2.c: Expected delay size exceeds spike buffer size, increase the buffer\n");
        exit(-1);
    }
}


void set_delay(desyn_t *syn, int pre_range[2], int post_range[2], double target_td){
    if (!syn->load_ntk){
        printf("Network is not loaded, delay cannot be updated\n");
    }
}


void add_spike(int nstep, desyn_t *syn, wbneuron_t *neuron){
    int nbuf=0;

    if (syn->is_const_delay) {
        if (nstep < syn->n_delay) return;
        nbuf = (nstep - syn->n_delay) % _spk_buf_size;
    }
    // printf("syn: %d\n", nbuf);
    
    int N = syn->N;
    for (int npost=0; npost<N; npost++){
        int num_pre = syn->num_indeg[npost];
        for (int n=0; n<num_pre; n++){
            int npre = syn->indeg_list[npost][n];

            if (!syn->is_const_delay){
                int dn = nstep - syn->n_delays[npost][n];
                if (dn < 0) continue;
                nbuf = dn % _spk_buf_size;
            }

            if (neuron->spk_buf[nbuf][npre] == 1){
                double wA = syn->w_list[npost][n] * syn->A;
                syn->expr[npost] += wA;
                syn->expd[npost] += wA;
            }
        }
    }
}


void update_desyn(desyn_t *syn, int nid){
    /*Update synaptic divided synaptic activity
    - nid: postsynpatic neuron ID */

    syn->expr[nid] *= syn->mul_expr;
    syn->expd[nid] *= syn->mul_expd;
}


double get_current(desyn_t *syn, int nid, double vpost){
    /* Get current of postsynaptic neuron from pre-
    nid = postsynaptic neuron ID
    vpost: membrane potential of post synaptic neuron (id=nid)
    */
    if (nid >= syn->N) return 0;
    return (syn->expr[nid] - syn->expd[nid]) * (vpost - syn->ev);
}


void destroy_desyn(desyn_t *syn){
    int N = syn->N;

    if (syn->num_indeg != NULL){
        free(syn->num_indeg);
        for (int n=0; n<N; n++){
            free(syn->indeg_list[n]);
        }
        free(syn->indeg_list);
    }

    if (syn->is_ext){
        free(syn->w_ext);
        free(syn->nu_ext);
        free(syn->expl);
    } else {
        for (int n=0; n<N; n++){
            free(syn->w_list[n]);
            // free(syn->indeg_list[n]);
        }
        free(syn->w_list);
        // free(syn->indeg_list);
    }

    if (!syn->is_const_delay){
        for (int n=0; n<N; n++){
            free(syn->n_delays[n]);
        }
        free(syn->n_delays);
    }

    free(syn->expr);
    free(syn->expd);
}


void init_extsyn(int N, desyn_t *ext_syn){
    ext_syn->N = N;
    ext_syn->expr = (double*) calloc(N, sizeof(double));
    ext_syn->expd = (double*) calloc(N, sizeof(double));
    
    ext_syn->num_indeg  = NULL;
    ext_syn->indeg_list = NULL;
    ext_syn->is_ext     = true;
    ext_syn->is_const_delay = true;
    
    ext_syn->load_ntk    = true;
    ext_syn->load_delay  = true;
    ext_syn->load_w      = false;
    ext_syn->load_attrib = false;
}


// void set_poisson(desyn_t *ext_syn, double nu, double w){
//     ext_syn->nu = nu;
//     ext_syn->w = w;
//     #ifdef USE_MKL
//     int N = ext_syn->N;
//     ext_syn->lambda = (double*) malloc(sizeof(double) * N);
//     for (int n=0; n<N; n++) ext_syn->lambda[n] = _dt/1000. * nu;
//     #else
//     ext_syn->expl = exp(-_dt/1000. * nu);
//     #endif
// }


void set_poisson(desyn_t *ext_syn, double nu_mu, double nu_sd, double w_mu, double w_sd){
    set_gaussian_coupling(ext_syn, w_mu, w_sd);
    int N = ext_syn->N;
    ext_syn->nu_ext = (double*) malloc(sizeof(double) * N);
    ext_syn->expl   = (double*) malloc(sizeof(double) * N);
    for (int n=0; n<N; n++){
        ext_syn->nu_ext[n] = genrand64_normal(nu_mu, nu_sd);
        ext_syn->expl[n] = exp(-_dt/1000.* ext_syn->nu_ext[n]);
    }
}



void add_ext_spike(desyn_t *ext_syn){
    int N = ext_syn->N;
    double A = ext_syn->A;
    // double wA = ext_syn->w * ext_syn->A;

    #ifdef USE_MKL
    int *num_ext_set = get_poisson_array_mkl(N, ext_syn->lambda);
    // #else
    // double expl = ext_syn->expl[n];
    #endif

    for (int n=0; n<N; n++){
        #ifdef USE_MKL
        int num_ext = num_ext_set[n];
        #else
        int num_ext = pick_random_poisson(ext_syn->expl[n]);
        #endif
        double wA = ext_syn->w_ext[n] * A;
        ext_syn->expr[n] += wA*num_ext;
        ext_syn->expd[n] += wA*num_ext;
    }

    #ifdef USE_MKL
    free(num_ext_set);
    #endif
}


void print_syn_network(desyn_t *syn, char *fname){
    FILE *fp = fopen(fname, "w");
    int N = syn->N;

    fprintf(fp, "adjacency list, w (N=%d)\n", N);
    for (int n=0; n<N; n++){
        int num_pre = syn->num_indeg[n];
        for (int i=0; i<num_pre; i++){
            double w = 0;
            if (syn->is_ext){
                w = syn->w_ext[n];
            } else {
                w = syn->w_list[n][i];
            }
            fprintf(fp, "%d<-%d,%f\n", n,syn->indeg_list[n][i], w);
        }

        // if ((n > 1500) && (n < 1600)){
        //     for (int i=0; i<num_pre; i++) printf("%d-%d\n", n, syn->indeg_list[n][i]);
        // }
    }
    fclose(fp);
}