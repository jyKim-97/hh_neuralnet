
#include "model.h"
#include <stdio.h>


const double taur=1, taud=3;
static inline double get_minf(double v);
double _dt = 0.005;
wbparams_t *params;
int num_neuron_types = 2;
int curr_type = 0;


/*
    Rule for defining the neuron model
    - structure tree
        - int N
        - double *v
        - double ...ion
        - spkbuf_t buf
    - init_<name>Neuron(int N, <name>Neuron *neuron)
    - destroy_<name>Neuron(<name>Neuron *neuron)
    - solve_<name>_v(double v, double I, double ...)
*/


/* 
    Definition about Wang-Buzsaki Neuron model
*/

void init_wbNeuron(int N, int buf_size, neuron_t *neuron){

    neuron->N = N;
    neuron->v = (double*) malloc(sizeof(double) * N);
    neuron->h_ion = (double*) malloc(sizeof(double) * N);
    neuron->n_ion = (double*) malloc(sizeof(double) * N);

    for (int n=0; n<N; n++){
        neuron->v[n] = -70; // initializing with constant value
        neuron->h_ion[n] = 0;
        neuron->n_ion[n] = 0;
    }

    init_spkBuf(N, buf_size, &(neuron->buf));

    params = (wbparams_t*) malloc(sizeof(wbparams_t) * num_neuron_types);
    for (int n=0; n<num_neuron_types; n++){
        params[n].cm  = 1;
        params[n].phi = 5;
        params[n].gl  = 0.1;
        params[n].gna = 35;
        params[n].gk  = 9;
    }
}


void destroy_wbNeuron(neuron_t *neuron){
    free(neuron->v);
    free(neuron->h_ion);
    free(neuron->n_ion);
    destroy_spkBuf(&(neuron->buf));
    free(params);
}


int print_id = 0;
double solve_wb_v(double v, double I, double h_ion, double n_ion){

    if (curr_type >= num_neuron_types){
        printf("current type exceed the total num of neuron types\n");
    }

    // get ion current
    double m_ion = get_minf(v);
    // double ina = wb_gna * pow(m_ion, 3) * h_ion * (v - wb_ena);
    // double ik  = wb_gk * pow(n_ion, 4) * (v - wb_ek);
    double ina = params[curr_type].gna * m_ion * m_ion * m_ion * h_ion * (v - wb_ena);
    double ik  = params[curr_type].gk * n_ion * n_ion * n_ion * n_ion * (v - wb_ek);
    double il  = params[curr_type].gl * (v - wb_el);

    // printf("ina: %5.2f, ik: %5.2f, il: %5.2f, I: %5.2f\n", ina, ik, il, I);
    // printf("n_ion4: %10.2f\n", n_ion * n_ion * n_ion * n_ion);

    double dv = (-ina - ik - il + I) / params[curr_type].cm;

    // if (print_id == 1){
    // printf("gna: %12.10f, gk: %12.10f, gl: %12.10f, cm: %12.10f\n", 
    //         params[curr_type].gna, params[curr_type].gk, params[curr_type].gl, params[curr_type].cm);
    // printf("ena: %12.10f, ek: %12.10f, el: %12.10f\n", wb_ena, wb_ek, wb_el);
    // printf("m: %12.10f, h: %12.10f, n: %12.10f, i: %12.10f\n", m_ion, h_ion, n_ion, I);
    // printf("ina: %12.10f, ik: %12.10f, il: %12.10f\n", ina, ik, il);
    // printf("v: %12.10f, dv: %12.10f\n", v, dv);
        
    // }

    return _dt * dv;
}


static inline double get_minf(double v){
    double am = -0.1 * (v+35) / (exp(-0.1 * (v + 35)) - 1);
    double bm = 4 * exp(-(v + 60)/18);
    return am / (am + bm);
}


double solve_wb_h(double h_ion, double v){
    double ah = 0.07 * exp(-(v + 58)/20);
    double bh = 1 / (exp(-0.1 * (v + 28)) + 1);
    return _dt * params[curr_type].phi * (ah * (1-h_ion) - bh * h_ion);
}


double solve_wb_n(double n_ion, double v){
    double an = -0.01 * (v + 34) / (exp(-0.1 * (v + 34)) - 1);
    double bn = 0.125 * exp(-(v + 44)/80);
    return _dt * params[curr_type].phi * (an * (1-n_ion) - bn * n_ion);
}

/*
    Definition about Spike buffer 
*/

void init_spkBuf(int N, double nd_max, spkbuf_t *buf){
    buf->N = N;
    buf->buf_size = nd_max; // TODO: nd_max = 0 일때는? (no-delay)
    buf->spk_buf = (int**) malloc(sizeof(int*) * N);
    for (int n=0; n<N; n++){
        buf->spk_buf[n] = (int*) calloc(nd_max, sizeof(int));
    }
}


void destroy_spkBuf(spkbuf_t *buf){
    for (int n=0; n<buf->N; n++){
        free(buf->spk_buf[n]);
    }
    free(buf->spk_buf);
}


#define THRESHOLD 0
#define FIRE(vold, vnew) ((vold-THRESHOLD < 0) && (vnew-THRESHOLD > 0))

void update_spkBuf(int nstep, spkbuf_t *buf, double *v_old, double *v_new){
    int buf_size = buf->buf_size;
    int n_buf = (buf_size == 0)? 0: (nstep+1) % buf_size;

    for (int n=0; n<buf->N; n++){
        if FIRE(v_old[n], v_new[n]){
            // printf("update spk: nstep=%d, pre=%d\n", nstep, n);
            // printf("Pre synaptic neuron fired at %d\n", nstep);
            buf->spk_buf[n][n_buf] = 1;
            // printf("Neuron %d fired in step %d\n", n, nstep);
        } else {
            buf->spk_buf[n][n_buf] = 0;
        }
    }
}


void init_ext_syn(int N, double dt, syn_t *syn){
    init_deSyn(N, 0, dt, syn);
}


void init_deSyn(int N, double ev, double dt, syn_t *syn){

    syn->N = N;
    syn->expr = (double*) calloc(N, sizeof(double));
    syn->expd = (double*) calloc(N, sizeof(double));

    syn->mul_expr = exp(-dt / taur);
    syn->mul_expd = exp(-dt / taud);
    syn->ev = ev;

    double tp = taur * taud / (taud - taur) * log(taud/taur);
    syn->A = 1 / (exp(-tp/taur) - exp(-tp/taud));

    // printf("tp=%.2f, A=%10.3f\n", tp, syn->A);

    init_netSyn(N, &(syn->ntk));
}


void destroy_deSyn(syn_t *syn){
    free(syn->expr);
    free(syn->expd);
}


void add_spike_syn(syn_t *syn, int post_id, int nstep, spkbuf_t *buf){

    // update가 이상함: tracking해서 체크해봐야될듯?
    // int N = syn->N;
    double A = syn->A;
    int buf_size = buf->buf_size;
    int num_pre = syn->ntk.num_edges[post_id];

    for (int n=0; n<num_pre; n++){
        int nd = syn->ntk.n_delay[post_id][n];
        if (nstep - nd < 0) continue;
        int n_buf = (buf_size == 0)? 0: (nstep - nd) % buf_size;

        int npre = syn->ntk.adj_list[post_id][n];
        if (buf->spk_buf[npre][n_buf] == 1){
            // printf("update syn: nstep=%d, pre=%d, post=%d, nd=%d, n_buf=%d\n", nstep, npre, post_id, nd, buf_size);
            // printf("n_buf: %d, nd: %d, buf_size: %d, npre: %d\n", n_buf, nd, buf_size, npre);
            double wA = syn->ntk.weight_list[post_id][n] * A;

            // printf("update syn: step: %d, with lag: %d, bufsize: %d\n", nstep, nd, buf_size);

            // printf("Spike detedted in step %d, with lag = %d\n", nstep, nd);

            syn->expr[post_id] += wA;
            syn->expd[post_id] += wA;
        }
    }
}


// Legacy code
/*
void add_spike_deSyn(syn_t *syn, int nstep, spkbuf_t *buf){
    int N = syn->N;
    int buf_size = buf->buf_size;

    // NOTE: indegree 저장 방식으로 바꾸면 더 빠르게 가능할 것 같은데?

    for (int npre=0; npre<N; npre++){
        int num_post = syn->ntk.num_edges[npre];

        for (int id=0; id<num_post; id++){
            int nd = syn->ntk.n_delay[npre][id];
            if (nstep - nd < 0) continue;
            int n_buf = (buf_size == 0)? 0: (nstep - nd) % buf_size;

            if (buf->spk_buf[npre][n_buf] == 1){
                // fprintf(stderr, "num_post = %4d\n", num_post);
                // fprintf(stderr, "npre=%3d, num_post=%4d, id=%d\n", npre, num_post, id);

                // printf("add exp, npre=%d, nstep=%d, n_buf=%d, buf_size=%d\n", npre, nstep, n_buf, buf->buf_size);
                int npost = syn->ntk.adj_list[npre][id];
                double w  = syn->ntk.weight_list[npre][id];
                double A = syn->A;

                // fprintf(stderr, "allocating to %4d...", npost);
                syn->expr[npost] += w * A;
                syn->expd[npost] += w * A;
                // fprintf(stderr, "done\n");
            }
        }
    }
}
*/


void update_deSyn(syn_t *syn, int id){
    syn->expr[id] *= syn->mul_expr;
    syn->expd[id] *= syn->mul_expd;
}


double get_current_deSyn(syn_t *syn, int id, double vpost){
    double expr = syn->expr[id];
    double expd = syn->expd[id];
    return (expr - expd) * (vpost - syn->ev);
    // return 
}

// todo: synapse에 normalization constant 넣어주기
void init_netSyn(int N, netsyn_t *ntk){
    ntk->N = N;
    ntk->num_edges = (int*) calloc(N, sizeof(int));
    ntk->adj_list = (int**) malloc(N * sizeof(int*));
    ntk->n_delay = (int**) malloc(N * sizeof(int*));
    ntk->weight_list = (double**) malloc(N * sizeof(double*));

    for (int n=0; n<N; n++){
        ntk->adj_list[n] = (int*) malloc(N * sizeof(int));
        ntk->n_delay[n] = (int*) malloc(N * sizeof(int));
        ntk->weight_list[n] = (double*) malloc(N * sizeof(double));

        // give NULL
        for (int i=0; i<N; i++){
            ntk->adj_list[n][i] = -1;
            ntk->n_delay[n][i] = -100;
            ntk->weight_list[n][i] = -100;
        }
    }
}


void destroy_netSyn(netsyn_t *ntk){
    int N = ntk->N;
    for (int npre=0; npre<N; npre++){
        free(ntk->adj_list[npre]);
        free(ntk->n_delay[npre]);
        free(ntk->weight_list[npre]);
    }

    free(ntk->num_edges);
    free(ntk->adj_list);
    free(ntk->n_delay);
    free(ntk->weight_list);
}


// double *solve_wbNeuron(double wb_v, double wb_h, double wb_n, double isyn, double iapp){
//     double am = wb_am(wb_v);
//     double bm = wb_bm(wb_v);
//     double ah = wb_ah(wb_v);
//     double bh = wb_bh(wb_v);
//     double an = wb_an(wb_v);
//     double bn = wb_bn(wb_v);

//     double m_inf = am / (am + bm);
//     double ina = wb_gna * pow(m_inf, 3) * wb_h * (wb_v - wb_ena);
//     double ik  = wb_gk * pow(wb_n, 4) * (wb_v - wb_ek);
//     double il  = wb_gl * (wb_v - wb_el);

//     // printf("v: %6.2f, ina: %6.2f, ik: %6.2f, il: %6.2f, m: %6.2f, n: %6.2f, h: %6.2f\n", wb_v, ina, ik, il, m_inf, wb_n, wb_h);

//     double dv = (-ina - ik - il - isyn + iapp) / wb_cm;
//     double dh = wb_phi * (ah * (1-wb_h) - bh * wb_h);
//     double dn = wb_phi * (an * (1-wb_n) - bn * wb_n);

//     double *dx = (double*) malloc(sizeof(double) * 3);
//     dx[0] = dv; dx[1] = dh; dx[2] = dn;
//     return dx;
// }