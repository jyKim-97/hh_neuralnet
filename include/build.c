// source code for building the network model
#include "build.h"

// extern neuron_t neuron;
// extern syn_t syn[MAX_TYPE];

void build_wb_ipop(int N, neuron_t *neuron, syn_t *syn_i, double w, double t_lag, enum odeType type){

    double del_t = 0;
    switch (type) {
        case Euler:
            del_t = _dt;
            break;
        case RK4:
            del_t = _dt * 0.5;
            break;
    }

    // initializing objects
    int n_lag = t_lag / _dt;
    init_wbNeuron(N, n_lag, neuron);
    init_deSyn(N, -80, del_t, syn_i);

    // set network
    build_homogen_net(&(syn_i->ntk), w, n_lag);
}


void build_homogen_net(netsyn_t *ntk, double w, int n_lag){
    int N = ntk->N;

    for (int npre=0; npre<N; npre++){
        ntk->num_edges[npre] = N-1;

        int id = 0;
        for (int npost=0; npost<N; npost++){
            if (npre == npost) continue;
            ntk->adj_list[npre][id] = npost;
            ntk->weight_list[npre][id] = w;
            ntk->n_delay[npre][id] = n_lag;
            id ++;
        }
    }
}


void build_eipop(buildInfo *info){

    double del_t = 0;
    switch (info->ode_method){
        case Euler: // Euler
            del_t = _dt;
            break;
    case RK4:
            del_t = _dt/2;
            break;
    }

    int N = info->N;
    init_wbNeuron(N, info->buf_size, &neuron);
    init_deSyn(N,   0, del_t, &(syn[0])); // type E
    init_deSyn(N, -80, del_t, &(syn[1])); // type I

    int id_pre = 0;
    for (int i=0; i<2; i++){ // E / I
        int id_post = 0;
        for (int j=0; j<2; j++){ // E / I
            
            int mean_outdeg = info->mean_outdeg[i][j];
            double w = info->w[i][j];
            int n_lag = info->n_lag[i][j];

            int pre_range[2] = {id_pre, id_pre+info->num_types[i]};
            int post_range[2] = {id_post, id_post+info->num_types[j]};

            // printf("pre: %d-%d, post: %d-%d, mean_outdeg: %d\n", pre_range[0], pre_range[1], post_range[0], post_range[1], mean_outdeg);
            // printf("%d, %d: ", i, j);
            build_randomnet(&(syn[i].ntk), mean_outdeg, w, n_lag, pre_range, post_range);
            id_post += info->num_types[j];

            // printf("(%d, %d) - n_deg = %d\n", i, j, syn[0].ntk.adj_list[2][10]);
        }

        id_pre += info->num_types[i];
    }

    syn[0].ntk.pre_range[0] = 0;
    syn[0].ntk.pre_range[1] = info->num_types[0];
    syn[1].ntk.pre_range[0] = info->num_types[0];
    syn[1].ntk.pre_range[1] = N;
    for (int i=0; i<2; i++){
        syn[i].ntk.post_range[0] = 0;
        syn[i].ntk.post_range[1] = N;
    }
}


#define MAX(a, b) (a>b? a:b)
#define LEN(arr) (arr[1] - arr[0])

void build_randomnet(netsyn_t *ntk, int mean_outdeg, double w, int n_lag, int pre_range[2], int post_range[2]){
    // for (int npre=0; npre)
    // int len = MAX(pre_range[1], post_range[1]);
    int len = ntk->N;
    int *used = (int*) calloc(len*len, sizeof(int));    
    int num_pre  = LEN(pre_range);
    int num_post = LEN(post_range);

    int total_deg = 0;
    while (total_deg < num_pre*mean_outdeg){
        // NOTE: mean outdeg로 변화
        // select two 
        int npre  = genrand64_real2() * num_pre + pre_range[0];
        int npost = genrand64_real2() * num_post + post_range[0];
        if (used[npre*len + npost] == 1) continue;

        int id = ntk->num_edges[npre];
        ntk->adj_list[npre][id] = npost;
        ntk->weight_list[npre][id] = w;
        ntk->n_delay[npre][id] = n_lag;
        ntk->num_edges[npre]++;
        used[npre*len + npost] = 1;
        total_deg++;
    }

    // printf("post: %d\n", ntk->adj_list[2][10]);

    for (int i=0; i<2; i++){
        ntk->pre_range[i] = pre_range[i];
        ntk->post_range[i] = post_range[i];
    }

    free(used);
}


void print_network(char fname[], netsyn_t *ntk){
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "pre,%d,%d,", ntk->pre_range[0], ntk->pre_range[1]);
    fprintf(fp, "post,%d,%d,\n", ntk->post_range[0], ntk->post_range[1]);

    for (int n=0; n<ntk->N; n++){
        int num_post = ntk->num_edges[n];
        fprintf(fp, "%d:", num_post);
        for (int id=0; id<num_post; id++){
            fprintf(fp, "%d,", ntk->adj_list[n][id]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}