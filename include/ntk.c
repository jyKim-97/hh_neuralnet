#include "ntk.h"


ntk_t get_empty_net(int N){
    ntk_t ntk;
    ntk.N = N;
    ntk.num_edges = (int*) calloc(N, sizeof(int));
    ntk.adj_list  = (int**) malloc(N * sizeof(int*));
    for (int n=0; n<N; n++){
        ntk.adj_list[n] = (int*) malloc(_block_size * sizeof(int));
    }
    ntk.edge_dir = indeg; // in-degree
    return ntk;
}


#define LEN(arr) (arr[1] - arr[0])


void gen_er_pout(ntk_t *ntk, double p_out, int pre_range[2], int post_range[2]){
    int num_post = LEN(post_range);
    gen_er_mdout(ntk, num_post*p_out, pre_range, post_range);
}


void gen_er_pin(ntk_t *ntk, double p_in, int pre_range[2], int post_range[2]){
    int num_pre = LEN(pre_range);
    gen_er_mdin(ntk, num_pre*p_in, pre_range, post_range);
}


void gen_er_mdout(ntk_t *ntk, double mdeg_out, int pre_range[2], int post_range[2]){
    double mdeg_in = cvt_mdeg_out2in(mdeg_out, LEN(pre_range), LEN(post_range));
    gen_er_mdin(ntk, mdeg_in, pre_range, post_range);
}


double cvt_mdeg_out2in(double mdeg_out, int num_pre, int num_post){
    return num_pre * mdeg_out / num_post;
}


void connect(ntk_t *ntk, int id_pre, int id_post){
    if (ntk->edge_dir == indeg){
        int id = ntk->num_edges[id_post];
        append_int(ntk->adj_list+id_post, id, id_pre);
        ntk->num_edges[id_post]++;

    } else if (ntk->edge_dir == outdeg){
        int id = ntk->num_edges[id_pre];
        append_int(ntk->adj_list+id_pre, id, id_post);
        ntk->num_edges[id_pre]++;

    } else {
        printf("Wrong edge direction type\n");
    }
}


void gen_er_mdin(ntk_t *ntk, double mdeg_in, int pre_range[2], int post_range[2]){
    int len = ntk->N; 
    int target_deg = mdeg_in * LEN(post_range);
    int num_pre = LEN(pre_range);
    int num_post = LEN(post_range);

    int total_deg = 0;
    int *used = (int*) calloc(len*len, sizeof(int));
    while (total_deg < target_deg){
        int npre  = genrand64_real2() * num_pre + pre_range[0];
        int npost = genrand64_real2() * num_post + post_range[0];
        // printf("total deg: %d, target deg: %d, npre=%d, npost=%d\n", total_deg, target_deg, npre, npost);
        if (used[npre*len + npost] == 1) continue;
        if (npre == npost) continue; // remove self-loop

        connect(ntk, npre, npost);

        used[npre*len + npost] = 1;
        total_deg++;
    }
    free(used);
}


void free_network(ntk_t *ntk){
    int N = ntk->N;
    for (int n=0; n<N; n++){
        free(ntk->adj_list[n]);
    }
    free(ntk->adj_list);
    free(ntk->num_edges);
}


void print_network(const char *fname, ntk_t *ntk){
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "network type: %d\n", ntk->edge_dir);
    for (int n=0; n<ntk->N; n++){
        int ne = ntk->num_edges[n];
        fprintf(fp, "%d:", ne);
        for (int i=0; i<ne; i++){
            fprintf(fp, "%d,", ntk->adj_list[n][i]);
        }
        fprintf(fp, "\n");
    }
}

