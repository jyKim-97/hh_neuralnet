/*
Community detection in the network
: Use Louvain method

Blondel, Vincent D., et al. "Fast unfolding of communities in large networks." Journal of statistical mechanics: theory and experiment 2008.10 (2008): P10008.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int **belong_id     = NULL; // original node id that included in the community
int *num_belong     = NULL; // the # of nodes that are included in the community
double *deg_in_com  = NULL; // sum of degrees inside the community
double *deg_com     = NULL; // sum of degrees in the community
double **deg_id2com = NULL; // degree from node id to each community


typedef struct _com_graph_t {
    int N; // # of nodes & community
    double **wmat; // weight matrix
    double *degree;
    double sum_deg2; // 2m
    int *num_each_com; // the # of elements in each commuinty
    int *id_com; // community id of each node, size=N
} com_graph_t;


#define POW(x) x*x
#define FREE2d(arr, arr_size) {for (int n=0; n<arr_size; n++) free(arr[n]);\
                               free(arr);}
#define CALLOC1d(N, dtype) (dtype*) calloc(N, sizeof(dtype))

// double **malloc2d_f(int N){
//     double **arr = (double**) malloc(sizeof(double*) * N);
//     for (int n=0; n<N; n++) arr[n] = (double*) calloc(N, sizeof(double));
//     return arr;
// }


// int **malloc2d_d(int N){
//     int **arr = (int**) malloc(sizeof(int*) * N);
//     for (int n=0; n<N; n++) arr[n] = (int*) calloc(N, sizeof(int));
//     return arr;
// }


int main(){
    char fname[100];
    com_graph_t net = load_graph(fname);

    double qmod = -1e6;
    double d_qmod = 1e6;
    double tol = 1e-3;

    while (d_qmod < tol){
        double qmod_new = UpdateCommunity_p1(&net);
        net = collapse_graph(&net);
        d_qmod = qmod_new - qmod;
        qmod = qmod_new;
    }
    
    /* NOTE: Export graph */

    /* destroy */
    
}


double UpdateCommunity_p1(com_graph_t *net){
    init_community_info(net);
    for (int n=0; n<net->N; n++){
        double dq_mod = flip_community(n, net);
    }
    destroy_community_info(net->N);

    double qmod = compute_modularity(net);
    return qmod;
}


double init_community_info(com_graph_t *net){
    int N = net->N;

    double *deg_com     = (double*) calloc(N, sizeof(double));
    double *deg_in_com  = (double*) calloc(N, sizeof(double));
    double **deg_id2com = malloc2d_f(N);

    for (int n=0; n<N; n++){
        deg_in_com[n] = net->wmat[n][n];
        deg_com[n]    = net->degree[n];
        deg_id2com[n][n] = net->wmat[n][n];
    }
}


void destroy_community_info(int N){
    free(deg_in_com); deg_in_com = NULL;
    free(deg_com);    deg_com = NULL;
    FREE2d(deg_id2com, N); deg_id2com = NULL;
}


double flip_community(int id, com_graph_t *net){
    // flip the community of 'id' node 
    int N = net->N;

    double m2 = net->sum_deg2;
    double ki = net->degree[id];
    double *d_qmod = (double*) calloc(N, sizeof(double));
    for (int cn=0; cn<N; cn++){
        if (net->num_each_com[cn] == 0){
            continue;
        }

        d_qmod[cn] = (deg_in_com[cn] + 2*deg_id2com[id][cn])/m2 - POW((deg_com[cn] + ki)/m2);
        d_qmod[cn] -= deg_in_com[cn]/m2 - POW(deg_com[cn]/m2) - POW(ki/m2);
    }

    int com_new = -1;
    double d_qmod_max = -1;
    for (int n=0; n<N; n++){
        if ((d_qmod[n] > 0) && (d_qmod[n] > d_qmod_max)){
            com_new = n;
            d_qmod_max = d_qmod[n];
        }
    }

    if (com_new != -1) update_communify(id, com_new, net);
    free(d_qmod);

    return d_qmod_max;
}


void update_communify(int id, int com_new, com_graph_t *net){
    int com_old = net->id_com[id];
    net->id_com[id] = com_new;
    net->num_each_com[com_old] -= 1;
    net->num_each_com[com_new] += 1;

    if (net->num_each_com[com_old] < 0){
        printf("Invalid num_eac_com, %d", com_old);
        exit(-1);
    }

    deg_in_com[com_old] -= deg_id2com[id][com_old];
    deg_in_com[com_new] += deg_id2com[id][com_new];

    deg_com[com_old] -= net->degree[id];
    deg_com[com_new] += net->degree[id];

    double tmp = deg_id2com[id][com_new];
    deg_id2com[id][com_new] = deg_id2com[id][com_old];
    deg_id2com[id][com_old] = tmp;
}


double compute_modularity(com_graph_t *net){
    double qmod = 0;
    double m2 = net->sum_deg2;
    for (int i=0; i<net->N; i++){
        double di = net->degree[i];
        qmod += net->wmat[i][i] - di*di/m2;
        for (int j=i+1; j<net->N; j++){
            if (net->id_com[i] != net->id_com[j]) continue;
            qmod += 2*(net->wmat[i][j] - di*net->degree[j]/m2);
        }
    }
    return qmod;
}


com_graph_t load_graph(const char *fname){
    FILE *fp;
    if ((fp = fopen(fname, "r+")) == NULL){
        printf("%s not exists", fname);
        exit(-1);
    }

    int N;
    com_graph_t net;
    /* read */

    compute_degree(&net);

    num_belong = (int*) calloc(N, sizeof(int));
    belong_id  = (int**) malloc(sizeof(int*) * N);
    for (int n=0; n<N; n++){
        belong_id[n] = (int*) malloc(sizeof(int) * 1);
        belong_id[n] = n;
        num_belong[n] = 1;
    }

    return net;
}


com_graph_t collapse_graph(com_graph_t *net){
    // get the # of communities (remove if # of elements==0)
    int N=net->N, n_left = 0;
    int *cvt_table = (int*) calloc(N, sizeof(int));
    for (int n=0; n<N; n++){
        if (net->num_each_com[n] == 0){
            continue;
            cvt_table[n] = -1; // to avoid invalid indexing
        }
        cvt_table[n] = n_left;
        n_left++;
    }

    // ci, cj
    com_graph_t net_next = get_empty_graph(n_left);
    for (int i=0; i<N; i++){
        if (net->num_each_com[i] == 0) continue;
        int ci = cvt_table[net->id_com[i]];
        net_next.wmat[ci][ci] += net->wmat[i][i];
        for (int j=i+1; j<N; j++){
            if (net->num_each_com[j] == 0) continue;
            int cj = cvt_table[net->id_com[j]];
            net_next.wmat[ci][cj] += net->wmat[i][j];
            net_next.wmat[cj][ci] += net_next.wmat[ci][cj];
        }
    }
    compute_degree(&net_next);
    move_community(net, cvt_table, n_left);

    free(cvt_table);
    destroy_graph(net);
    return net_next;
}


int dsum(int arr_size, double *arr){
    int s = 0;
    for (int n=0; n<arr_size; n++) s += arr[n];
    return s;
}


void move_community(com_graph_t *net, int *cvt_table, int Nnew){
    int *num_belong_new = CALLOC1d(Nnew, int);
    for (int n=0; n<net->N; n++){
        if (net->num_each_com[n] == 0) continue;
        int cn = cvt_table[n]; // old community id -> new
        num_belong_new[cn] = dsum(belong_id[n], num_belong[n]);
    }
    
    int *num_alloc = CALLOC1d(Nnew, int);

    int **belong_id_new = (int**) malloc(sizeof(int*) * Nnew);
    for (int n=0; n<Nnew; n++) belong_id_new[n] = (int*) calloc(num_belong_new[n], sizeof(int));
    for (int n=0; n<net->N; n++){
        int cn = cvt_table[net->id_com[n]];
        for (int i=0; i<num_belong[n]; i++){
            belong_id_new[cn][i+num_alloc[cn]++] = belong_id[n][i];
        }
    }
    
    // copy
    free(num_belong);
    num_belong = (int*) malloc(sizeof(int) * Nnew);
    memcpy(num_belong, num_belong_new, sizeof(int) * Nnew);
    
    FREE2d(belong_id, net->N);
    belong_id = (int**) malloc(Nnew * sizeof(int*));
    for (int n=0; n<Nnew; n++){
        int nsize = sizeof(int) * num_belong[n];
        belong_id[n] = (int*) malloc(nsize);
        memcpy(belong_id[n], belong_id_new[n], nsize);
    }

    FREE2d(belong_id_new, Nnew);
    free(num_belong_new);
    free(num_alloc);
}


void compute_degree(com_graph_t *net){
    for (int i=0; i<net->N; i++){
        net->degree[i] += net->wmat[i][i];
        for (int j=i+1; j<net->N; j++){
            net->degree[i] += net->wmat[i][j];
            net->degree[j] += net->degree[i];
        }

        net->sum_deg2 += net->degree[i];
    }
}


void destroy_graph(com_graph_t *net){
    free(net->num_each_com);
    free(net->id_com);
    FREE2d(net->wmat, net->N);
}


com_graph_t get_empty_graph(int N){
    com_graph_t net;

    net.N        = N;
    net.wmat     = malloc2d_f(N);
    net.id_com   = (int*) malloc(sizeof(int) * N);
    net.num_each_com = (int*) malloc(sizeof(int) * N);
    net.degree   = (double*) calloc(N, sizeof(double));
    net.sum_deg2 = 0;

    for (int n=0; n<N; n++){
        net.id_com[n] = n;
        net.num_each_com[n] = 1;
    }

    return net;
}

