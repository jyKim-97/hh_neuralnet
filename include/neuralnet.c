#include "neuralnet.h"


// double iapp = 0;
int num_syn_types = 2;
wbneuron_t neuron;
desyn_t syns[MAX_TYPE], ext_syn;
double taur=1, taud=3;

static void add_spike_total_syns(int nstep);
static void update_total_syns(int nid);
static double get_total_syns_current(int nid, double vpost);
static void fprintf2d_d(FILE *fp, double x[MAX_TYPE][MAX_TYPE]);


void build_rk4(nn_info_t *info){
    int N = info->N;
    // set with default parameter ranges
    init_wbneuron(N, &neuron);
    init_desyn(N, &ext_syn);
    for (int n=0; n<num_syn_types; n++){
        init_desyn(N, &syns[n]);
    }

    set_attrib(&syns[0],   0, taur, taud, 0.5);
    set_attrib(&syns[1], -80, taur, taud, 0.5);
    set_attrib(&ext_syn,   0, taur, taud, 0.5);
    set_poisson(&ext_syn, info->nu_ext, info->w_ext);

    // generate network
    int e_range[2] = {0, N*0.8};
    int i_range[2] = {N*0.8, N};
    ntk_t ntk_e = get_empty_net(N);
    gen_er_mdin(&ntk_e, info->mdeg_in[0][0], e_range, e_range);
    gen_er_mdin(&ntk_e, info->mdeg_in[0][1], e_range, i_range);
    set_network(&syns[0], &ntk_e);
    // free_network(&ntk_e);

    ntk_t ntk_i = get_empty_net(N);
    gen_er_mdin(&ntk_i, info->mdeg_in[1][0], i_range, e_range);
    gen_er_mdin(&ntk_i, info->mdeg_in[1][1], i_range, i_range);
    set_network(&syns[1], &ntk_i);
    // free_network(&ntk_i);
    
    set_coupling(&syns[0], e_range, e_range, info->w[0][0]);
    set_coupling(&syns[0], e_range, i_range, info->w[0][1]);
    set_coupling(&syns[1], i_range, e_range, info->w[1][0]);
    set_coupling(&syns[1], i_range, i_range, info->w[1][1]);
    check_coupling(&syns[0]);
    check_coupling(&syns[1]);

    set_const_delay(&syns[0], info->t_lag/_dt);
    set_const_delay(&syns[1], info->t_lag/_dt);
}


void write_info(nn_info_t *info, char *fname){
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "Size: %d\n", info->N);
    fprintf(fp, "ntypes: %d\n", num_syn_types);
    fprintf(fp, "type_range:\n");
    fprintf(fp, "mean indegree:\n");
    fprintf2d_d(fp, info->mdeg_in);
    fprintf(fp, "w:\n");
    fprintf2d_d(fp, info->w);
    fprintf(fp, "t_lag: %f\n", info->t_lag);
    fprintf(fp, "nu_pos: %f\n", info->nu_ext);
    fprintf(fp, "w_pos: %f\n", info->w_ext);

    fprintf(fp, "Additional\n");
    fprintf(fp, "connection_prob:\n");
    int N = info->N;
    fprintf(fp, "e->e: %f\n", info->mdeg_in[0][0]/(0.8*N));
    fprintf(fp, "e->i: %f\n", info->mdeg_in[0][1]/(0.8*N));
    fprintf(fp, "i->e: %f\n", info->mdeg_in[1][0]/(0.2*N));
    fprintf(fp, "i->i: %f\n", info->mdeg_in[1][0]/(0.2*N));
    fclose(fp);
}


void update_rk4(int nstep, double iapp){
    int N = neuron.N;

    double *v_prev = (double*) malloc(sizeof(double) * N);
    add_spike_total_syns(nstep);

    for (int id=0; id<N; id++){
        v_prev[id] = neuron.vs[id];
        wbparams_t *ptr_params = neuron.params+id;
        double v0=neuron.vs[id], h0=neuron.hs[id], n0=neuron.ns[id];
        
        // solve 1st
        double isyn = get_total_syns_current(id, v0);
        double dv1 = solve_wb_v(ptr_params, v0, h0, n0, iapp-isyn);
        double dh1 = solve_wb_h(ptr_params, h0, v0);
        double dn1 = solve_wb_n(ptr_params, n0, v0);

        // solve 2nd
        double v1=v0+dv1*0.5, h1=h0+dh1*0.5, n1=n0+dn1*0.5;
        update_total_syns(id);
        isyn = get_total_syns_current(id, v1);
        double dv2 = solve_wb_v(ptr_params, v1, h1, n1, iapp-isyn);
        double dh2 = solve_wb_h(ptr_params, h1, v1);
        double dn2 = solve_wb_n(ptr_params, n1, v1);

        // solve 3rd
        double v2=v0+dv2*0.5, h2=h0+dh2*0.5, n2=n0+dn2*0.5;
        isyn = get_total_syns_current(id, v2);
        double dv3 = solve_wb_v(ptr_params, v2, h2, n2, iapp-isyn);
        double dh3 = solve_wb_h(ptr_params, h2, v2);
        double dn3 = solve_wb_n(ptr_params, n2, v2);

        // solve 2nd
        double v3=v0+dv3, h3=h0+dh3, n3=n0+dn3;
        update_total_syns(id);
        isyn = get_total_syns_current(id, v3);
        double dv4 = solve_wb_v(ptr_params, v3, h3, n3, iapp-isyn);
        double dh4 = solve_wb_h(ptr_params, h3, v3);
        double dn4 = solve_wb_n(ptr_params, n3, v3);

        neuron.vs[id] = v0 + 1./6*(dv1 + 2*dv2 + 2*dv3 + dv4);
        neuron.hs[id] = h0 + 1./6*(dh1 + 2*dh2 + 2*dh3 + dh4);
        neuron.ns[id] = n0 + 1./6*(dn1 + 2*dn2 + 2*dn3 + dn4);
    }

    check_fire(&neuron, v_prev);
    free(v_prev);
}


void destroy_neuralnet(void){
    destroy_wbneuron(&neuron);
    for (int n=0; n<num_syn_types; n++){
        destroy_desyn(&syns[n]);
    }
    destroy_desyn(&ext_syn);
    #ifdef USE_MKL
    end_stream();
    #endif
}


static void add_spike_total_syns(int nstep){
    for (int n=0; n<num_syn_types; n++){
        add_spike(nstep, &syns[n], &neuron);
    }
    add_ext_spike(&ext_syn);
}


static void update_total_syns(int nid){
    for (int n=0; n<num_syn_types; n++){
        update_desyn(&syns[n], nid);
    }
    update_desyn(&ext_syn, nid);
}


static double get_total_syns_current(int nid, double vpost){
    double isyn = get_current(&ext_syn, nid, vpost);
    for (int n=0; n<num_syn_types; n++){
        isyn += get_current(&syns[n], nid, vpost);
    }
    return isyn;
}


static void fprintf2d_d(FILE *fp, double x[MAX_TYPE][MAX_TYPE]){
    for (int i=0; i<num_syn_types; i++){
        for (int j=0; j<num_syn_types; j++){
            fprintf(fp, "%f, ", x[i][j]);
        }
        fprintf(fp, "\n");
    }
}


