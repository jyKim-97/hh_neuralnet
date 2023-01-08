#include "neuralnet.h"


const double ev_e = 0;
const double ev_i = -80;


// double iapp = 0;
int num_syn_types = 2;
wbneuron_t neuron;
desyn_t syns[MAX_TYPE], ext_syn;
double taur_default=0.3, taud_default=1;
int const_current = false;

static void add_spike_total_syns(int nstep);
static void update_total_syns(int nid);
static double get_total_syns_current(int nid, double vpost);
static void fprintf2d_d(FILE *fp, double x[MAX_TYPE][MAX_TYPE]);
static void fill_connection_prob(nn_info_t *info);


nn_info_t get_empty_info(void){
    nn_info_t info = {0,};
    for (int i=0; i<MAX_TYPE; i++){
        info.type_range[i] = 0;
        for (int j=0; j<MAX_TYPE; j++){
            info.mdeg_in[i][j] = -1;
            info.p_out[i][j] = -1;
            info.w[i][j] = 0;
        }

        info.taur[i] = taur_default;
        info.taud[i] = taud_default;
    }
    info.const_current = false;
    return info;
}


void build_ei_rk4(nn_info_t *info){
    int N = info->N;
    // set with default parameter ranges
    init_wbneuron(N, &neuron);
    init_extsyn(N, &ext_syn);
    for (int n=0; n<num_syn_types; n++){
        init_desyn(N, &syns[n]);
    }

    // set_attrib(&syns[0],   0, taur, taud, 0.5);
    // set_attrib(&syns[1], -80, taur, taud, 0.5);
    set_attrib(&syns[0], ev_e, info->taur[0], info->taud[0], 0.5);
    set_attrib(&syns[1], ev_i, info->taur[1], info->taud[1], 0.5);

    // generate network
    int e_range[2] = {0, N*0.8};
    int i_range[2] = {N*0.8, N};
    ntk_t ntk_e = get_empty_net(N);
    if (info->p_out[0][0] >= 0){
        gen_er_pout(&ntk_e, info->p_out[0][0], e_range, e_range);
        gen_er_pout(&ntk_e, info->p_out[0][1], e_range, i_range);
    } else if (info->mdeg_in[0][0] >= 0) {
        gen_er_mdin(&ntk_e, info->mdeg_in[0][0], e_range, e_range);
        gen_er_mdin(&ntk_e, info->mdeg_in[0][1], e_range, i_range);
    } else {
        printf("Set the connection info (line54)!\n"); exit(1);
    }
    set_network(&syns[0], &ntk_e);
    free_network(&ntk_e);

    ntk_t ntk_i = get_empty_net(N);
    if (info->p_out[1][0] >= 0){
        gen_er_pout(&ntk_i, info->p_out[1][0], i_range, e_range);
        gen_er_pout(&ntk_i, info->p_out[1][1], i_range, i_range);
    } else if (info->mdeg_in[0][0] >= 0) {
        gen_er_mdin(&ntk_i, info->mdeg_in[1][0], i_range, e_range);
        gen_er_mdin(&ntk_i, info->mdeg_in[1][1], i_range, i_range);
    } else {
        printf("Set the connection info (line67)!\n"); exit(1);
    }
    set_network(&syns[1], &ntk_i);
    free_network(&ntk_i);
    
    set_coupling(&syns[0], e_range, e_range, info->w[0][0]);
    set_coupling(&syns[0], e_range, i_range, info->w[0][1]);
    set_coupling(&syns[1], i_range, e_range, info->w[1][0]);
    set_coupling(&syns[1], i_range, i_range, info->w[1][1]);
    check_coupling(&syns[0]);
    check_coupling(&syns[1]);

    set_const_delay(&syns[0], info->t_lag);
    set_const_delay(&syns[1], info->t_lag);

    if (info->const_current){
        const_current = true;
    } else {
        set_attrib(&ext_syn, 0, taur_default, taud_default, 0.5);
        set_poisson(&ext_syn, info->nu_ext_mu, info->nu_ext_sd, info->w_ext_mu, info->w_ext_sd);
    }
}


void write_info(nn_info_t *info, char *fname){
    fill_connection_prob(info);

    FILE *fp = open_file(fname, "w");
    fprintf(fp, "Size: %d\n", info->N);
    fprintf(fp, "ntypes: %d\n", num_syn_types);
    fprintf(fp, "type_range:\n");
    fprintf(fp, "w:\n");
    fprintf2d_d(fp, info->w);
    
    fprintf(fp, "taur, taud\n");
    for (int n=0; n<num_syn_types; n++){
        fprintf(fp, "%f, %f\n", info->taur[n], info->taud[n]);
    }

    fprintf(fp, "t_lag: %f\n", info->t_lag);
    fprintf(fp, "nu_pos_mu: %f\n", info->nu_ext_mu);
    fprintf(fp, "nu_pos_sd: %f\n", info->nu_ext_sd);
    fprintf(fp, "w_pos_mu: %f\n", info->w_ext_mu);
    fprintf(fp, "w_pos_sd: %f\n", info->w_ext_sd);

    fprintf(fp, "mean indegree:\n");
    fprintf2d_d(fp, info->mdeg_in);
    fprintf(fp, "connection prob (out):\n");
    fprintf2d_d(fp, info->p_out);
    fclose(fp);
}


void update_rk4(int nstep, double iapp){
    int N = neuron.N;

    double *v_prev = (double*) malloc(sizeof(double) * N);

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

        neuron.vs[id] = v0 + (dv1 + 2*dv2 + 2*dv3 + dv4)/6.;
        neuron.hs[id] = h0 + (dh1 + 2*dh2 + 2*dh3 + dh4)/6.;
        neuron.ns[id] = n0 + (dn1 + 2*dn2 + 2*dn3 + dn4)/6.;
    }

    check_fire(&neuron, v_prev);
    add_spike_total_syns(nstep);
    free(v_prev);
}


static int stack = 0;
void write_all_vars(int nstep, FILE *fp){
    int N = neuron.N;
    int ncol = N;
    for (int n=0; n<num_syn_types; n++) ncol += N;
    if (!const_current) ncol += N;

    double *vars = (double*) malloc(sizeof(double) * ncol);
    int id=0;
    for (int n=0; n<N; n++){
        vars[id++] = neuron.vs[n];
    }
    for (int i=0; i<num_syn_types; i++){
        for (int n=0; n<N; n++){
            vars[id++] = syns[i].expr[n] - syns[i].expd[i];
        }
    }
    if (!const_current){
        for (int n=0; n<N; n++){
            vars[id++] = ext_syn.expr[n] - ext_syn.expd[n];
        }
    }

    if (id > ncol){
        printf("Index exceeds the expected length: %d/%d\n", id, ncol);
        exit(1);
    }
    
    if (stack == 0){
        stack = 1;
        float tmp[1] = {(float) ncol};
        fwrite(tmp, sizeof(float), 1, fp);
    }

    save(ncol, nstep, vars, fp);
    free(vars);
}


void destroy_neuralnet(void){
    destroy_wbneuron(&neuron);
    for (int n=0; n<num_syn_types; n++){
        destroy_desyn(&syns[n]);
    }
    if (!const_current) destroy_desyn(&ext_syn);
    #ifdef USE_MKL
    end_stream();
    #endif
}


static void add_spike_total_syns(int nstep){
    for (int n=0; n<num_syn_types; n++){
        add_spike(nstep, &syns[n], &neuron);
    }
    if (!const_current) add_ext_spike(&ext_syn);
}


static void update_total_syns(int nid){
    for (int n=0; n<num_syn_types; n++){
        update_desyn(&syns[n], nid);
    }
    if (!const_current) update_desyn(&ext_syn, nid);
}


static double get_total_syns_current(int nid, double vpost){
    double isyn = 0;
    if (!const_current) isyn = get_current(&ext_syn, nid, vpost);
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


static void fill_connection_prob(nn_info_t *info){
    int N = info->N;
    int num_cells[2] = {0.8*N, 0.2*N};
    if (num_syn_types != 2){ printf("Need to correct this part: line265"); exit(1); }

    if (info->p_out[0][0] >= 0){
        for (int i=0; i<num_syn_types; i++){
            for (int j=0; j<num_syn_types; j++){
                info->mdeg_in[i][j] = num_cells[i] * info->p_out[i][j]; 
            }
        }
    } else if (info->mdeg_in[0][0] >= 0){
        for (int i=0; i<num_syn_types; i++){
            for (int j=0; j<num_syn_types; j++){
                info->p_out[i][j] = info->mdeg_in[i][j] / num_cells[i] * num_cells[j];
            }
        }
    }
}
