#include "neuralnet.h"


const double ev_e = 0;
const double ev_i = -80;


static int num_cells = 0;
static int num_types = 2;
wbneuron_t neuron;
desyn_t syns[MAX_TYPE], ext_syn[MAX_TYPE];
double taur_default=0.3, taud_default=1;
int const_current = false;

// multiple types of external inputs
static int num_ext_types = -1;
int *ext_class = NULL;
int *ext_index = NULL;

double cell_ratio2[2] = {0.8, 0.2};
double cell_ratio3[3] = {4./5, 1./10, 1./10};
double cell_ratio4[4] = {2./5, 1./5, 2./5, 1./5};
int cell_range[MAX_TYPE][2] = {0,};


static void add_spike_total_syns(int nstep);
static void update_total_syns(int nid);
static double get_total_syns_current(int nid, double vpost);
static void fprintf2d_d(FILE *fp, double x[MAX_TYPE][MAX_TYPE]);
static void fill_connection_prob(nn_info_t *info);
static void set_cell_range(void);


nn_info_t init_build_info(int N, int _num_types){

    num_cells = N;
    num_types = _num_types;

    nn_info_t info = {0,};

    info.N = N;
    info.num_types = num_types;
    set_cell_range();

    for (int i=0; i<MAX_TYPE; i++){
        info.type_range[i] = cell_range[i][1];
        for (int j=0; j<MAX_TYPE; j++){
            info.mdeg_in[i][j] = -1;
            info.p_out[i][j] = -1;
            info.w[i][j] = 0;
        }
        info.nu_ext_multi[i] = 0;
        info.w_ext_multi[i] = 0;
        
        info.taur[i] = taur_default;
        info.taud[i] = taud_default;
    }
    info.const_current = false;
    info.num_ext_types = 1;

    return info;
}


void build_ei_rk4(nn_info_t *info){

    // verify the information

    // generate empty synapse (+ external poisson input)
    init_wbneuron(num_cells, &neuron);
    for (int n=0; n<num_types; n++){
        init_desyn(num_cells, &syns[n]);
    }

    num_ext_types = info->num_ext_types;
    ext_class = (int*) calloc(num_cells, sizeof(int));
    ext_index = (int*) calloc(num_cells, sizeof(int));
    if (num_ext_types == 1){
        init_extsyn(num_cells, &(ext_syn[0]));
        for (int n=0; n<num_cells; n++) ext_index[n] = n;
    } else if (num_ext_types > 1){
        for (int n=0; n<info->N; n++){
            ext_class[n] = -1;
            ext_index[n] = -1;
        }

    } else {
        printf("Typed wrong number of poisson inputs: %d\n", info->num_ext_types);
        exit(1);
    }

    double ev_set[MAX_TYPE] = {ev_e, ev_i, ev_i, ev_i};
    if (num_types == 1){
        ev_set[0] = ev_i;
    } else if (num_types == 2){
        ev_set[0] = ev_e; 
        ev_set[2] = ev_e; 
    } else if (num_types == 4){
        ev_set[2] = ev_e;
    } else {
        printf("num_types (%d) exceeds expected (neuralnet.c: build_ei_rk4)\n", num_types);
        exit(1);
    }

    for (int i=0; i<num_types; i++){
        set_attrib(&syns[i], ev_set[i], info->taur[i], info->taud[i], 0.5);
    }

    // set the cell type range
    set_cell_range();
    
    // check input type & generate network
    int flag_ntk = -1;
    if (info->p_out[0][0] >= 0) flag_ntk = 0;
    else if (info->mdeg_in[0][0] >= 0) flag_ntk = 1;

    if (flag_ntk == -1){
        printf("Set the connection info (neuralnet.c)!\n"); exit(1);
    }

    for (int n=0; n<num_types; n++){
        ntk_t ntk = get_empty_net(num_cells);
        for (int i=0; i<num_types; i++){
            if (flag_ntk == 0){
                gen_er_pout(&ntk, info->p_out[n][i], cell_range[n], cell_range[i]);
            } else {
                gen_er_mdin(&ntk, info->mdeg_in[n][i], cell_range[n], cell_range[i]);
            }
        }
        set_network(&syns[n], &ntk);
        free_network(&ntk);
    }

    // set weight of the network
    for (int n=0; n<num_types; n++){
        for (int i=0; i<num_types; i++){
            set_coupling(&syns[n], cell_range[n], cell_range[i], info->w[n][i]);
        }
        check_coupling(&syns[n]);
    }

    // set time delay of the network
    for (int n=0; n<num_types; n++){
        set_const_delay(&syns[n], info->t_lag);
    }

    if (info->const_current){
        const_current = true;
    } else {
        // NOTE: set poisson neuron for each cell types
        if (num_ext_types == 1){
            set_attrib(&(ext_syn[0]), 0, taur_default, taud_default, 0.5);
            set_poisson(&(ext_syn[0]), info->nu_ext_mu, info->nu_ext_sd, info->w_ext_mu, info->w_ext_sd);
        }
    }
}


void set_multiple_ext_input(nn_info_t *info, int type_id, int num_targets, int *target_id){
    if (type_id >= num_ext_types){
        printf("type id (%d) exceed expected maximal id (%d) (neuralnet.c: set_multiple_ext_input)\n", type_id, num_ext_types-1);
        exit(1);
    }

    double nu_ext_sd = 0, w_ext_sd = 0;

    for (int n=0; n<num_targets; n++){
        int nid = target_id[n];
        if (nid >= num_cells){
            printf("nid (%d) exceeds the network size\n", nid);
            exit(1);
        }

        ext_class[nid] = type_id;
        ext_index[nid] = n;
    }

    init_extsyn(num_targets, &(ext_syn[type_id]));
    set_attrib(&(ext_syn[type_id]), 0, taur_default, taud_default, 0.5);
    set_poisson(&(ext_syn[type_id]), info->nu_ext_multi[type_id], nu_ext_sd, info->w_ext_multi[type_id], w_ext_sd);
}


void check_multiple_input(){
    if (num_ext_types > 1){
        for (int n=0; n<num_cells; n++){
            if (ext_class[n] == -1){
                printf("Allocate the external input (neuralnet.c: check_multiple_input)\n");
                exit(1);
            }
        }
    }
}


static void set_cell_range(void){

    if (num_types == 1){
        cell_range[0][0] = 0;
        cell_range[0][1] = num_cells;
    }else if (num_types == 2){
        cell_range[0][0] = 0;
        cell_range[0][1] = num_cells * cell_ratio2[0];
        cell_range[1][0] = num_cells * cell_ratio2[0];
        cell_range[1][1] = num_cells;
    } else if (num_types == 3){
        cell_range[0][0] = 0;
        cell_range[0][1] = num_cells * cell_ratio3[0];
        cell_range[1][0] = num_cells * cell_ratio3[0];
        cell_range[1][1] = cell_range[1][0] + num_cells * cell_ratio3[1];
        cell_range[2][0] = cell_range[1][1];
        cell_range[2][1] = cell_range[1][1] + num_cells * cell_ratio3[1];
    } else if (num_types == 4){
        int tmp_num_types[4] = {num_cells*0.4, num_cells*0.1, num_cells*0.4, num_cells*0.1};
        int ncum = 0;
        for (int n=0; n<4; n++){
            cell_range[n][0] = ncum;
            cell_range[n][1] = ncum + tmp_num_types[n];
            ncum += tmp_num_types[n];
        }
    } else {
        printf("not expected number of types (neuralnet.c: set_cell_range)\n");
        exit(1);
    }

    // check the number (this line is for num_types == 3)
    int num_set = 0;
    for (int n=0; n<num_types; n++){
        num_set += cell_range[n][1] - cell_range[n][0];
    }

    if (num_cells > num_set){
        if (num_types != 3){
            printf("Unexpected case, check the input cell number (neuralnet.c: set_cell_range)\n");\
            exit(1);
        }

        int dn = num_cells - num_set;
        int mod = dn % num_types;

        if (mod == 2){
            cell_range[1][1] += 1;
            cell_range[2][0] = cell_range[1][1];
            cell_range[2][1] += 2;
        } else {
            printf("mod = %d, check the input cell number (neuralnet.c: set_cell_range)\n", mod);
            exit(1);
        }
    } else if (num_cells < num_set) {
        printf("num_set exceeds num_cells, check the input cell number (neuralnet.c: set_cell_range)\n");
        exit(1);
    }
}


void write_info(nn_info_t *info, char *fname){
    fill_connection_prob(info);

    FILE *fp = open_file(fname, "w");
    fprintf(fp, "Size: %d\n", info->N);
    fprintf(fp, "ntypes: %d\n", num_types);
    fprintf(fp, "type_range: ");
    for (int n=0; n<num_types; n++){
        fprintf(fp, "%d, ", info->type_range[n]);
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "w:\n");
    fprintf2d_d(fp, info->w);
    
    fprintf(fp, "taur, taud:\n");
    for (int n=0; n<num_types; n++){
        fprintf(fp, "%f, %f\n", info->taur[n], info->taud[n]);
    }

    fprintf(fp, "mean indegree:\n");
    fprintf2d_d(fp, info->mdeg_in);
    fprintf(fp, "connection prob (out):\n");
    fprintf2d_d(fp, info->p_out);

    fprintf(fp, "t_lag: %f\n", info->t_lag);

    if (num_ext_types == 1){
        fprintf(fp, "nu_pos_mu: %f\n", info->nu_ext_mu);
        fprintf(fp, "nu_pos_sd: %f\n", info->nu_ext_sd);
        fprintf(fp, "w_pos_mu: %f\n", info->w_ext_mu);
        fprintf(fp, "w_pos_sd: %f\n", info->w_ext_sd);
    } else {
        for (int n=0; n<num_ext_types; n++){
            fprintf(fp, "poisson type %d (sd=0)\n", n);
            fprintf(fp, "nu_pos_mu: %f\n", info->nu_ext_multi[n]);
            fprintf(fp, "w_pos_mu: %f\n", info->w_ext_multi[n]);
        }
    }

    fclose(fp);
}


extern int flag_nan; // monitor nan value

void update_rk4(int nstep, double iapp){

    double *v_prev = (double*) malloc(sizeof(double) * num_cells);

    for (int id=0; id<num_cells; id++){
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

    if (flag_nan == 1){
        printf("Nan deteced in simulation. Terminate the process\n");
        exit(1);
    }
}


static int stack = 0;
void write_all_vars(int nstep, FILE *fp){
    int ncol = num_cells;
    for (int n=0; n<num_types; n++) ncol += num_cells;
    if (!const_current) ncol += num_cells;
    if (num_ext_types > 1){
        printf("This function is not devloped for multiple excitation sources\n");
        return;
    }

    double *vars = (double*) malloc(sizeof(double) * ncol);
    int id=0;
    for (int n=0; n<num_cells; n++){
        vars[id++] = neuron.vs[n];
    }
    for (int i=0; i<num_types; i++){
        for (int n=0; n<num_cells; n++){
            vars[id++] = syns[i].expr[n] - syns[i].expd[i];
        }
    }
    if (!const_current){
        for (int n=0; n<num_cells; n++){
            vars[id++] = ext_syn[0].expr[n] - ext_syn[0].expd[n];
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
    for (int n=0; n<num_types; n++){
        destroy_desyn(&syns[n]);
    }
    // free poisson synaptic input
    if (!const_current){
        for (int n=0; n<num_ext_types; n++){
            destroy_desyn(&(ext_syn[n]));
        }
        free(ext_class);
        free(ext_index);
    }

    #ifdef USE_MKL
    end_stream();
    #endif
}


static void add_spike_total_syns(int nstep){
    for (int n=0; n<num_types; n++){
        add_spike(nstep, &syns[n], &neuron);
    }

    if (!const_current){
        for (int n=0; n<num_ext_types; n++){
            add_ext_spike(&(ext_syn[n]));
        }
    }
}


static void update_total_syns(int nid){
    for (int n=0; n<num_types; n++){
        update_desyn(&syns[n], nid);
    }
    
    if (!const_current){
        int ntype = ext_class[nid];
        int id = ext_index[nid];
        update_desyn(&(ext_syn[ntype]), id);
    }
}


static double get_total_syns_current(int nid, double vpost){
    double isyn = 0;
    for (int n=0; n<num_types; n++){
        isyn += get_current(&syns[n], nid, vpost);
    }

    if (!const_current){
        int ntype = ext_class[nid];
        int id = ext_index[nid];
        isyn += get_current(&(ext_syn[ntype]), id, vpost);
    }

    return isyn;
}


static void fprintf2d_d(FILE *fp, double x[MAX_TYPE][MAX_TYPE]){
    for (int i=0; i<num_types; i++){
        for (int j=0; j<num_types; j++){
            fprintf(fp, "%f, ", x[i][j]);
        }
        fprintf(fp, "\n");
    }
}


static void fill_connection_prob(nn_info_t *info){

    if (info->p_out[0][0] >= 0){
        for (int i=0; i<num_types; i++){
            int n = cell_range[i][1] - cell_range[i][0];
            for (int j=0; j<num_types; j++){
                info->mdeg_in[i][j] = n * info->p_out[i][j]; 
            }
        }
    } else if (info->mdeg_in[0][0] >= 0){
        for (int i=0; i<num_types; i++){
            int ni = cell_range[i][1] - cell_range[i][0];
            for (int j=0; j<num_types; j++){
                int nj = cell_range[i][1] - cell_range[i][0];
                info->p_out[i][j] = info->mdeg_in[i][j] / ni * nj;
            }
        }
    }
}
