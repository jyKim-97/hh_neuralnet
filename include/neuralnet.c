#include "neuralnet.h"


const double ev_e = 0;
const double ev_i = -80;

static int num_cells = 0;
static int num_types = 2;
static int num_pcells = 0;

nnpop_t nnpop; // nnpop references whole network providing proxy in the other source codes
wbneuron_t neuron;
desyn_t syns[MAX_TYPE] = {0,};
desyn_t ext_syn[MAX_TYPE] = {0,};

// Poisson input neuorn
pneuron_t pneuron;
desyn_t psyn;

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

// network construction
static void init_background_input(nn_info_t *info);
static void init_poisson_input(nn_info_t *info);
static void set_cell_range(nn_info_t *info);


static void add_spike_total_syns(int nstep);
static void update_total_syns(int nid);
static double get_total_syns_current(int nid, double vpost);
static void fprintf2d_d(FILE *fp, int num, double x[MAX_TYPE][MAX_TYPE]);
static void fill_connection_prob(nn_info_t *info);
static void check_info(nn_info_t *info);

#define REPORT_ERROR(msg) print_error(msg, __FILE__, __LINE__)


// void init_nn(int N, int _num_types){
//     if (N <= 0){
//         char msg[300];
//         sprintf(msg, "Invalid N: %d", N);
//         REPORT_ERROR(msg);
//     }

//     num_cells = N;
//     num_types = _num_types;
//     num_pcells = 0;
    
//     for (int n=0; n<MAX_TYPE; n++){
//         cell_range[n][0] = 0;
//         cell_range[n][1] = 0;
//     }
// }


// nn_info_t init_build_info(int N, int _num_types){
nn_info_t init_build_info(){

    nn_info_t info = {0,};

    info.N = 0;
    info.num_types = 0;
    info.pN = 0;

    for (int i=0; i<MAX_TYPE; i++){
        info.type_range[i] = cell_range[i][1];
        info.type_id[i] = -1;

        for (int j=0; j<MAX_TYPE; j++){
            // network
            info.mdeg_in[i][j] = NULL_CON;
            info.p_out[i][j]   = NULL_CON;
            info.w[i][j]       = NULL_CON;
        }
        // info.nu_ext_multi[i] = 0;
        // info.w_ext_multi[i] = 0;
        info.nu_ext_mu[i] = 0;
        info.w_ext_mu[i]  = 0;
        
        info.taur[i] = taur_default;
        info.taud[i] = taud_default;

        // poisson neuron connection
        info.pp_out[i] = NULL_CON;
        info.pw[i]     = NULL_CON;
        info.prange[i][0] = -1;
        info.prange[i][1] = -1;
    }
    
    info.nu_ext_sd = 0;
    info.w_ext_sd = 0;

    info.const_current = false;
    info.num_ext_types = 1;

    info.pfr = 0;
    info.ptaur = taur_default;
    info.ptaur = taud_default;

    return info;
}


void set_type_info(nn_info_t *info, int num_types, int type_id[], int type_range[]){
    info->num_types = num_types;
    for (int n=0; n<num_types; n++){
        info->type_id[n] = type_id[n];
        info->type_range[n] = type_range[n];
    }
}

// void set_pneuron_input(nn_info_t *info, int NP, double taur, double taud){
//     info->NP = NP;
//     num_pcells = NP;
//     info->ptaur = taur;
//     info->ptaud = taud;
// }


static void build_desyn(nn_info_t *info){

    // initialize synapse
    for (int n=0; n<num_types; n++){
        init_desyn(num_cells, &syns[n]);
    }

    // printf("ntype: %d\n", num_types);
    // for (int n=0; n<MAX_TYPE; n++){
    //     printf("%d,", info->type_id[n]);
    // }
    // printf("\n");

    // set equilibrium potential
    double ev_set[MAX_TYPE] = {0,};
    if (info->type_id[0] != -1){
        double ev_order[3] = {ev_e, ev_i, ev_i};
        for (int n=0; n<MAX_TYPE; n++){
            int nid = info->type_id[n];
            if (nid == -1) break;
            if (nid > 2){
                char msg[300];
                sprintf(msg, "Invalid type ID: %d", nid);
                REPORT_ERROR(msg);
            }
            ev_set[n] = ev_order[nid];
        }
    } else { // automatically allocate
        if (num_types == 1){
            ev_set[0] = ev_i; 
        } else if (num_types == 2){
            ev_set[0] = ev_e; ev_set[1] = ev_i;
        } else if (num_types == 4){
            ev_set[0] = ev_e; ev_set[1] = ev_i;
            ev_set[2] = ev_e; ev_set[3] = ev_i;
        } else {
            char msg[300];
            sprintf(msg, "Invalid num_types (%d) without specified type", num_types);
            REPORT_ERROR(msg);
        }
    }

    // set attribute of synapse
    for (int i=0; i<num_types; i++){
        set_attrib(&syns[i], ev_set[i], info->taur[i], info->taud[i], 0.5);
    }


    // connect network
    int flag_ntk = -1;
    if (info->p_out[0][0] != NULL_CON){
        flag_ntk = 0;
    } else if (info->mdeg_in[0][0] != NULL_CON){
        flag_ntk = 1;
    } else REPORT_ERROR("Set the connection info");

    for (int n=0; n<num_types; n++){
        ntk_t ntk = get_empty_net(num_cells); // in- network
        for (int i=0; i<num_types; i++){
            if (flag_ntk == 0){ // n: pre, i: post
                gen_er_pout(&ntk, info->p_out[n][i], cell_range[n], cell_range[i]);
            } else { 
                gen_er_mdin(&ntk, info->mdeg_in[n][i], cell_range[n], cell_range[i]);
            }
        }

        set_network(&syns[n], &ntk); // n: pre
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
}


static void init_background_input(nn_info_t *info){

    if (info->const_current){
        if (num_ext_types != 0) REPORT_ERROR("constant current will be given, but the # ext types is not 0");
        const_current = true;
        return;
    }
    
    if (num_ext_types <= 0) REPORT_ERROR("Invalid number of ext types");
    
    ext_class = (int*) calloc(num_cells, sizeof(int));
    ext_index = (int*) calloc(num_cells, sizeof(int));

    if (num_ext_types == 1){
        if (info->nu_ext_mu[1] != 0) REPORT_ERROR("The # of ext types does not match to # of parameters");

        init_extsyn(num_cells, &(ext_syn[0]));
        for (int n=0; n<num_cells; n++) ext_index[n] = n;

        set_attrib(&(ext_syn[0]), ev_e, taur_default, taud_default, 0.5);
        set_poisson(&(ext_syn[0]), info->nu_ext_mu[0], info->nu_ext_sd, info->w_ext_mu[0], info->w_ext_sd);
    } else {
        for (int n=0; n<num_cells; n++){
            ext_class[n] = -1;
            ext_index[n] = -1;
        }
    }
}


static void init_poisson_input(nn_info_t *info){
    if (num_pcells == 0) return;

    init_pneuron(num_pcells, &pneuron);
    set_pneuron_attrib(&pneuron, info->pfr);

    init_desyn(num_cells, &psyn);
    set_attrib(&psyn, 0, info->ptaur, info->ptaud, 0.5);

    // connect network - HARD FIX to one-to-one connection
    int npost = 0;
    for (int n=0; n<num_types; n++){
        if (info->prange[n][0] == -1) break;
        npost += info->prange[n][1] - info->prange[n][0];
    }
    if (info->pN != npost) REPORT_ERROR("Size does not match");

    int nstack = 0;
    ntk_t pntk = get_empty_net(num_cells);
    for (int n=0; n<num_types; n++){
        int num = info->prange[n][1] - info->prange[n][0];
        int pre_range[2] = {nstack, nstack+num};
        if (info->prange[n][0] == -1) break;

        // printf("%d:%d -> %d:%d\n", pre_range[0],ntk pre_range[1], info->prange[n][0], info->prange[n][1]);

        // gen_er_pout(&pntk, info->pp_out[n], pre_range, info->prange[n]);
        gen_er_pout(&pntk, ONE2ONE, pre_range, info->prange[n]);
        nstack += num;
    }
    set_network(&psyn, &pntk);
    free_network(&pntk);

    nstack = 0;
    for (int n=0; n<num_types; n++){
        int num = info->prange[n][1] - info->prange[n][0];
        int pre_range[2] = {nstack, nstack+num};
        if (info->prange[n][0] == -1) break;
        set_coupling(&psyn, pre_range, info->prange[n], info->pw[n]);
        nstack += num;
    }

    check_coupling(&psyn);
    set_const_delay(&psyn, 0);
}


void check_building(){

    // external
    for (int n=0; n<num_cells; n++){
        // printf("%d: %d\n", n, ext_class[n]);
        if (ext_class[n] == -1){
            REPORT_ERROR("Configure background Poisson synaptic input");
        }
    }
}


static void check_info(nn_info_t *info){
    if (info->N == 0){
        REPORT_ERROR("The number of neuron is 0");
    }

    if (info->num_types == 0){
        REPORT_ERROR("The number of neuron types are not defined");
    }

    if ((info->p_out[0][0] == NULL_CON) && (info->mdeg_in[0][0] == NULL_CON)){
        REPORT_ERROR("The connection info is blank");
    }

    if (info->type_range[0] == -1){
        REPORT_ERROR("Set the number each type neuron (type_range)");
    }

    if (info->type_id[0] == -1){
        REPORT_ERROR("Type ID is not defined");
    }
}


void build_ei_rk4(nn_info_t *info){

    check_info(info);

    // set proxy
    nnpop.N = info->N;
    nnpop.num_types = info->num_types;
    nnpop.neuron = &neuron;
    nnpop.syns = syns;
    nnpop.ext_syn = ext_syn;

    nnpop.pneuron = &pneuron;
    nnpop.psyn = &psyn;

    // set global variables
    num_cells = info->N;
    num_pcells = info->pN; // Poisson neuron
    if (num_cells == 0) REPORT_ERROR("The number of neurons is 0");

    num_types = info->num_types;
    num_ext_types = info->num_ext_types;
    set_cell_range(info); // check type_range

    // generate empty synapse (+ external poisson input)
    init_wbneuron(num_cells, &neuron); // initialize neuron
    build_desyn(info); // set synaptic network
    init_background_input(info); // set external input
    init_poisson_input(info); 

}


void set_multiple_ext_input(nn_info_t *info, int type_id, int num_targets, int *target_id){
    if (type_id >= num_ext_types){
        char msg[100];
        sprintf(msg, "type id (%d) exceed expected maximal id (%d)\n", type_id, num_ext_types-1);
        REPORT_ERROR(msg);
    }

    // double nu_ext_sd = 0, w_ext_sd = 0;

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
    // set_poisson(&(ext_syn[type_id]), info->nu_ext_multi[type_id], nu_ext_sd, info->w_ext_multi[type_id], w_ext_sd);
    set_poisson(&(ext_syn[type_id]), info->nu_ext_mu[type_id], info->nu_ext_sd, info->w_ext_mu[type_id], info->w_ext_sd);
}


// void check_multiple_input(){
//     if (num_ext_types > 1){
//         for (int n=0; n<num_cells; n++){
//             if (ext_class[n] == -1){
//                 REPORT_ERROR("Allocate the external input");
//                 // printf("Allocate the external input (neuralnet.c: check_multiple_input)\n");
//                 // exit(1);
//             }
//         }
//     }
// }


static void set_cell_range(nn_info_t *info){

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
        if (info->type_range[0] == 0){
            char msg[300] = "The number of each cell type is not determined";
            REPORT_ERROR(msg);
        }
        
        int ncum = 0;
        for (int n=0; n<num_types; n++){
            cell_range[n][0] = ncum;
            cell_range[n][1] = ncum + info->type_range[n];
            ncum += info->type_range[n];
        }
    }

    // check the number (this line is for num_types == 3)
    int num_set = 0;
    for (int n=0; n<num_types; n++){
        num_set += cell_range[n][1] - cell_range[n][0];
    }

    if (num_cells > num_set){
        if (num_types != 3) REPORT_ERROR("The number of cells and sum of type range does not match");

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

    // printf("cell_range\n");
    // for (int n=0; n<num_types; n++){
    //     printf("[%4d, %4d]\n", cell_range[n][0], cell_range[n][1]);
    // }
    // printf("\n");
}


void write_info(nn_info_t *info, char *fname){
    // :NOTE: NEED TO ADD Poisson info

    fill_connection_prob(info);

    // printf("p_out:\n");
    // for (int i=0; i<MAX_TYPE; i++){
    //     for (int j=0; j<MAX_TYPE; j++){
    //         printf("%8.3f", info->p_out[i][j]);
    //     }
    //     printf("\n");
    // }

    // printf("mdeg_in:\n");
    // for (int i=0; i<MAX_TYPE; i++){
    //     for (int j=0; j<MAX_TYPE; j++){
    //         printf("%8.3f", info->mdeg_in[i][j]);
    //     }
    //     printf("\n");
    // }

    FILE *fp = open_file(fname, "w");
    fprintf(fp, "Size: %d\n", info->N);
    fprintf(fp, "ntypes: %d\n", info->num_types);
    fprintf(fp, "type_range: ");
    for (int n=0; n<info->num_types; n++){
        fprintf(fp, "%d(%d), ", info->type_range[n], info->type_id[n]);
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "w:\n");
    fprintf2d_d(fp, info->num_types, info->w);
    
    fprintf(fp, "taur, taud:\n");
    for (int n=0; n<info->num_types; n++){
        fprintf(fp, "%f, %f\n", info->taur[n], info->taud[n]);
    }

    fprintf(fp, "mean indegree:\n");
    fprintf2d_d(fp, info->num_types, info->mdeg_in);
    fprintf(fp, "connection prob (out):\n");
    fprintf2d_d(fp, info->num_types, info->p_out);

    fprintf(fp, "t_lag: %f\n", info->t_lag);

    for (int n=0; n<info->num_ext_types; n++){
        fprintf(fp, "poisson type %d (sd=0)\n", n);
        fprintf(fp, "nu_pos_mu: %f\n", info->nu_ext_mu[n]);
        fprintf(fp, "w_pos_mu: %f\n", info->w_ext_mu[n]);
    }
    fprintf(fp, "nu_pos_sd: %f\n", info->nu_ext_sd);
    fprintf(fp, "w_pos_sd: %f\n", info->w_ext_sd);

    // if (info->num_ext_types == 1){
    //     // fprintf(fp, "nu_pos_mu: %f\n", info->nu_ext_mu);
    //     // fprintf(fp, "nu_pos_sd: %f\n", info->nu_ext_sd);
    //     fprintf(fp, "w_pos_mu: %f\n", info->w_ext_mu);
    //     fprintf(fp, "w_pos_sd: %f\n", info->w_ext_sd);
    // } else {
    //     for (int n=0; n<info->num_ext_types; n++){
    //         fprintf(fp, "poisson type %d (sd=0)\n", n);
    //         fprintf(fp, "nu_pos_mu: %f\n", info->nu_ext_mu[n]);
    //         fprintf(fp, "w_pos_mu: %f\n", info->w_ext_mu[n]);
    //     }
    //     fprintf(fp, "w_pos_mu: %f\n", info->w_ext_mu);
    //     fprintf(fp, "w_pos_sd: %f\n", info->w_ext_sd);
    // }

    fprintf(fp, "time_step: %f\n", _dt);

    fprintf(fp, "Poisson_inputs: %d\n", info->pN);
    fprintf(fp, "Target range:\n");
    for (int n=0; n<info->num_types; n++){
        if (info->prange[n][0] == -1) break;
        fprintf(fp, "[%d, %d]\n", info->prange[n][0], info->prange[n][1]);
    }
    fprintf(fp, "connection prob(out):\n");
    for (int n=0; n<info->num_types; n++){
        if (info->pw[n] == NULL_CON) break;
        fprintf(fp, "%f\n", info->pp_out[n]);
    }
    fprintf(fp, "w:\n");
    for (int n=0; n<info->num_types; n++){
        if (info->pw[n] == NULL_CON) break;
        fprintf(fp, "%f\n", info->pw[n]);
    }
    fprintf(fp, "ptaur, ptaud:\n");
    fprintf(fp, "%f, %f\n", info->ptaur, info->ptaud);

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

    if (flag_nan == 1) REPORT_ERROR("Nan deteced in simulation. Terminate the process");
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
        char msg[100];
        sprintf(msg, "Index exceeds the expected length: %d/%d\n", id, ncol);
        REPORT_ERROR(msg);
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

    if (num_pcells > 0) add_pneuron_spike(&psyn, &pneuron);

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

    if (num_pcells > 0){
        update_desyn(&psyn, nid);
    }
    
    if (!const_current){
        int ntype = ext_class[nid];
        int id = ext_index[nid];
        update_desyn(&(ext_syn[ntype]), id);
    }
}


static double get_total_syns_current(int nid, double vpost){
    /*Update total synaptic current*/
    double isyn = 0;
    for (int n=0; n<num_types; n++){
        isyn += get_current(&syns[n], nid, vpost);
    }

    if (num_pcells > 0){
        // double itmp = get_current(&psyn, nid, vpost);
        isyn += get_current(&psyn, nid, vpost);
        // isyn += itmp;

        // printf("nid: %d, vpost: %f, isyn: %f, (%f, %f)\n", nid, vpost, itmp, psyn.expr[nid], psyn.expd[nid]);
    }

    if (!const_current){
        int ntype = ext_class[nid];
        int id = ext_index[nid];
        isyn += get_current(&(ext_syn[ntype]), id, vpost);
    }

    return isyn;
}


static void fprintf2d_d(FILE *fp, int num, double x[MAX_TYPE][MAX_TYPE]){
    for (int i=0; i<num; i++){
        for (int j=0; j<num; j++){
            fprintf(fp, "%f, ", x[i][j]);
        }
        fprintf(fp, "\n");
    }
}


static void fill_connection_prob(nn_info_t *info){
    
    if (info->p_out[0][0] != NULL_CON){
        for (int i=0; i<MAX_TYPE; i++){
            // int num = cell_range[i][1] - cell_range[i][0];
            int num = info->type_range[i];
            for (int j=0; j<MAX_TYPE; j++){
                if (info->p_out[i][j] == NULL_CON){
                    break;
                }

                if (info->p_out[i][j] == ONE2ONE){ 
                    info->mdeg_in[i][j] = ONE2ONE;
                } else {
                    info->mdeg_in[i][j] = num * info->p_out[i][j]; 
                }
            }
        }
    } else if (info->mdeg_in[0][0] != NULL_CON){ // TODO: NEED TO CHECK OUT AND IN
        for (int i=0; i<MAX_TYPE; i++){
            // int ni = cell_range[i][1] - cell_range[i][0];
            int ni = info->type_range[i];
            for (int j=0; j<MAX_TYPE; j++){
                if (info->mdeg_in[i][j] == NULL_CON){
                    break;
                }
                
                if (info->mdeg_in[i][j] == ONE2ONE){
                    info->p_out[i][j] = ONE2ONE;
                } else {
                    int nj = info->type_range[j];
                    info->p_out[i][j] = info->mdeg_in[i][j] / ni * nj;
                }
            }
        }
    }
}
