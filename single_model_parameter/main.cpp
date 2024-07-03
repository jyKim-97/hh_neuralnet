#include <iostream>
#include <cmath>
#include <mpi.h>
#include <utils.h>


// #define DEBUG
#define THRESHOLD 0
double _dt = 0.005;
const int num_var = 5;

// g++ -std=c++11 -O3 -o main.out main.cpp
// mpic++ -std=c++11 -I../include -O3 -o main.out main.cpp ../include/utils.c


class wbNeuron {
    public:
        double v=-70, h_ion=0, n_ion=1;
        bool spike;
        double t_eq=500;
        double iapp=0;
        double cm=1, gna=35, gk=9, gl=0.1;
        const double phi=5, el=-65, ena=55, ek=-90; // mV

    double measure_fr(double tmax);
    void run2eq(void);
    void update(void);
    double get_minf(void);
    double solve_v(double v, double h_ion, double n_ion);
    double solve_h(double v, double h_ion, double n_ion);
    double solve_n(double v, double h_ion, double n_ion);
    // double ext_noise(void);
};

double *linspace(double x0, double x1, int len_x);
void print_arr(FILE *fp, int N, const double *arr);


#ifdef DEBUG
int main(){
    wbNeuron cell;
    cell.iapp = 5;
    double fr = cell.measure_fr(1000);
    std::cout << "firing rate: " << fr <<"Hz" << std::endl;
}

#else

int main(int argc, char **argv){
    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // set parameter range

    index_t idxer;
    int max_len[num_var] = {21, 21, 21, 21, 20};
    set_index_obj(&idxer, num_var, max_len);
    int len = idxer.len;

    double *cm_set  = linspace(  1.,  5., max_len[0]);
    double *gl_set  = linspace(0.01,  1., max_len[1]);
    double *gk_set  = linspace(  1., 30., max_len[2]);
    double *gna_set = linspace( 40., 80., max_len[3]);
    double *ic_set  = linspace(  0.,  2, max_len[4]);

    // save parameter
    if (world_rank == 0){
        FILE *fp = fopen("./params.txt", "w");
        fprintf(fp, "cm:");  print_arr(fp, max_len[0], cm_set);
        fprintf(fp, "gl:");  print_arr(fp, max_len[1], gl_set);
        fprintf(fp, "gk:");  print_arr(fp, max_len[2], gk_set);
        fprintf(fp, "gna:"); print_arr(fp, max_len[3], gna_set);
        fprintf(fp, "ic:");  print_arr(fp, max_len[4], ic_set);
        fclose(fp);
    }

    double *fr_save = new double[len];

    double tmax = 5000;
    for (int n=world_rank; n<len; n+=world_size){
        
        update_index(&idxer, n);
        wbNeuron cell;

        cell.cm   = cm_set[idxer.id[0]];
        cell.gl   = gl_set[idxer.id[1]];
        cell.gk   = gk_set[idxer.id[2]];
        cell.gna  = gna_set[idxer.id[3]];
        cell.iapp = ic_set[idxer.id[4]];

        double fr = cell.measure_fr(tmax);
        fr_save[n] = fr;

        printf("Job %8d/%8d Done\n", n, len);
    }

    printf("Rank %d Done\n", world_rank);

    // Finalize
    if (world_rank == 0){
        MPI_Status status;
        for (int rank=1; rank<world_size; rank++){
            double *fr_tmp = new double[len];
            MPI_Recv(fr_tmp, len, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &status);

            for (int i=rank; i<len; i+=world_size){
                fr_save[i] = fr_tmp[i];
            }
            delete[] fr_tmp;

            printf("Recv from Node %4d\n", rank);
        }

        // save after 
        FILE *fp = fopen("./firing_rate_mpi.txt", "w");
        printf("len: %d\n", len);
        for (int n=0; n<len; n++){

            update_index(&idxer, n);
            for (int i=0; i<num_var-1; i++){
                fprintf(fp, "%d,", idxer.id[i]);
            }
            fprintf(fp, "%d:%lf\n", idxer.id[num_var-1], fr_save[n]);
        }
        fclose(fp);

    } else {
        MPI_Send(fr_save, len, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    delete[] fr_save;
    MPI_Finalize();
}


#endif



double wbNeuron::measure_fr(double tmax){
    int nmax = tmax / _dt;
    double fr = 0;

    #ifdef DEBUG
    FILE *fp = fopen("./check_v.txt", "w");
    #endif

    run2eq();
    for (int n=0; n<nmax; n++){
        update();
        if (this->spike == 1){ fr += 1; }

        #ifdef DEBUG
        fprintf(fp, "%f,", (float) this->v);
        #endif
    }

    #ifdef DEBUG
    fclose(fp);
    #endif

    return fr / tmax * 1000;
}


void wbNeuron::run2eq(void){
    int nmax = this->t_eq / _dt;
    for (int n=0; n<nmax; n++){
        update();
    }
}


void wbNeuron::update(void){
    // update with rk4 method
    double v0 = this->v;
    double h0 = this->h_ion;
    double n0 = this->n_ion;

    // rk4 - 1
    double dv1 = solve_v(v0, h0, n0);
    double dh1 = solve_h(v0, h0, n0);
    double dn1 = solve_n(v0, h0, n0);
    
    // rk4 - 2
    double dv2 = solve_v(v0+dv1*0.5, h0+dh1*0.5, n0+dn1*0.5);
    double dh2 = solve_h(v0+dv1*0.5, h0+dh1*0.5, n0+dn1*0.5);
    double dn2 = solve_n(v0+dv1*0.5, h0+dh1*0.5, n0+dn1*0.5);

    // rk4 - 3
    double dv3 = solve_v(v0+dv2*0.5, h0+dh2*0.5, n0+dn2*0.5);
    double dh3 = solve_h(v0+dv2*0.5, h0+dh2*0.5, n0+dn2*0.5);
    double dn3 = solve_n(v0+dv2*0.5, h0+dh2*0.5, n0+dn2*0.5);

    // rk4 - 4
    double dv4 = solve_v(v0+dv3*0.5, h0+dh3*0.5, n0+dn3*0.5);
    double dh4 = solve_h(v0+dv3*0.5, h0+dh3*0.5, n0+dn3*0.5);
    double dn4 = solve_n(v0+dv3*0.5, h0+dh3*0.5, n0+dn3*0.5);

    this->v     += (dv1 + 2*dv2 + 2*dv3 + dv4)/6.;
    this->h_ion += (dh1 + 2*dh2 + 2*dh3 + dh4)/6.;
    this->n_ion += (dn1 + 2*dn2 + 2*dn3 + dn4)/6.;

    // check spike
    
    if ((v0 < THRESHOLD) && (this->v >= THRESHOLD)){
        this->spike = true;
    } else {
        this->spike = false;
    }
}


double wbNeuron::get_minf(void){
    double am = -0.1 * (v+35) / (exp(-0.1 * (v + 35)) - 1);
    double bm = 4 * exp(-(v + 60)/18);
    return am/(am+bm);
}


double wbNeuron::solve_v(double v, double h_ion, double n_ion){
    double m_ion = get_minf();
    double ina = gna * m_ion * m_ion * m_ion * h_ion * (v - ena);
    double ik = gk * n_ion * n_ion * n_ion * n_ion * (v - ek);
    double il = gl * (v - el);
    double dv = (-ina - ik - il + iapp) / cm;
    return _dt * dv;
}


double wbNeuron::solve_h(double v, double h_ion, double n_ion){
    double ah = 0.07 * exp(-(v + 58)/20);
    double bh = 1 / (exp(-0.1 * (v + 28)) + 1);
    return _dt * phi * (ah * (1-h_ion) - bh * h_ion);
}

double wbNeuron::solve_n(double v, double h_ion, double n_ion){
    double an = -0.01 * (v + 34) / (exp(-0.1 * (v + 34)) - 1);
    double bn = 0.125 * exp(-(v + 44)/80);
    return _dt * phi * (an * (1-n_ion) - bn * n_ion);
}


double *linspace(double x0, double x1, int len_x){
    double *x = (double*) malloc(sizeof(double) * len_x);
    if (len_x == 1){
        // printf("Too few length selected. x is set to %.2f\n", x0);
        x[0] = x0;
        return x;
    }
    for (int n=0; n<len_x; n++){
        x[n] = n*(x1-x0)/(len_x-1)+x0;
    }
    return x;
}


void print_arr(FILE *fp, int N, const double *arr){
    for (int n=0; n<N; n++){
        fprintf(fp, "%f,", arr[n]);
    }
    fprintf(fp, "\n");
}