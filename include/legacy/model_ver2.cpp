#include "model_ver2.h"

#define DEBUG

constexpr int _buf_size = 1000;


wbNeuron::wbNeuron(){
    vn = -70;
    hn = 0;
    nn = 0;
    spk_buffer = new bool[_buf_size];
}


void wbNeuron::solve(double ic, double factor){
    double dv = solve_v();
    double dh = solve_h();
    double dn = solve_n();
}