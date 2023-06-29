#ifndef MODEL
#define MODEL

#include <iostream>
#define _dt 0.005


class wbNeuron{

    private:
        double phi=5, el=-65, ena=55, ek=-90; // mV
        double v0, h0, n0; // -> temporal vars

    public:
        double vn, hn, nn;
        bool *spk_buffer;
        double iapp=0;
        double cm=1, gna=35, gk=9, gl=0.1;
        
    wbNeuron(void);
    void update(double ic);
    void update_temporal(double ic, double factor);
    void sync_temporal();
    double get_minf(void);
    double solve_v();
    double solve_h();
    double solve_n();
    ~wbNeuron();
};


class deSynapse{
    public:
        double taur, taud;
};


#endif