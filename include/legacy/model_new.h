#ifndef _neuralnet
#define _neuralnet

#include <iostream>
#include <cmath>
#include "ntk.h"
#include "rng.h"

#define _dt 0.005
#define _buffer_size 1000
#define _threshold 0


class wbNeuron{
    private:
        double phi=5, el=-65, ena=55, ek=-90; // mV
        double v0, h0, n0;
        double dv[4], dh[4], dn[4]; // -> temporal vars
        int stack=0;

    void solve_v(double ic, double factor=1);
    void solve_h(double factor=1);
    void solve_n(double factor=1);
    double get_minf(void);

    public:
        double vn=-70, hn=0, nn=0;
        bool spike;
        double iapp=0;
        double cm=1, gna=35, gk=9, gl=0.1;
        
    wbNeuron(void){};
    void update(double ic);
    void solve(double ic, double factor=1);
    void init_rk4();
    void apply_rk4();
    void check_fire();
};


class Synapse{

    private:
        void init_synapse(double ode_factor);

    public:
        double A, expr, expd;
        double rs=0, ds=0, ev=0, taur=1, taud=3;
        bool params_load=false;

    Synapse(void) {};
    void set_params(double _ev, double _taur=1, double _taud=3, double ode_factor=0.5);
    // need to add spike update method when you inherit the class
    void update(void);
    double get_current(double vpost);
};


class delaySynapse:public Synapse{
    private:
        int num_pre=-1, *in_node=NULL;
        double *w_in=NULL;
        bool is_const_delay;
        int delay_const, *delays=NULL;
        double spk_buffer[_buffer_size]{0,};
        int spk_step=0;

    public:
        // checkpoint varaibles
        bool ntk_load=false, strength_load=false, delay_load=false;

    void check_setting(void);
    void allocate_network(ntk_t *ntk, int nid);
    void allocate_strength(double g);
    void allocate_strength(int num_types, double gs[], int up_limit[]);
    void set_delay(double t_delay);
    void set_delay(int num_types, double t_delay[], int up_limit[]);
    // template <typename T>
    // void update_buffer(T *neurons);
    void update_buffer(wbNeuron *neurons);
    void add_spike(void);
    ~delaySynapse();
};


class PosSynapse:public Synapse{
    private:
        double nu, w, expl;

    public:
        bool attrib_load=false;

    void check_setting(void);
    void set_attrib(double _nu, double _g);
    void add_pos_spike(void);
};


#endif