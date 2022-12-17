#include "model_new.h"

void wbNeuron::update(double ic){

    init_rk4();
    solve(ic, 0.5);
    solve(ic, 0.5);
    solve(ic, 1);
    solve(ic, 0);

    apply_rk4();
    check_fire();
}


void wbNeuron::init_rk4(){
    v0 = vn;
    h0 = hn;
    n0 = nn;
    stack = 0;
}


void wbNeuron::check_fire(void){
    if ((v0 < _threshold) && (vn >= _threshold)){
        spike = true;
    } else {
        spike = false;
    }
}


void wbNeuron::solve(double ic, double factor){
    solve_v(ic);
    solve_h();
    solve_n();
    stack++;
}


void wbNeuron::apply_rk4(void){
    vn = v0 + (dv[0] + 2*dv[1] + 2*dv[2] + dv[3])/6.;
    hn = h0 + (dh[0] + 2*dh[1] + 2*dh[2] + dh[3])/6.;
    nn = n0 + (dn[0] + 2*dn[1] + 2*dn[2] + dn[3])/6.;
}


double wbNeuron::get_minf(void){
    double am = -0.1 * (vn+35) / (exp(-0.1 * (vn + 35)) - 1);
    double bm = 4 * exp(-(vn + 60)/18);
    return am / (am + bm);
}


void wbNeuron::solve_v(double ic, double factor){
    double mn = get_minf();
    
    double ina = gna * mn * mn * mn * hn * (vn - ena);
    double ik  = gk * nn * nn * nn * nn * (vn - ek);
    double il  = gl * (vn - el);
    dv[stack] = (-ina - ik - il + ic) / cm;
    vn = v0 + factor * dv[stack];
}


void wbNeuron::solve_h(double factor){
    double ah = 0.07 * exp(-(vn + 58)/20);
    double bh = 1 / (exp(-0.1 * (vn + 28)) + 1);
    dh[stack] = _dt * phi * (ah * (1-hn) - bh * hn);
    hn = h0 + factor * dh[stack];
}


void wbNeuron::solve_n(double factor){
    double an = -0.01 * (vn + 34) / (exp(-0.1 * (vn + 34)) - 1);
    double bn = 0.125 * exp(-(vn + 44)/80);
    dn[stack] = _dt * phi * (an * (1-nn) - bn * nn);
    nn = n0 + factor * dn[stack];
}


Synapse::Synapse(double _ev, double ode_factor){
    ev = _ev;
    init_synapse(ode_factor);
}


Synapse::Synapse(double _ev, double _taur, double _taud, double ode_factor){
    ev = _ev;
    taur = _taur;
    taud = _taud;
    init_synapse(ode_factor);

}


void Synapse::init_synapse(double ode_factor){
    expr = exp(-_dt * ode_factor / taur);
    expd = exp(-_dt * ode_factor / taud);

    double tp = taur * taud / (taud - taur) * log(taud/taur);
    A = 1 / (exp(-tp/taur) - exp(-tp/taud));
}


void Synapse::update(void){
    rs *= expr;
    ds *= expd;
}


double Synapse::get_current(double vpost){
    return (rs - ds) * (vpost - ev);
}


void delaySynapse::check_setting(void){
    if (!ntk_load){
        std::cout << "Network setting is not done!" << std::endl;
    }

    if (!strength_load){
        std::cout << "Strength setting is not done!" << std::endl;
    }

    if (!delay_load){
        std::cout << "Delay setting is not done! Set with 0 automatically" << std::endl;
        set_delay(0);
    }
}


void delaySynapse::allocate_network(ntk_t *ntk, int nid){
    if (ntk->edge_dir == outdeg){
        std::cout << "Out degree type inserted" << std::endl;
        return;
    }

    num_pre = ntk->num_edges[nid];
    in_node = new int[num_pre];
    for (int n=0; n<num_pre; n++){
        in_node[n] = ntk->adj_list[nid][n];
    }
    ntk_load = true;
}


void delaySynapse::allocate_strength(double g){
    if (!ntk_load){
        std::cout << "Network is not loaded" << std::endl;
    }

    w_in = new double[num_pre];
    for (int n=0; n<num_pre; n++){
        w_in[n] = g*A;
    }

    strength_load = true;
}


void delaySynapse::allocate_strength(int num_types, double gs[num_types], int up_limit[num_types]){
    if (!ntk_load){
        std::cout << "Network is not loaded" << std::endl;
    }

    w_in = new double[num_pre];
    for (int n=0; n<num_pre; n++){
        int nid = in_node[n];
        if (nid > up_limit[num_types-1]){
            std::cout << "Post Node exceeds the range!" << std::endl;
        }

        int type = 0;
        while (nid < up_limit[type]){
            type++;
        }; type--;

        w_in[n] = gs[type]*A;
    }

    strength_load = true;
}


void delaySynapse::set_delay(double t_delay){
    delay_const = t_delay / _dt;
    is_const_delay = true;
    delay_load = true;
}


void delaySynapse::set_delay(int num_types, double t_delay[num_types], int up_limit[num_types]){
    is_const_delay = false;
    if (!ntk_load){
        std::cout << "Network is not generated" << std::endl;
        return;
    }

    delays = new int[num_pre];
    for (int n=0; n<num_pre; n++){
        int nid = in_node[n];
        if (nid > up_limit[num_types-1]){
            std::cout << "Post Node exceeds the range!" << std::endl;
        }

        int type = 0;
        while (nid < up_limit[type]){
            type++;
        }; type--;

        delays[n] = t_delay[type]/_dt;
    }

    delay_load = true;
}


template <typename T>
void delaySynapse::update_buffer(T *neurons){

    int nbuf;
    if (is_const_delay){ nbuf = (spk_step + delay_const) % _buffer_size; }

    for (int n=0; n<num_pre; n++){
        int nid = in_node[n];
        if (!is_const_delay){ nbuf = (spk_step + delays[n]) % _buffer_size; }

        if (neurons[nid].spike){
            spk_buffer[nbuf] += w_in[nid];
        }
    }
}


void delaySynapse::add_spike(void){
    rs += spk_buffer[spk_step];
    ds += spk_buffer[spk_step];
    spk_buffer[spk_step] = 0;
    spk_step = spk_step==_buffer_size? 0:spk_step+1;
}


delaySynapse::~delaySynapse(void){
    if (!is_const_delay){
        delete[] delays;
    }

    delete[] w_in;
    delete[] in_node;
}


void PosSynapse::allocate_pos_attrib(double _nu, double _g){
    nu = _nu;
    w = _g;
    double l = _dt/1000. * nu;
    expl = exp(-l);
}


void PosSynapse::add_pos_spike(void){
    int num_ext = pick_random_poisson(expl);
    rs += w * num_ext;
    ds += w * num_ext;
}


void PosSynapse::check_setting(void){
    if (!attrib_load){
        std::cout << "Attribute setting is not done!" << std::endl;
    }
}
