import numpy as np
import multiprocess as mp
from scipy.linalg import null_space
import os
import pickle as pkl


class EA:
    def __init__(self, num_params, log_dir="./log", mu=2, num_offspring=5, num_parent=10, max_iter=1e6, tol=1e-3):
        self.max_iter = int(max_iter)
        self.num_parent = int(num_parent)
        self.num_offspring = int(num_offspring)
        self.num_params = int(num_params)
        self.tol = tol
        self.pmin = None
        self.pmax = None
        self.log_dir = log_dir

        self.mu = mu
        self.param_vec = np.zeros([self.num_params, self.num_parent])
        self.fit_score = np.zeros([self.num_parent])
        self.fobj = None
        # A Functional Specialization Hypothesis for Designing Genetic Algorithms 
        self.sgm_eta = 1/np.sqrt(self.mu)
        self.sgm_xi  = 0.35/np.sqrt(self.num_parent - self.mu)

    def set_object_func(self, f):
        # object function need to return float (fitness)
        # res = f(arr)
        self.fobj = f

    def set_min_max(self, pmin, pmax):
        if (len(pmin) != self.num_params) or (len(pmin) != self.num_params):
            print("The number of min-max is wrong")
            return

        self.pmin = np.array(pmin)
        self.pmax = np.array(pmax)

    def check_setting(self):
        if self.num_parent < self.num_offspring:
            raise ValueError("The number of offspring is larger than # parents")

        if self.mu >= self.num_params:
            raise ValueError("mu cannot be larger than # params")

        if self.pmin is None or self.pmax is None:
            raise AttributeError("Boundary is not defined: call set_min_max")

        if self.fobj is None:
            raise AttributeError("Object function is not defined: call set_object_func")

    def run(self, auto_init=True):
        job_id = 0
        ncycle = 0
        dscore = 0

        # check setting
        self.check_setting()

        if auto_init:
            self.random_initialization()

        for n in range(self.max_iter):
            self.next_generation()
            self.print_log(n)

    def random_initialization(self):
        for n in range(self.num_parent):
            self.param_vec[:, n] = np.random.uniform(self.pmin, self.pmax)

        with mp.Pool(self.num_parent) as p:
            self.fit_score = np.array(p.map(self.fobj, self.param_vec.T))

    def next_generation(self):
        # select 2 parents
        parent_id = np.arange(self.num_parent)
        id_selected = np.random.choice(parent_id, 2, replace=False)

        # get offsprings & evaluate scores
        offspring = self.crossover()

        # with mp.Pool(self.num_offspring) as p:
        #     fitness = p.map(self.fobj, offspring.T)

        fitness = []
        for n in range(self.num_offspring):
            fitness.append(self.fobj(offspring[:,n]))

        # evalutate score
        pop_scores = np.zeros(self.num_offspring+2)
        pop_scores[:-2] = fitness
        pop_scores[-2:] = self.fit_score[id_selected]

        # Roulette-wheel selection (X)
        # choose best two solutions
        id_order = np.argsort(pop_scores)[::-1]
        for n in range(2):
            nid = id_order[n]
            new_id = id_selected[n]
            if nid < self.num_offspring:
                self.param_vec[:, new_id] = offspring[:, nid]
            else:
                nold = id_selected[nid - self.num_offspring] 
                self.param_vec[:, new_id] = self.param_vec[:, nold]

            self.fit_score[new_id] = pop_scores[nid]

    def crossover(self):
        offspring = np.zeros([self.num_params, self.num_offspring])
        for n in range(self.num_offspring):
            # check boundary condition
            flag = True
            while flag:
                offspring_tmp = self.crossover_undx()
                if all(offspring_tmp <= self.pmax) and all(offspring_tmp >= self.pmin):
                    offspring[:, n] = offspring_tmp
                break

        return offspring

    def crossover_undx(self):
        # ==================================
        # Ref)
        # H. Kita & M. Yamamura, IEEE, 1999, A Functional Specialization Hypothesis for Designing Genetic Algorithms
        # I. Ono et al., A Real-coded Genetic Algorithm using the Unimodal Normal Distribution Crossover
        # K. Deb et al., A Computationally Efficient Evolutionary Algorithm for Real-Parameter Optimization
        # ==================================

        # select mu parents (mu < n), span the vectorspace V
        id_select, id_remain = self.pick_id(self.num_parent, self.mu)
        g_vec = np.average(self.param_vec[:, id_select], axis=1)
        d_vec = self.param_vec[:, id_select[:-1]] - g_vec[:, np.newaxis]

        # check is d_vec is null vector
        if get_norm(d_vec) < 1e-5:
            return g_vec
        else:
            # find the basis of vector space W that orthogonal to vector space V
            basis = null_space(d_vec.T)

            # find the distance from the vector space which spanned by d_vec
            nd = np.random.choice(id_remain)
            v = self.param_vec[:, nd] - g_vec
            coord = np.array([np.dot(v, basis[:,n]) for n in range(basis.shape[1])])
            D = np.sqrt(np.sum(coord**2))

        # get offspring
        offspring = g_vec
        # 1st term
        eta = np.random.randn(self.mu-1, 1) * self.sgm_eta
        offspring += np.squeeze(np.dot(d_vec, eta))
        # 2nd term
        xi = np.random.randn(self.num_params-self.mu+1, 1) * self.sgm_xi
        offspring += np.squeeze(D * np.dot(basis, xi))

        return offspring

    def crossover_pcx(self):
        # select mu parents (mu < n), span the vectorspace V
        id_select, id_remain = self.pick_id(self.num_parent, self.mu)
        g_vec = np.average(self.param_vec[:, id_select], axis=1)
        d_vec = self.param_vec[:, id_select[:-1]] - g_vec[:, np.newaxis]

        # select one parents from selected id
        nd = np.random.choice(id_select)

    def pick_id(self, max_id, num_pick):
        id_remain = list(range(max_id))
        id_select = np.random.choice(id_remain, num_pick, replace=False)
        for nid in id_select:
            id_remain.remove(nid)
        id_remain = np.array(id_remain)
        return id_select, id_remain

    def mutate(self):
        pass

    def print_log(self, nstep, skip_save_param=10):
        # save fitness
        with open(os.path.join(self.log_dir, "log.txt"), "a") as fid:
            for n in range(self.num_parent):
                fid.write("%f,"%(self.fit_score[n]))
            fid.write("\n")

        # save parameters
        if nstep % skip_save_param == 0:
            with open(os.path.join(self.log_dir, "params_%d.pkl"%(nstep)), "wb") as fid:
                pkl.dump(self.param_vec, fid)


def get_norm(vec):
    return np.sqrt(np.sum(vec**2))


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q