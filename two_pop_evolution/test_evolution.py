import evolve
import matplotlib.pyplot as plt
import numpy as np
# need to install drawnow
# from drawnow import drawnow
from tqdm import tqdm
import matplotlib.animation as anim


# def fpotential(data):
#     x0, y0 = 5, 5
#     return -(data[0]-x0)**2 - (data[1]-y0)**2


def fpotential(args):
    A = 10
    data = args[0]
    data = np.array(data)
    return -A + np.sum(data**2 - A * np.cos(2*np.pi*data))


def get_landscape(fobj, pmin, pmax, w=50):

    x = np.linspace(pmin[0], pmax[0], w)
    y = np.linspace(pmin[1], pmax[1], w)
    z = np.zeros([w, w])
    for i in range(w):
        for j in range(w):
            z[i, j] = fobj([[x[i], y[j]], -1])
    return x, y, z



if __name__ == "__main__":
    solver = evolve.EA(2, mu=2, num_select=5, num_offspring=20, num_parent=50)
    solver.set_object_func(fpotential)

    # pmin = np.array([0, 0])
    # pmax = np.array([10, 10])
    pmin = np.array([-5.12, -5.12])
    pmax = np.array([5.12, 5.12])
    solver.set_min_max(pmin, pmax)

    param_set = []
    solver.random_initialization()
    solver.reset_job_id()
    param_set.append(solver.param_vec.copy())
    for n in tqdm(range(2000)):
        solver.next_generation()
        solver.print_log(n)
        param_set.append(solver.param_vec.copy())

    x, y, z = get_landscape(fpotential, pmin, pmax)
    # fig = plt.figure(dpi=120, figsize=(4,4))
    # plt.contour(x, y, -z, 50, cmap="jet")
    # plt.show()

    fig, ax = plt.subplots(1, 1, dpi=120, figsize=(5,5), facecolor="w")
    ax.contour(x, y, -z, 20, cmap="jet")
    p, = ax.plot([], [], "ko", markerfacecolor="k", lw=2)
    pobj = [p]

    def init():
        ax.set_xlim([pmin[0], pmax[0]])
        ax.set_ylim([pmin[1], pmax[1]])
        # ax.axis("square")
        return pobj

    def update(nt):
        x = param_set[nt][0]
        y = param_set[nt][1]
        pobj[0].set_data(x, y)
        return pobj


    ani = anim.FuncAnimation(fig, update, frames=np.arange(0,2000,10),
                    init_func=init, blit=True)

    writergif = anim.PillowWriter(fps=30) 
    ani.save("./test_evolution.gif", writer=writergif)
    plt.close()
    plt.show()
    

