import numpy as np
import matplotlib.pyplot as plt


fname = "./check_v.txt"
dt = 0.005


if __name__ == "__main__":
    with open(fname, "r") as fp:
        vs = [float(x) for x in fp.readline().split(",")[:-1]]

    ts = np.arange(len(vs))*dt/1e3

    plt.figure(dpi=120, figsize=(4,4), facecolor='w')
    plt.plot(ts, vs, 'k', lw=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("V (mV)")
    plt.tight_layout()
    plt.savefig("./check_v.png")
    # plt.show()