import numpy as np
import pickle as pkl
from tqdm import trange
from sklearn.manifold import TSNE
import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", default="../simulation_data/purified_data.pkl")
    parser.add_argument("--fout", help="output file name", default="./tsne_out.pkl")
    parser.add_argument("--seed", default=200, type=int, help="init seed")
    parser.add_argument("--norm", default=True, type=bool, help="normalize data or not")
    return parser


def load_data(fname):
    with open(fname, "rb") as fp:
    # with open("../simulation_data/purified_data.pkl", "rb") as fp:
        return pkl.load(fp)["data"]
    

def fit_tsne(data, seed, perplex):
    np.random.seed(seed)
    tsne_obj = TSNE(n_components=2, init="pca", n_jobs=20,
                    perplexity=perplex, n_iter=1000, angle=0.5)
    return tsne_obj.fit_transform(data)


def znorm(data):
    m = np.average(data, axis=1)[:, np.newaxis]
    s = np.std(data, axis=1)[:, np.newaxis]
    return (data - m)/s

    
def main(fname=None, fout="./tsne_out.pkl", seed=100, norm=True):
    np.random.seed(seed)
    seeds = np.random.randint(low=1, high=1e4, size=10)
    perplexities = [5, 10, 20, 40, 50]
    
    data = load_data(fname)
    if norm:
        data = znorm(data)
    
    e2_data = []
    params = []
    div = len(perplexities)
    for n in trange(len(seeds) * len(perplexities)):
        s = seeds[n // div]
        p = perplexities[n % div]
        e2_data.append(fit_tsne(data.copy().T, s, p))
        params.append({"seed": s, "perplexity": p})

    with open(fout, "wb") as fp:            
        pkl.dump({"tsne_data": e2_data,
                  "params": params,
                  "seeds": seeds,
                  "perplexities": perplexities}, fp)

    print("Export data to %s"%(fout))
    

if __name__ == "__main__":
    main(**vars(build_arg_parser().parse_args()))
    