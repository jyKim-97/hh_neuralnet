import numpy as np


def set_seed(seed):
    np.random.seed(seed)


# mapping label function
def mapping(row_names, num_in_pop=True):

    def replace_number(key):
        if not num_in_pop:
            return key
        
        for n in range(3):
            nid = key.find("(%d)"%(n))
            if nid > -1:
                key = key[::-1].replace("%d"%(n), key2number[n], 1)[::-1]
                return key
        return key

    # map
    key2lb = {"frs_m": "z",
              "chi": "\chi",
              "ac2p_1st": "A_1",
              "ac2p_large": "A_{large}",
              "tau_1st": "\\tau_1",
              "tau_large": "\\tau_{large}",
              "tlag_1st": "\\tau_{1}",
              "tlag_large": "\\tau_{large}",
              "tlag_cc": "\\tau_{cc}",
              "cc1p": "C",
              "leading_ratio": "\eta",
              "leading_ratio(abs)": "|\eta|",
              "dphi": "\Delta \phi"}

    key2number = {0: "T", 1: "F", 2: "S"}
    
    labels = row_names.copy()
    key2lb_list = list(key2lb.keys())
    for n, key in enumerate(row_names):
        for km in key2lb_list:
            if key.find(km) > -1:
                if km == "leading_ratio" and key.find("abs") > -1:
                    continue
                
                labels[n] = labels[n].replace(km, key2lb[km])
                break
        
        # change number
        labels[n] = replace_number(labels[n])
        if "_std" in key:
            labels[n] = labels[n].replace("_std", "")
            labels[n] = "\\sigma[%s]"%(labels[n])

        labels[n] = "$%s$"%(labels[n])
    return labels


def concat_data(post_data, key_to_rm=["cv"], include_std=False, show_mm_scale=False, norm_mm=True):
    import matplotlib.pyplot as plt

    """ Load data & concat """
    # concat data into one large matrix, row: features, col: data points
    row_names = []
    col_names = []

    # Convince that summary_data and summary_data_var have exactly same structure (not the value of them)
    summary_data = post_data["summary_data"]
    summary_data_std = post_data["summary_data_var"]

    # get size
    key_test = "nr0np0"
    Nr, Nc = np.shape(summary_data[key_test]["chi"])[:2]
    ndim = 0
    is_passed = {k: False for k in key_to_rm}
    for key in summary_data[key_test].keys():
        if key in key_to_rm:
            is_passed[key] = True
            continue

        shape = np.shape(summary_data[key_test][key])
        if len(shape) == 3:
            ndim += shape[2]
        else:
            ndim += 1
    
    # check key passing
    flag_pass = True
    for k, v in is_passed.items():
        if not v:
            print("%s not skipped, check the input"%(k))
            flag_pass = False
    
    if not flag_pass:
        print("Existing key in data")
        print(post_data.keys())
        return None

    # align data
    flag_row = True
    row_names = []
    align_data = []
    for k1 in summary_data.keys():
        keys = summary_data[k1].keys()

        if include_std:
            points = np.zeros([2*ndim, Nr*Nc])
        else:
            points = np.zeros([ndim, Nr*Nc])

        n = 0
        for k2 in keys:
            if k2 in key_to_rm:
                continue

            data = summary_data[k1][k2]
            if include_std:
                data_var = summary_data_std[k1][k2]

            shape = np.shape(data)
            if len(shape) == 3:
                ndim_sub = shape[2]
                for i in range(ndim_sub):
                    points[n, :] = np.array(data[:,:,i]).flatten(order="C")
                    if include_std:
                        points[n+ndim, :] = np.array(data_var[:,:,i]).flatten(order="C")

                    n += 1
                    if flag_row:
                        num = i + 1 if k2 == "cv" else i
                        
                        row_names.append(k2+"(%d)"%(num))
                        if include_std:
                            row_names.append(k2+"_std"+"(%d)"%(num))
            else:
                points[n, :] = np.array(data).flatten(order="C")
                if include_std:
                    points[n+ndim, :] = np.array(data_var).flatten(order="C")

                n += 1
                if flag_row:
                    # row_names.append(k2)
                    row_names.append(k2)
                    if include_std:
                        row_names.append(k2+"_std")

        flag_row = False
        for nr in range(Nr):
            for nc in range(Nc):
                col_names.append([k1, nr, nc])
        
        if len(align_data) == 0:
            align_data = points
        else:
            align_data = np.concatenate([align_data, points], axis=1)

    if include_std:
        # realign row_names if include_var=True
        row_names = row_names[0::2] + row_names[1::2]
        # remove overlap data: leading_ratio, leading_ratio(abs), dphi
        overlap_keys = ("leading_ratio", "leading_ratio(abs)", "dphi")
        for k in overlap_keys:
            if k+"_std" in row_names:
                nid = row_names.index(k+"_std")
                align_data = np.delete(align_data, nid, 0)
                row_names.pop(nid)
                print("skipped overlap key:", k+"_std")
    
    # check dimension
    print("datadim:", align_data.shape, end=", ")
    print("nrow: %d, ncol: %d"%(len(row_names), len(col_names)))

#     # Post processing: inverse tau
    key_inds = [n for n, s in enumerate(row_names) if "tau" in s and "std" in s]
    print("correcting std names target keys", [row_names[n] for n in key_inds])
    for n in key_inds:
        row_names[n] = row_names[n].replace("1/", "")
    
    xmin = np.min(align_data, axis=1)
    xmax = np.max(align_data, axis=1)

    if show_mm_scale:
        xavg = np.average(align_data, axis=1)
        plt.figure(dpi=100, figsize=(4, 4))
        plt.plot((xmax - xmin) / xavg, 'k.--', lw=1)
        plt.ylabel("(max(X) - min(X)) / mean(X)", fontsize=14)
        plt.xlabel("features", fontsize=14)
        plt.show()

    if norm_mm:
        align_data = (align_data - xmin[:, np.newaxis]) / (xmax - xmin)[:, np.newaxis]

    return align_data, row_names, col_names


def default_colors(col_names):
    cs = []
    for tag in col_names:
        idr, idp = int(tag[0][2]), int(tag[0][5])
        cr = 1 if idr == 0 else 0
        cb = idp/6
        cs.append([cr, cb, 0])
    return cs


def get_palette(cmap="jet"):
    from matplotlib.cm import get_cmap
    return get_cmap(cmap)


def save_fig(fig_name, fdir="./fig", dpi=100):
    import matplotlib.pyplot as plt
    # figname: don't type expander
    from datetime import datetime
    import os
    
    now = datetime.now()
    
    fname = os.path.join(fdir, fig_name)
    fname = fname + "_%d%d%d.png"%(now.year, now.month, now.day)
    print("save to %s"%fname)
    plt.savefig(fname, dpi=dpi)


def get_date_string():
    from datetime import datetime
    now = datetime.now()
    date = "%02d%02d%02d"%(now.year%100, now.month, now.day)
    return date


def draw_categorical_colorbar(n_category, dn=2, ax=None, cax=None, label=None, label_fontsize=None, **kwargs):
    import matplotlib.pyplot as plt
    
    cbar = plt.colorbar(ax=ax, cax=cax, **kwargs)
    ct = np.arange(1, n_category+1, dn).astype(int)
    ct_x = 0.5 + ct / n_category * (n_category-1)
    cbar.set_ticks(ct_x, labels=["%d"%(n) for n in ct])
    if label is not None:
        cbar.set_label(label, fontsize=label_fontsize)
    return cbar


def save_pkl(fname, **kwargs):
    import os
    import pickle as pkl
    
    if os.path.exists(fname):
        inp = input("File name %s exists, do you want to overwrite?")
        if inp == "n":
            print("change save file name")
            return
    
    if "date" not in list(kwargs.keys()):
        kwargs["date"] = get_date_string()
        
    with open(fname, 'wb')  as fp:
        pkl.dump(kwargs, fp)
    
    return
    

def load_pkl(fname):
    import pickle as pkl
    
    with open(fname, "rb") as fp:
        data = pkl.load(fp)
    
    if "date" in list(data.keys()):
        print("version: %s"%(data["date"]))
    
    return data