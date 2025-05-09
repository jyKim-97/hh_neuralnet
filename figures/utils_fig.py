import pickle as pkl
import xarray as xa
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


fdir_prev = "./prev"
fdir_cur  = "./figures"

# set Figure Max Width: 15

def save_pickle(fname, data, replace=False):
    if not replace and os.path.isfile(fname):
        raise ValueError("File %s exist"%(fname))
    
    with open(fname, "wb") as fp:
        pkl.dump(fp)


def load_pickle(fname):
    with open(fname, "rb") as fp:
        return pkl.load(fp)
    
    
def load_dataarray(fname):
    if os.path.exists(fname):
        dataarray = xa.open_dataarray(fname)
        return dataarray
    else:
        raise FileNotFoundError(f"File {fname} does not exist.")
    
    
def load_dataset(fname):
    if os.path.exists(fname):
        dataset = xa.open_dataset(fname)
        return dataset
    else:
        raise FileNotFoundError(f"File {fname} does not exist.")
    
    
def set_plt():
    font_files = fm.findSystemFonts(fontpaths="./font")
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
        
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    # Tick property
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5

    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.left'] = True

    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11

    # Line setting
    plt.rcParams["lines.linewidth"] = 1.2
    
    
def show_spline(ax, top=False, right=False, bottom=False, left=False):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)
    
    
def show_scalebar(ax,
                  size=10, label="1 s", vh="horizontal",
                  anchor_pos=None, pad=0.5, lw=2, color="k",
                  fontsize=12):
    
    """
    ------
    Parameters
    ax: matplotlib.axes.Axes
        The axes to draw the scale bar on.
    size: float
        Length of the scale bar in data coordinates.
    label: str
        Label for the scale bar.
    vh: str
        Orientation of the scale bar. Options are 'horizontal' or 'vertical'.
    anchor_pos: tuple
        Position of the scale bar in data coordinates. If None, the position is
        determined automatically.
    pad: float
        Padding between the scale bar and the label.
    lw: float
        Line width of the scale bar.
    color: str or tuple
        Color of the scale bar.
    fontsize: int
        Font size of the label.
    -------
    """
    
    assert vh in ["horizontal", "vertical"], "vh must be 'horizontal' or 'vertical'"
    
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    dx = xl[1] - xl[0]
    dy = yl[1] - yl[0]
    
    if anchor_pos is None:
        y0 = yl[0] + 0.1*dy
        x0 = xl[0] + 0.1*dx
    else:
        assert len(anchor_pos) == 2, "anchor_pos must be a tuple of (x, y)"
        x0, y0 = anchor_pos

    if vh == "horizontal":
        ax.plot([x0, x0+size], [y0, y0], color=color, lw=lw)
        ax.text(x0+size/2, y0-pad, label, ha="center", va="top", color=color,
                fontsize=fontsize)
    elif vh == "vertical":
        ax.plot([x0, x0], [y0, y0+size], color=color, lw=lw)
        ax.text(x0-pad, y0+size/2, label, ha="right", va="center", rotation=90, color=color,
                fontsize=fontsize)
        
        
def get_axlim(ax):
    return ax.get_xlim(), ax.get_ylim()
    
    


# def show_scalebar(ax, size=1, label="1 s", loc="lower right",
#                   color='k',
#                   pad=0.5, borderpad=0.5,
#                   sep=5, frameon=False, size_vertical=0):
    
#     """ 
#     ---------
#     Parameters
#     size: float
#         Length of the scale bar in data coordinates.
#     label: str
#         Label for the scale bar.
#     loc: str
#         Location of the scale bar. Options are:
#         'upper right', 'upper left', 'lower left', 'lower right'.
#     pad: float
#         Padding between the scale bar and the axes.
#     borderpad: float
#         Padding between the scale bar and the border of the figure.
#     sep: float
#         Separation between the label and the scale bar.
#     frameon: bool
#         Whether to draw a frame around the scale bar.
#     color: str
#         Color of the scale bar.
#     size_vertical: float
#         Length of the vertical scale bar in data coordinates.
    
#     """
    
#     from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#     scalebar = AnchoredSizeBar(ax.transData,
#                                size=size,  # this is the size of the bar in data coordinates
#                                label=label, loc=loc, pad=pad, borderpad=borderpad, sep=sep, 
#                                frameon=frameon, color=color, size_vertical=size_vertical)
#     ax.add_artist(scalebar)
    
    
    
def remove_ticklabels(ax):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    
def save_fig(fname):
    fname = fname.split(".")[0]
    plt.savefig(os.path.join(fdir_cur, fname+".png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(fdir_cur, fname+".svg"), dpi=300, bbox_inches="tight")
    print("Saved figure to %s"%(fname))
    

def backup_fig():
    import shutil
    from datetime import datetime
    
    x = datetime.now()
    fdir_backup = os.path.join(fdir_prev, x.strftime("%y%m%d"))
    if not os.path.exists(fdir_backup):
        os.makedirs(fdir_backup)
    
    print("Copy files to %s"%(fdir_backup))
    fnames = [f for f in os.listdir() if "figure" in f and "ipynb" in f]
    print(fnames)
    for f in fnames:
        shutil.copyfile(f, os.path.join(fdir_backup, f))
    shutil.copytree("./figures", os.path.join(fdir_backup, "figures"))
    
    