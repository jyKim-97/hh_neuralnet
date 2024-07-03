"""
Python library for drawing summary from hh_neuralnet simulation
"""

import matplotlib.pyplot as plt
import matplotlib.offsetbox
from matplotlib.lines import Line2D
import hhsignal
import hhtools

# Shared parameters
pop_labels = ("Total(0)", "Fast(1)", "Slow(2)")



# scalebar
class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None, 
                 frameon=True, linecolor="k",
                 fontcolor="k", fontsize=10, fontstyle="normal", **kwargs):
        if not ax:
            ax = plt.gca()
            
        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], color=linecolor)
        vline1 = Line2D([0,0], [-extent/2.,extent/2.], color=linecolor)
        vline2 = Line2D([size,size], [-extent/2.,extent/2.], color=linecolor)
        vline3 = Line2D([size/2,size/2], [-extent/4.,extent/4.], color=linecolor)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        size_bar.add_artist(vline3)
        txt = matplotlib.offsetbox.TextArea(label, textprops=dict(color=fontcolor, size=fontsize, style=fontstyle))
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],  
                                 align="center", pad=ppad, sep=sep) 
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, 
                 borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
                 **kwargs)


def add_scalebar(barsize, label, loc="lower left", ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
        
    ob = AnchoredHScaleBar(ax=ax, size=barsize, label=label, loc=loc,
                           pad=0.08, sep=2, **kwargs)
    ax.add_artist(ob)
    
    
# draw psd summary
def show_psd_summary(detail_data, frange=(10, 90), xrange=(3, 7), mbin_t=0.1, wbin_t=0.5,
                     vmin=0.05, vmax=None, fs=2000, figsize=(3.5, 6), dpi=120,
                     suptitle=None):
    
    t = detail_data["ts"]
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for ntp in range(3):
        v = detail_data["vlfp"][ntp]
        psd_t, fpsd, tpsd = hhsignal.get_stfft(v, t,
                                               fs=fs, mbin_t=mbin_t, wbin_t=wbin_t, frange=frange)
        
        plt.subplot(3,1,ntp+1)
        
        # draw
        hhtools.imshow_xy(psd_t, x=tpsd, y=fpsd, cmap="jet", interpolation="bicubic", vmin=vmin, vmax=vmax)
        plt.xlim(xrange)
        plt.colorbar()
        # plt.colorbar(ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        plt.ylabel("Frequency (Hz)", fontsize=14)
        plt.xticks([])
        
        # add scalebar
        add_scalebar(1, "1 s", loc="lower left", frameon=False,
                     linecolor="w", fontcolor="w", fontsize=12, fontstyle="italic")
        
        # ob = AnchoredHScaleBar(size=1, label="1 s", loc="lower left", frameon=False,
        #                     pad=0.08, sep=2, linecolor="w", fontcolor="w",
        #                     fontsize=12, fontstyle="italic")
        # add_scalebar(ob)
        
        if ntp == 0:
            plt.title(suptitle, fontsize=15)
            
        plt.text(xrange[1]-2, 79, pop_labels[ntp],
                ha="left", va="center",
                fontweight="bold", fontstyle="italic", fontsize=14, color="w")
        
    plt.tight_layout()    
    # plt.show()
        
    return fig