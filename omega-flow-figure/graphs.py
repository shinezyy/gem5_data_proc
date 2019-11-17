#!/usr/bin/env python3

import sys
sys.path.append('.')

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class GraphDefaultConfig(object):
    colors = ["r","gray"]
    fig_size = (14, 4)

class GraphHelper(object):
    def __init__(self):
        self.config = GraphDefaultConfig()

    def simple_bar_graph(self, data, xticklabels, legends, fig_size=None, xlabel="", ylabel="", colors=None, xlim=(None,None), ylim=(None,None)):
        if colors is None:
            colors = self.config.colors
        if fig_size is None:
            fig_size = self.config.fig_size

        fig, ax = plt.subplots()
        fig.set_size_inches(fig_size[0], fig_size[1], forward=True)
        width, interval = 0.6, 0.4

        assert(len(data.shape) == 2)
        num_configs, num_points = data.shape
        assert(len(legends) == num_configs)

        print(num_points, num_configs)
        shift = 0.0
        rects = []
        for i, data in enumerate(data):
            tick_starts = np.arange(0, num_points * num_configs, (width + interval) * num_configs) + shift
            rect = plt.bar(tick_starts, data, edgecolor='black', color=colors[i], width=width)
            rects.append(rect)
            shift += width + interval

        ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=(width+interval)*num_configs*3, offset=-interval/2))
        ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(base=(width+interval)*num_configs, offset=-interval/2))

        ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
        # ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_markersize(10)
            tick.tick2line.set_markersize(0)

        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(2)
            # tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('left')

        ax.set_xticklabels(xticklabels, minor=True, rotation=90)

        ax.grid(axis="y", linestyle="--", color='gray')
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        ax.legend(rects, legends, fontsize='small', ncol=num_configs, 
                loc='lower left', bbox_to_anchor=(0.788,0.88))

        plt.tight_layout()

        return plt, (ax,)
    
    def save_to_file(self, plt, name):
        for f in ['eps', 'png']:
            plt.savefig(f'./{f}/{name}.{f}', format=f'{f}')