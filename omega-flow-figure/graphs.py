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
    colors = ["red", "gray", "None", "green", "orange", "purple"]
    edgecolor = "black"
    edgecolors = ["black", "blue"]
    fig_size = (14, 4)
    bar_width, bar_interval = 0.6, 0.4

class GraphMaker(object):
    def __init__(self, fig_size=None):
        self.config = GraphDefaultConfig()
        if fig_size is None:
            fig_size = self.config.fig_size
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(fig_size[0], fig_size[1], forward=True)

    def set_graph_general_format(self, xlim, ylim, xticklabels, xlabel, ylabel):
        self.ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
        # ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        for tick in self.ax.xaxis.get_major_ticks():
            tick.tick1line.set_markersize(10)
            tick.tick2line.set_markersize(0)
        for tick in self.ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(2)
            # tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('left')
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xticklabels(xticklabels, minor=True, rotation=90)
        self.ax.grid(axis="y", linestyle="--", color='gray', alpha=0.3)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def simple_bar_graph(self, data, xticklabels, legends, xlabel="", ylabel="", 
            colors=None, edgecolor=None, xlim=(None,None), ylim=(None,None), overlap=False):
        if colors is None:
            colors = self.config.colors
        if edgecolor is None:
            edgecolor = self.config.edgecolor

        assert(len(data.shape) == 2)
        num_configs, num_points = data.shape
        assert(len(legends) == num_configs)
        if overlap:
            num_configs = 1

        print(num_points, num_configs)
        shift = 0.0
        rects = []
        bar_size = self.config.bar_width + self.config.bar_interval
        for i, d in enumerate(data):
            tick_starts = np.arange(0, num_points * num_configs, (bar_size * num_configs)) + shift
            rect = plt.bar(tick_starts, d, edgecolor=edgecolor, color=colors[i], width=self.config.bar_width)
            rects.append(rect)
            shift += 0 if overlap else self.config.bar_width + self.config.bar_interval
        self.ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=bar_size*num_configs*3, offset=-self.config.bar_interval/2))
        self.ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(base=bar_size*num_configs, offset=-self.config.bar_interval/2))
        self.set_graph_general_format(xlim, ylim, xticklabels, xlabel, ylabel)
        self.ax.legend(rects, legends, fontsize='small', ncol=num_configs)

        plt.tight_layout()
        return self.fig, self.ax
    
    def reduction_bar_graph(self, data_high, data_low, xticklabels, legends, xlabel="", ylabel="", 
            colors=None, edgecolors=None, xlim=(None,None), ylim=(None,None), legendorder=None):
        if colors is None:
            colors = [self.config.colors, ["None"]*len(self.config.colors)]
        if edgecolors is None:
            edgecolors = [self.config.colors, self.config.edgecolors]

        print(data_high.shape, data_low.shape)
        assert(len(data_high.shape) == 2 and len(data_low.shape) == 2)
        num_configs, num_points = data_high.shape
        assert((num_configs, num_points) == data_low.shape)
        # assert(num_configs == 2)

        same_high_data = False
        num_legends = 2 * num_configs
        if num_configs == 2:
            if np.array_equal(data_high[0], data_high[1]):
                num_legends = num_configs + 1
                edgecolors = [[self.config.edgecolor] * 2] * 2
                same_high_data = True
            print(legends, num_legends)
            assert(len(legends) == num_legends)

        print(num_points, num_configs)
        rects = []

        bar_size = self.config.bar_width + self.config.bar_interval
        for i, data in enumerate([data_low, data_high]):
            shift = 0.0
            for j, d in enumerate(data):
                tick_starts = np.arange(0, num_points * num_configs, bar_size * num_configs) + shift
                rect = plt.bar(tick_starts, d, edgecolor=edgecolors[i][j], color=colors[i][j], width=self.config.bar_width)
                rects.append(rect)
                shift += bar_size
        rects = rects[2:] + rects[:2]
        if legendorder is not None:
            new_rects, new_legends = [], []
            for i in legendorder:
                new_rects.append(rects[i])
                new_legends.append(legends[i])
            rects = new_rects
            legends = new_legends

        self.ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=bar_size*num_configs*3, offset=-self.config.bar_interval/2))
        self.ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(base=bar_size*num_configs, offset=-self.config.bar_interval/2))
        self.set_graph_general_format(xlim, ylim, xticklabels, xlabel, ylabel)
        
        if same_high_data:
            rects = [rects[0]] + rects[2:]
        self.ax.legend(rects, legends, fontsize='small', ncol=num_legends)

        plt.tight_layout()
        return self.fig, self.ax

    def save_to_file(self, plt, name):
        for f in ['eps', 'png']:
            filename = f'./{f}/{name}.{f}'
            print("save to", filename)
            plt.savefig(filename, format=f'{f}')
