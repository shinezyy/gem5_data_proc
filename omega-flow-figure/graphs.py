#!/usr/bin/env python3

import sys

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

class GraphDefaultConfig(object):
    colors = ["red", "gray", "None", "green", "orange", "purple"]
    edgecolor = "black"
    edgecolors = ["black", "blue"]
    fig_size = (14, 3)
    bar_width, bar_interval = 0.6, 0.4

class GraphMaker(object):
    def __init__(self, fig_size=None, multi_ax=False, *args, **kwargs):
        self.config = GraphDefaultConfig()
        if fig_size is None:
            fig_size = self.config.fig_size
        if multi_ax:
            self.fig, self.ax = plt.subplots(*args, **kwargs)
        else:
            self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(fig_size[0], fig_size[1], forward=True)
        self.cur_ax = self.ax

    def set_graph_general_format(self, xlim, ylim, xticklabels, xlabel, ylabel,
            title=None):
        self.cur_ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
        # cur_ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # print(self.cur_ax.xaxis.get_major_ticks())
        for tick in self.cur_ax.xaxis.get_major_ticks():
            tick.tick1line.set_markersize(10)
            tick.tick2line.set_markersize(0)
        for tick in self.cur_ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(2)
            # tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('left')
        self.cur_ax.set_xlim(xlim)
        self.cur_ax.set_ylim(ylim)
        self.cur_ax.set_xticklabels(xticklabels, minor=True, rotation=90)
        self.cur_ax.grid(axis="y", linestyle="--", color='gray', alpha=0.3)
        self.cur_ax.set_xlabel(xlabel)
        self.cur_ax.set_ylabel(ylabel)
        if title is not None:
            self.cur_ax.title.set_text(title)

    def simple_bar_graph(self, data, xticklabels, legends, xlabel="", ylabel="",
            colors=None, edgecolor=None, linewidth=None, xlim=(None,None), ylim=(None,None), overlap=False,
            set_format=True, xtick_scale=1, title=None, with_borders=False):
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
            tick_starts *= xtick_scale
            rect = plt.bar(tick_starts, d, edgecolor=edgecolor, linewidth=linewidth,
                    color=colors[i], width=self.config.bar_width * xtick_scale)
            rects.append(rect)
            shift += 0 if overlap else self.config.bar_width + self.config.bar_interval

        # self.cur_ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(
        #     base=bar_size*num_configs*3, offset=-self.config.bar_interval/2))
        # self.cur_ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(
        #     base=bar_size*num_configs, offset=-self.config.bar_interval/2))

        tick_locators = mpl.ticker.IndexLocator(
            base=bar_size*num_configs*3,
            # offset=-self.config.bar_interval/2 + self.config.bar_width,
            offset=-self.config.bar_interval/4,
            )

        if with_borders:
            for mt in tick_locators.tick_values(0, bar_size*num_configs*3*num_points):
                self.cur_ax.axvline(mt*xtick_scale - self.config.bar_width/3*2,
                        c='black', linestyle='-.',
                        )

        self.cur_ax.xaxis.set_major_locator(tick_locators)

        self.cur_ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(
            base=bar_size*num_configs,
            # offset=-self.config.bar_interval/2 + self.config.bar_width,
            offset=-self.config.bar_interval/4,
            ))


        if set_format:
            self.set_graph_general_format(xlim, ylim, xticklabels, xlabel, ylabel, title=title)
        self.cur_ax.legend(rects, legends, fontsize='small', ncol=num_configs)
        plt.tight_layout()
        return self.fig, self.cur_ax

    def set_cur_ax(self, new_ax):
        self.cur_ax = new_ax

    def reduction_bar_graph(self, data_high, data_low, xticklabels, legends, xlabel="", ylabel="",
            colors=None, edgecolors=None, xlim=(None,None), ylim=(None,None), legendorder=None,
            set_format=True, xtick_scale=1, title=None,
            with_borders=False,
            ):
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
                tick_starts *= xtick_scale
                rect = plt.bar(tick_starts, d,
                        edgecolor=edgecolors[i][j],
                        color=colors[i][j],
                        width=self.config.bar_width * xtick_scale)
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

        tick_locators = mpl.ticker.IndexLocator(
            base=bar_size*num_configs*3,
            # offset=-self.config.bar_interval/2 + self.config.bar_width,
            offset=-self.config.bar_interval/4,
            )

        if with_borders:
            for mt in tick_locators.tick_values(0, bar_size*num_configs*3*num_points):
                self.cur_ax.axvline(mt*xtick_scale - self.config.bar_width/3*2,
                        c='black', linestyle='-.',
                        )

        self.cur_ax.xaxis.set_major_locator(tick_locators)

        self.cur_ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(
            base=bar_size*num_configs,
            # offset=-self.config.bar_interval/2 + self.config.bar_width,
            offset=-self.config.bar_interval/4,
            ))

        if set_format:
            self.set_graph_general_format(xlim, ylim, xticklabels, xlabel, ylabel, title=title)

        if same_high_data:
            rects = [rects[0]] + rects[2:]
        self.cur_ax.legend(rects, legends, fontsize='small', ncol=num_legends)
        plt.tight_layout()
        return self.fig, self.cur_ax

    def save_to_file(self, name):
        for f in ['eps', 'png']:
            filename = f'./{f}/{name}.{f}'
            print("save to", filename)
            plt.savefig(filename, format=f'{f}')
