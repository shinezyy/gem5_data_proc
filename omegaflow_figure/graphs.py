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
    # colors = ["red", "gray", "purple", "None", "green", "orange"]
    colors = plt.get_cmap('Dark2').colors
    edgecolor = "black"
    edgecolors = ["black", "blue"]
    fig_size = (14, 3)
    bar_width, bar_interval = 0.6, 0.4

class GraphMaker(object):
    def __init__(self, fig_size=None, multi_ax=False, legend_loc=None,
            with_xtick_label=True, frameon=True, *args, **kwargs):
        self.config = GraphDefaultConfig()
        if fig_size is None:
            fig_size = self.config.fig_size
        if multi_ax:
            self.fig, self.ax = plt.subplots(*args, **kwargs)
        else:
            self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(fig_size[0], fig_size[1], forward=True)
        self.cur_ax = self.ax
        if legend_loc is None:
            self.loc = 'upper left'
        else:
            self.loc = legend_loc
        self.frameon = frameon

        self.with_xtick_label = with_xtick_label

    def compute_legend(self, n):
        if n <= 2:
            return n
        elif n == 4:
            return 2
        else:
            return int(np.ceil(n/2))

    def set_graph_general_format(self, xlim, ylim, xticklabels, xlabel, ylabel,
            title=None):
        # self.cur_ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
        # cur_ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # print(self.cur_ax.xaxis.get_major_ticks())
        # for tick in self.cur_ax.xaxis.get_major_ticks():
        #     tick.tick1line.set_markersize(2)
        #     tick.tick2line.set_markersize(0)
        #     # tick.tick2line.set_fontsize(14)
        # for tick in self.cur_ax.xaxis.get_minor_ticks():
        #     tick.tick1line.set_markersize(2)
        #     tick.label.set_fontsize(14)
        #     # tick.tick2line.set_markersize(0)
        #     tick.label1.set_horizontalalignment('center')

        self.cur_ax.set_xlim(xlim)
        self.cur_ax.set_ylim(ylim)
        if self.with_xtick_label:
            print(xticklabels)
            # self.cur_ax.set_xticklabels(xticklabels, minor=True, rotation=0)
            self.cur_ax.set_xticklabels(xticklabels, rotation=90, fontsize=14)
        self.cur_ax.grid(axis="y", linestyle="--", color='gray', alpha=0.3)
        self.cur_ax.set_xlabel(xlabel)
        self.cur_ax.set_ylabel(ylabel, fontsize=14)
        if title is not None:
            self.cur_ax.title.set_text(title)
            self.cur_ax.title.set_fontsize(14)

    def simple_bar_graph(
            self, data, xticklabels, legends, xlabel="", ylabel="",
            colors=None, edgecolor=None, linewidth=None,
            xlim=(None,None), ylim=(None,None), overlap=False,
            set_format=True, xtick_scale=1, title=None, with_borders=False,
            markers=None,
            dont_legend=False,
            show_tick=False,
            ):
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
        rects = []
        bar_size = self.config.bar_width + self.config.bar_interval
        assert markers is not None
        for i, d in enumerate(data):
            tick_starts = np.arange(0, num_points, bar_size)
            tick_starts *= xtick_scale
            rect, = plt.plot(tick_starts, d,
                    marker=markers[i],
                    # edgecolor=edgecolor, linewidth=linewidth,
                    color=colors[i],
                    # width=self.config.bar_width * xtick_scale
                    )
            rects.append(rect)

        self.cur_ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(
            base=bar_size*num_configs*3, offset=-self.config.bar_interval/2))
        # self.cur_ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(
        #     base=bar_size*num_configs, offset=-self.config.bar_interval/2))

        tick_locators = mpl.ticker.IndexLocator(
            base=bar_size,
            # offset=-self.config.bar_interval/2 + self.config.bar_width,
            offset=-self.config.bar_interval/4,
        )

        # if with_borders:
        #     for mt in tick_locators.tick_values(0, bar_size*num_configs*3*num_points):
        #         self.cur_ax.axvline(mt*xtick_scale - self.config.bar_width/3*2,
        #                 c='black', linestyle='-.',
        #                 )

        if show_tick:
            self.cur_ax.xaxis.set_major_locator(tick_locators)
        else:
            for tick in self.cur_ax.xaxis.get_major_ticks():
                tick.tick1line.set_markersize(0)
                tick.label.set_fontsize(0)
                tick.tick2line.set_markersize(0)

        # self.cur_ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(
        #     base=bar_size,
        #     # offset=-self.config.bar_interval/2 + self.config.bar_width,
        #     offset=-self.config.bar_interval/4,
        #     ))


        if set_format:
            self.set_graph_general_format(xlim, ylim, xticklabels, xlabel, ylabel, title=title)

        if not dont_legend:
            print(rects, legends)
            self.cur_ax.legend(rects, legends, fontsize=13,
                    ncol=self.compute_legend(num_configs),
                    loc=self.loc,
                    frameon=self.frameon,
                    fancybox=True,
                    framealpha=0.5,
                    )
        # plt.tight_layout()
        return self.fig, self.cur_ax

    def set_cur_ax(self, new_ax):
        self.cur_ax = new_ax

    def reduction_bar_graph(
            self, data_high, data_low, xticklabels, legends, xlabel="", ylabel="",
            colors=None, edgecolors=None, xlim=(None,None), ylim=(None,None), legendorder=None,
            set_format=True, xtick_scale=1, title=None,
            with_borders=False,
            markers=None,
            redundant_baseline=False,
            dont_legend=False,
            show_tick=False,
            ):
        assert colors is not None

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
            print(legends, num_legends)
            assert(len(legends) == num_legends)
        same_high_data = redundant_baseline

        print(num_points, num_configs)
        rects = []

        bar_size = self.config.bar_width + self.config.bar_interval

        for i, data in enumerate([data_low, data_high]):
            for j, d in enumerate(data):
                if same_high_data and i > 0 and j > 0:
                    continue
                tick_starts = np.arange(0, num_points, bar_size)
                tick_starts *= xtick_scale
                # rect = plt.bar(tick_starts, d,
                #         edgecolor=edgecolors[i][j],
                #         color=colors[i][j],
                #         width=self.config.bar_width * xtick_scale)

                rect, = plt.plot(tick_starts, d,
                        marker=markers[i][j],
                        # edgecolor=edgecolor, linewidth=linewidth,
                        color=colors[i][j],
                        # width=self.config.bar_width * xtick_scale
                        )
                rects.append(rect)

        rects = rects[2:] + rects[:2]
        if legendorder is not None:
            new_rects, new_legends = [], []
            for i in legendorder:
                new_rects.append(rects[i])
                new_legends.append(legends[i])
            rects = new_rects
            legends = new_legends

        tick_locators = mpl.ticker.IndexLocator(
            base=bar_size,
            offset=-self.config.bar_interval/4,
        )
        if show_tick:
            self.cur_ax.xaxis.set_major_locator(tick_locators)
        else:
            for tick in self.cur_ax.xaxis.get_major_ticks():
                tick.tick1line.set_markersize(0)
                tick.label.set_fontsize(0)
                tick.tick2line.set_markersize(0)

        #
        # if with_borders:
        #     for mt in tick_locators.tick_values(0, bar_size*num_configs*3*num_points):
        #         self.cur_ax.axvline(mt*xtick_scale - self.config.bar_width/3*2,
        #                 c='black', linestyle='-.',
        #                 )
        # self.cur_ax.xaxis.set_major_locator(tick_locators)

        # self.cur_ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(
        #     base=bar_size,
        #     # offset=-self.config.bar_interval/2 + self.config.bar_width,
        #     offset=-self.config.bar_interval/4,
        #     ))

        if set_format:
            self.set_graph_general_format(xlim, ylim, xticklabels, xlabel, ylabel, title=title)

        if same_high_data:
            legends = [legends[0]] + legends[2:]

        if not dont_legend:
            self.cur_ax.legend(rects, legends, fontsize=13,
                    ncol=self.compute_legend(num_legends),
                    loc=self.loc,
                    frameon=self.frameon,
                    fancybox=True,
                    framealpha=0.5,
                    )
        # plt.tight_layout()
        return self.fig, self.cur_ax

    def save_to_file(self, name):
        for f in ['eps', 'png', 'pdf']:
            d = f
            # if f == 'pdf':
            #     d = 'eps'
            filename = f'./{d}/{name}.{f}'
            print("save to", filename)
            plt.savefig(filename, format=f'{f}')
