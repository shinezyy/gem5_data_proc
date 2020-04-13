import numpy as np
import matplotlib.pyplot as plt

from omegaflow_model.color_rotater import ColorRotater


def main():
    rate_low = 3.1
    rate_high = 6.0

    def low_func(x):
        return rate_low/x

    def high_func(x):
        return rate_high/x

    fig, ax = plt.subplots()
    colors = ColorRotater()

    step = 0.01

    ipc_max = 4.0
    ipc_cross_low_x = low_func(ipc_max)
    ipc_cross_high_x = high_func(ipc_max)

    right_lim = 8.0

    leftest_xs = np.arange(1e-10, ipc_cross_low_x + step, step)
    right_xs = np.arange(ipc_cross_low_x, right_lim, step)
    mid_xs = np.arange(ipc_cross_low_x, ipc_cross_high_x, step)
    left_xs = np.arange(1e-10, ipc_cross_high_x, step)
    rightest_xs = np.arange(ipc_cross_high_x, right_lim, step)

    # fill the leftest rectangle
    ax.fill_between(leftest_xs, ipc_max, color=colors.get())

    low_curve_bottom = low_func(right_xs)
    low_curve_upper = low_func(leftest_xs)

    # fill between x axis and the low curve
    ax.fill_between(right_xs, low_curve_bottom, color=colors.last())
    # draw low curve
    obj, = ax.plot(right_xs, low_curve_bottom, marker=',', color=colors.get())
    obj, = ax.plot(leftest_xs, low_curve_upper, marker=',', color=colors.get())

    high_curve_bottom = high_func(rightest_xs)
    high_curve_upper = high_func(left_xs)

    # fill between the low curve and the high curve
    high_curve_bottom_filler = high_func(right_xs)
    ipc_max_filler = [ipc_max] * len(high_curve_bottom_filler)
    high_curve_bottom_filler = np.min((high_curve_bottom_filler, ipc_max_filler), axis=0)
    ax.fill_between(right_xs, low_curve_bottom, high_curve_bottom_filler, color=colors.get())
    # draw high curve
    obj, = ax.plot(rightest_xs, high_curve_bottom, marker=',', color=colors.get())
    obj, = ax.plot(left_xs, high_curve_upper, marker=',', color=colors.get())

    # fille between the hight curve and the horizontal line
    ax.fill_between(rightest_xs, high_curve_bottom, ipc_max, color=colors.get())

    ax.set_xlim([0, right_lim])
    ax.set_ylim([0, ipc_max])
    plt.show()


if __name__ == '__main__':
    main()

