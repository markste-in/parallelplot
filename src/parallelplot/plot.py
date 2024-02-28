# Packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib as mpl
from collections.abc import Iterable

from parallelplot.cmaps import purple_blue

def plot(df, target_column = "", title = "", cmap = None):

    if not cmap:
        cmap = mpl.colormaps['hot']

    my_vars_names = df.columns.to_list()
    my_vars = my_vars_names


    # df_plot = df[my_vars]
    df_plot = df.dropna().reset_index(drop = True)

    if target_column:
        df_plot = df_plot.sort_values(by=target_column, ascending=False)

    # Convert to numeric matrix:
    ym = []
    dics_vars = []
    for v, var in enumerate(my_vars):
        if df_plot[var].dtype.kind not in ["i", "u", "f"]:
            dic_var = dict([(val, c) for c, val in enumerate(df_plot[var].unique())])
            dics_vars += [dic_var]
            ym += [[dic_var[i] for i in df_plot[var].tolist()]]
        else:
            ym += [df_plot[var].tolist()]
    ym = np.array(ym).T

    # Padding:
    ymins = ym.min(axis = 0)
    ymaxs = ym.max(axis = 0)
    dys = ymaxs - ymins
    ymins -= dys*0.05
    ymaxs += dys*0.05

    # Reverse some axes for better visual:
    axes_to_reverse = [1, 2]
    for a in axes_to_reverse:
        ymaxs[a], ymins[a] = ymins[a], ymaxs[a]
    dys = ymaxs - ymins

    # Adjust to the main axis:
    zs = np.zeros_like(ym)
    zs[:, 0] = ym[:, 0]
    zs[:, 1:] = (ym[:, 1:] - ymins[1:])/dys[1:]*dys[0] + ymins[0]

    range_min, range_max = df_plot[target_column].min(), df_plot[target_column].max()
    print(range_min,range_max)
    tick_label_size = 16

    with plt.style.context("dark_background"):
        # Plot:
        fig, host_ax = plt.subplots(
            figsize=(20, 10),
            tight_layout=True
        )

        # Make the axes:
        axes = [host_ax] + [host_ax.twinx() for i in range(ym.shape[1] - 1)]
        dic_count = 0
        for i, ax in enumerate(axes):
            ax.set_ylim(
                bottom=ymins[i],
                top=ymaxs[i]
            )
            ax.spines.top.set_visible(False)
            ax.spines.bottom.set_visible(False)
            ax.ticklabel_format(style='plain')
            if ax != host_ax:
                ax.spines.left.set_visible(False)
                ax.yaxis.set_ticks_position("right")
                ax.spines.right.set_position(
                    (
                        "axes",
                        i / (ym.shape[1] - 1)
                    )
                )
            if df_plot.iloc[:, i].dtype.kind not in ["i", "u", "f"]:
                dic_var_i = dics_vars[dic_count]
                ax.set_yticks(
                    range(len(dic_var_i)),
                )
                ax.set_yticklabels(
                    [key_val for key_val in dics_vars[dic_count].keys()],
                    fontsize=tick_label_size
                )
                dic_count += 1
            else:
                ax.tick_params(axis='y', labelsize=tick_label_size)
        host_ax.set_xlim(
            left=0,
            right=ym.shape[1] - 1
        )
        host_ax.set_xticks(
            range(ym.shape[1])
        )
        host_ax.set_xticklabels(
            my_vars_names,
            fontsize=tick_label_size
        )
        host_ax.tick_params(
            axis="x",
            which="major",
            pad=7,
        )

        # Make the curves:
        host_ax.spines.right.set_visible(False)
        host_ax.xaxis.tick_top()
        for j in range(ym.shape[0]):
            verts = list(zip([x for x in np.linspace(0, len(ym) - 1, len(ym) * 3 - 2,
                                                     endpoint=True)],
                             np.repeat(zs[j, :], 3)[1: -1]))
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            # color_first_cat_var = my_palette[dics_vars[0][df_plot.iloc[j, 0]]]
            cmap_value = ((df_plot.iloc[j, -1] - range_min) / range_max)
            print(cmap_value)
            patch = patches.PathPatch(
                path,
                facecolor="none",
                lw=3,
                alpha=0.3,
                edgecolor=cmap(cmap_value)
            )
            host_ax.add_patch(patch)

    if title:
        plt.title(title, fontsize=30)

    plot_color_gradients(cmap)

    return fig, axes

def plot_color_gradients(cmap_list):

    if not isinstance(cmap_list, list):
        cmap_list = [cmap_list]

    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
    fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

    #axs[0].set_title(f"{cmap_category} colormaps", fontsize=14)

    if not isinstance(axs, Iterable):
        axs = [axs]
    for i, (ax, cmap_entry) in enumerate(zip(axs, cmap_list)):
        if isinstance(cmap_entry, str):
            cmap=mpl.colormaps[cmap_entry]
            cmap_name = cmap_entry
        else:
            cmap = cmap_entry
            cmap_name = f"cmap {i}"
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.text(-.01, .5, cmap_name, va='center', ha='right', fontsize=10, color=cmap(5),
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()
    return fig, axs

data = np.random.random((100,4))
df = pd.DataFrame(data, columns=["a","b","c","d"])

plot(df=df, target_column = 'd', title="booooo")

plt.show()