# Packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from collections.abc import Iterable

from parallelplot.cmaps import purple_blue


def plot(df,
         target_column="",
         title="",
         cmap=None,
         figsize=None,
         title_font_size=18,
         tick_label_size=10,
         style=None,
         order="max",
         random_seed=None,
         alpha=0.3,
         lw=1):
    """
    Create a parallel coordinates plot from DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The data to plot
    target_column : str
        Column name to use for coloring
    title : str
        Title of the plot
    cmap : matplotlib.colors.Colormap
        Colormap to use (defaults to 'hot')
    figsize : tuple
        Figure size as (width, height) in inches
    title_font_size : int
        Font size for the title
    tick_label_size : int
        Font size for tick labels
    style : str, optional
        Matplotlib style to use. If "dark_background", will use dark background style.
        Default is None (uses standard matplotlib style)
    order : str, optional
        How to order the rows for plotting. Options:
        - "max": Sort by target column descending (default)
        - "min": Sort by target column ascending
        - "random": Shuffle rows randomly (reduces overplotting)
    random_seed : int, optional
        Random seed for reproducible randomization. Only used if order="random".

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : list
        List of axis objects
    """
    if not cmap:
        cmap = mpl.colormaps['hot']

    # Define formatter function to round floats to 3 decimal places
    def format_float(x, pos):
        if isinstance(x, (float, np.float64, np.float32)):
            return f"{x:.3f}"
        return str(x)
    
    float_formatter = FuncFormatter(format_float)

    # Drop NA values and reset index
    df_plot = df.dropna().reset_index(drop=True)
    
    # If target column is specified, move it to the end (last column)
    if target_column and target_column in df_plot.columns:
        # Get all columns and move target to end
        cols = df_plot.columns.tolist()
        cols.remove(target_column)
        cols = cols + [target_column]
        
        # Reorder the DataFrame
        df_plot = df_plot[cols]
        
        # Apply ordering based on the order parameter
        if order.lower() == "random":
            # Set random seed for reproducibility if provided
            if random_seed is not None:
                np.random.seed(random_seed)
            
            # Shuffle the entire DataFrame randomly
            df_plot = df_plot.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        elif order.lower() == "min":
            # Sort by target column ascending
            df_plot = df_plot.sort_values(by=target_column, ascending=True)
        
        elif order.lower() == "max":
            # Sort by target column descending
            df_plot = df_plot.sort_values(by=target_column, ascending=False)
        
        else:
            # Invalid order parameter, default to "max"
            print(f"Warning: Invalid order parameter '{order}'. Using 'max' instead.")
            df_plot = df_plot.sort_values(by=target_column, ascending=False)
    
    # Get column names after possible reordering
    my_vars_names = df_plot.columns.to_list()
    my_vars = my_vars_names

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
    ymins = ym.min(axis=0).astype(float)  # Convert to float to handle decimal padding
    ymaxs = ym.max(axis=0).astype(float)
    dys = ymaxs - ymins
    ymins -= dys * 0.05
    ymaxs += dys * 0.05

    # Reverse some axes for better visual:
    axes_to_reverse = [1, 2]
    for a in axes_to_reverse:
        if a < len(ymins):  # Ensure we don't go out of bounds
            ymaxs[a], ymins[a] = ymins[a], ymaxs[a]
    dys = ymaxs - ymins

    # Adjust to the main axis:
    zs = np.zeros_like(ym)
    zs[:, 0] = ym[:, 0]
    zs[:, 1:] = (ym[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    # Handle target column for coloring - create a mapping if it's a string column
    # If we have a target column, it's now always the last column
    target_idx = len(df_plot.columns) - 1 if target_column and target_column in df_plot.columns else 0
    
    if target_column and df_plot[target_column].dtype.kind in ["O", "S", "U"]:  # If target column is string/object
        unique_values = df_plot[target_column].unique()
        # Create a mapping from string values to numbers between 0 and 1
        target_map = {val: i / (len(unique_values) - 1 if len(unique_values) > 1 else 1)
                      for i, val in enumerate(sorted(unique_values))}
        # No need to define range_min and range_max for string columns
        print("Target column is string. Using sorted unique values for coloring.")
    else:
        # For numeric columns, continue using min and max
        if target_column and target_column in df_plot.columns:
            range_min, range_max = df_plot[target_column].min(), df_plot[target_column].max()
            print(f"{range_min:.3f}, {range_max:.3f}")
        else:
            # Default values if no target column
            range_min, range_max = 0, 1

    # Set up the plotting context with the appropriate style
    if style == "dark_background":
        # Use dark background style if specified
        context_manager = plt.style.context("dark_background")
    else:
        # Use a null context manager for default style
        context_manager = plt.style.context({})  # Empty dict means no style changes
        
    with context_manager:
        # Plot:
        fig, host_ax = plt.subplots(
            figsize=figsize if isinstance(figsize, tuple) else (10, 5),
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
            
            # Set the formatter to round floats to 3 decimal places
            ax.yaxis.set_major_formatter(float_formatter)
            
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
        
        # Modify labels to indicate target column
        x_labels = my_vars_names.copy()
        if target_column and target_column in df_plot.columns:
            x_labels[-1] = f"{x_labels[-1]}\n(Target)"
            
        host_ax.set_xticklabels(
            x_labels,
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

            # Calculate cmap_value differently based on data type
            if target_column and df_plot[target_column].dtype.kind in ["O", "S", "U"]:  # String column
                # Use the mapping for color
                curr_value = df_plot.iloc[j, target_idx]
                cmap_value = target_map[curr_value]
            elif target_column:  # Numeric column
                # Original calculation
                cmap_value = (df_plot.iloc[j, target_idx] - range_min) / (
                        range_max - range_min) if range_max > range_min else 0.5
            else:
                # Default value if no target column
                cmap_value = 0.5

            patch = patches.PathPatch(
                path,
                facecolor="none",
                lw=lw,
                alpha=alpha,
                edgecolor=cmap(cmap_value)
            )
            host_ax.add_patch(patch)

    if title:
        plt.title(title, fontsize=title_font_size)

    return fig, axes