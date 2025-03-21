# Packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from collections.abc import Iterable
from typing import List, Dict, Tuple, Optional, Union, Any

from parallelplot.cmaps import purple_blue


def plot(df: pd.DataFrame,
         target_column: str = "",
         title: str = "",
         cmap: Optional[mpl.colors.Colormap] = None,
         figsize: Optional[Tuple[float, float]] = None,
         title_font_size: int = 18,
         tick_label_size: int = 10,
         style: Optional[str] = None,
         order: str = "max",
         random_seed: Optional[int] = None,
         alpha: float = 0.3,
         lw: float = 1,
         axes_to_reverse: Optional[List[int]] = [],
         hide_axes: bool = False) -> Tuple[plt.Figure, List[plt.Axes]]:
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
    alpha : float
        Transparency level for the lines (0 to 1)
    lw : float
        Line width for the parallel coordinate lines
    axes_to_reverse: list
        Axes with reversed order
    hide_axes: bool
        Whether to hide all axes, their spines, ticks, and labels

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : list
        List of axis objects
    """
    # Set default colormap if none provided
    if not cmap:
        cmap = mpl.colormaps['hot']

    # Define formatter function to round floats to 2 decimal places
    def format_float(x: Union[float, int], decimal_places: int = 2) -> str:
        """Format axis ticks: round floats to specified decimal places, keep integers untouched, and use scientific notation for small or large numbers."""
        if isinstance(x, (float, np.float64, np.float32)):
            if abs(x) >= 1e5 or abs(x) < 1e-3:  # Use scientific notation for large or small numbers
                return f"{x:.1e}"
            return f"{x:.{decimal_places}f}".rstrip('0').rstrip('.')  # Round floats and remove trailing zeros
        return str(x)  # Keep integers as they are

    
    # Create formatter function for axis labels
    float_formatter = FuncFormatter(format_float)

    # Drop NA values and reset index to ensure clean data
    df_plot = df.dropna().reset_index(drop=True)
    
    # If target column is specified, move it to the end (last column)
    if target_column and target_column in df_plot.columns:
        # Get all columns and move target to end
        cols = df_plot.columns.tolist()
        cols.remove(target_column)
        cols = cols + [target_column]
        
        # Reorder the DataFrame with target column at the end
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
    column_names = df_plot.columns.to_list()
    columns = column_names  # Using more descriptive variable name

    # Convert DataFrame to normalized numeric matrix for plotting
    # This step handles categorical columns by converting them to numeric indices
    normalized_data = []  # Will hold the numeric values for each column
    categorical_mappings = []  # Will store mappings of categorical values to numeric indices
    
    for column_idx, column_name in enumerate(columns):
        # Check if column is categorical (non-numeric)
        if df_plot[column_name].dtype.kind not in ["i", "u", "f"]:
            # Create a mapping dictionary from categorical values to numeric indices
            # Example: {'Low': 0, 'Medium': 1, 'High': 2}
            categorical_mapping = dict([(value, index) for index, value in enumerate(df_plot[column_name].unique())])
            categorical_mappings.append(categorical_mapping)
            
            # Convert categorical values to numeric using the mapping
            # For each value in the column, look up its numeric index in the mapping
            numeric_values = [categorical_mapping[value] for value in df_plot[column_name].tolist()]
            normalized_data.append(numeric_values)
        else:
            # For numeric columns, just use the values directly
            normalized_data.append(df_plot[column_name].tolist())
    
    # Transpose the data so rows are observations and columns are variables
    # Shape changes from [n_columns, n_rows] to [n_rows, n_columns]
    normalized_data_matrix = np.array(normalized_data).T

    # Calculate padding for each axis to add visual space
    # This adds 5% padding on both ends of each axis
    axis_mins = normalized_data_matrix.min(axis=0).astype(float)
    axis_maxs = normalized_data_matrix.max(axis=0).astype(float)
    axis_ranges = axis_maxs - axis_mins
    
    # Add padding (5% of the range on each end)
    axis_mins -= axis_ranges * 0.05
    axis_maxs += axis_ranges * 0.05
    
    # Handle columns with only one unique value
    for i in range(len(axis_mins)):
        if abs(axis_mins[i] - axis_maxs[i]) < 1e-10:  # If min and max are effectively equal
            if axis_mins[i] == 0:
                # If the value is zero, set limits to -0.5 and 0.5
                axis_mins[i] = -0.5
                axis_maxs[i] = 0.5
            else:
                # Otherwise, add +/- 10% of the value
                value = axis_mins[i]
                axis_mins[i] = value - abs(value) * 0.1
                axis_maxs[i] = value + abs(value) * 0.1

    # Reverse specific axes for better visual appearance
    # This is a design choice for certain types of data
    for axis_idx in axes_to_reverse:
        if axis_idx < len(axis_mins):  # Ensure we don't go out of bounds
            axis_maxs[axis_idx], axis_mins[axis_idx] = axis_mins[axis_idx], axis_maxs[axis_idx]
    
    # Recalculate ranges after potential reversals
    axis_ranges = axis_maxs - axis_mins

    # Normalize all variables to the scale of the first variable for consistent plotting
    # This ensures all variables use the same scale on the plot
    scaled_data = np.zeros_like(normalized_data_matrix)
    scaled_data[:, 0] = normalized_data_matrix[:, 0]  # First column remains unchanged
    
    # Scale all other columns to match the range of the first column
    for col in range(1, normalized_data_matrix.shape[1]):
        # Formula: ((value - min) / range) * range_of_first_column + min_of_first_column
        scaled_data[:, col] = (
            (normalized_data_matrix[:, col] - axis_mins[col]) / axis_ranges[col] * axis_ranges[0] + axis_mins[0]
        )

    # Handle target column for coloring - create a mapping if it's a string column
    # If we have a target column, it's now always the last column
    target_idx = len(df_plot.columns) - 1 if target_column and target_column in df_plot.columns else 0
    
    if target_column and df_plot[target_column].dtype.kind in ["O", "S", "U"]:  # If target column is string/object
        unique_values = df_plot[target_column].unique()
        # Create a mapping from string values to numbers between 0 and 1 for coloring
        target_map = {
            val: i / (len(unique_values) - 1 if len(unique_values) > 1 else 1)
            for i, val in enumerate(sorted(unique_values))
        }
        print("Target column is string. Using sorted unique values for coloring.")
    else:
        # For numeric columns, calculate min and max for color scaling
        if target_column and target_column in df_plot.columns:
            range_min, range_max = df_plot[target_column].min(), df_plot[target_column].max()
            # print(f"{range_min:.3f}, {range_max:.3f}")
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
        # Create the main figure and host axis
        fig, host_ax = plt.subplots(
            figsize=figsize if isinstance(figsize, tuple) else (10, 5),
            tight_layout=True
        )

        # Create parallel axes - one for each column
        # First axis is the host_ax, additional axes are created with twinx()
        axes = [host_ax] + [host_ax.twinx() for i in range(normalized_data_matrix.shape[1] - 1)]
        categorical_mapping_idx = 0  # Counter for tracking which categorical mapping to use
        
        # Configure each axis
        for i, ax in enumerate(axes):
            # Set the y-limits for this axis
            ax.set_ylim(
                bottom=axis_mins[i],
                top=axis_maxs[i]
            )
            
            # Remove top and bottom spines for cleaner appearance
            ax.spines.top.set_visible(False)
            ax.spines.bottom.set_visible(False)
            
            # Set the formatter to round floats to 3 decimal places
            ax.yaxis.set_major_formatter(float_formatter)
            
            # Configure secondary axes (all except the first)
            if ax != host_ax:
                ax.spines.left.set_visible(False)
                ax.yaxis.set_ticks_position("right")
                
                # Position the axis at the appropriate x location
                ax.spines.right.set_position(
                    (
                        "axes",  # Position relative to axes coordinates
                        i / (normalized_data_matrix.shape[1] - 1)  # Normalized position (0 to 1)
                    )
                )
            
            # Handle categorical variables differently for tick labels
            if df_plot.iloc[:, i].dtype.kind not in ["i", "u", "f"]:
                # For categorical columns, use the category names as tick labels
                current_mapping = categorical_mappings[categorical_mapping_idx]
                ax.set_yticks(
                    range(len(current_mapping)),  # Position ticks at integer positions
                )
                ax.set_yticklabels(
                    [category_name for category_name in current_mapping.keys()],  # Use category names
                    fontsize=tick_label_size
                )
                categorical_mapping_idx += 1
            else:
                # For numeric columns, just set the font size
                ax.tick_params(axis='y', labelsize=tick_label_size)
        
        # Configure x-axis limits and ticks
        host_ax.set_xlim(
            left=0,
            right=normalized_data_matrix.shape[1] - 1
        )
        host_ax.set_xticks(
            range(normalized_data_matrix.shape[1])
        )
        
        # Modify column labels to indicate target column
        x_labels = column_names.copy()
        if target_column and target_column in df_plot.columns:
            x_labels[-1] = f"{x_labels[-1]}\n(Target)"
            
        host_ax.set_xticklabels(
            x_labels,
            fontsize=tick_label_size
        )
        host_ax.tick_params(
            axis="x",
            which="major",
            pad=7,  # Add padding below tick labels
        )

        # Configure host axis appearance
        host_ax.spines.right.set_visible(False)
        host_ax.xaxis.tick_top()  # Move x-axis ticks to the top
        
        # Hide all axes if requested
        if hide_axes:
            # Hide all tick marks and labels on the host axis
            host_ax.set_xticks([])
            host_ax.set_yticks([])
            host_ax.set_xticklabels([])
            host_ax.set_yticklabels([])
            
            # Hide all spines on the host axis
            for spine in host_ax.spines.values():
                spine.set_visible(False)
            
            # Hide all tick marks, labels, and spines on the other axes
            for ax in axes[1:]:  # Skip host_ax since we already processed it
                ax.set_yticks([])
                ax.set_yticklabels([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # Draw the parallel coordinate lines
        for row_idx in range(normalized_data_matrix.shape[0]):
            # Create smooth curves using Bezier paths
            # Generate x-coordinates for the path with 3x density for smooth curves
            x_coords = np.linspace(0, len(normalized_data_matrix) - 1, len(normalized_data_matrix) * 3 - 2, endpoint=True)
            
            # Repeat each y-coordinate 3 times and slice to match the x-coordinates
            y_coords = np.repeat(scaled_data[row_idx, :], 3)[1: -1]
            
            # Create vertex list as tuples of (x,y) coordinates
            vertices = list(zip(x_coords, y_coords))
            
            # Path codes: MOVETO for first point, CURVE4 for all others to create Bezier splines
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(vertices) - 1)]
            path = Path(vertices, codes)

            # Calculate color value based on target column
            if target_column and df_plot[target_column].dtype.kind in ["O", "S", "U"]:  # String column
                # Use the mapping for categorical target color
                current_value = df_plot.iloc[row_idx, target_idx]
                color_value = target_map[current_value]
            elif target_column:  # Numeric target column
                # Normalize value between 0 and 1 for colormap
                color_value = (df_plot.iloc[row_idx, target_idx] - range_min) / (
                        range_max - range_min) if range_max > range_min else 0.5
            else:
                # Default value if no target column
                color_value = 0.5

            # Create and add the path to the plot
            line = patches.PathPatch(
                path,
                facecolor="none",  # No fill
                lw=lw,  # Line width
                alpha=alpha,  # Transparency
                edgecolor=cmap(color_value)  # Color based on target value
            )
            host_ax.add_patch(line)

    # Add title if provided
    if title:
        plt.title(title, fontsize=title_font_size)

    return fig, axes