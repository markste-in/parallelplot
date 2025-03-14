from matplotlib.colors import LinearSegmentedColormap

# Number of color levels
n_levels = 256

purple_blue_colors = ["#F41EF3", "#1E3CF4"]
purple_blue = LinearSegmentedColormap.from_list("my_palette", purple_blue_colors, N=n_levels)
