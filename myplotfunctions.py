"""
myplotfunctions.py

A collection of plotting functions for visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def create_circle():
    """
    Create a circular path object centered at (0.5, 0.5) with radius 0.5.
    
    This function generates a matplotlib Path object representing a circle
    by computing points along a circular arc and combining them with a
    specified center and radius.
    
    Returns
    -------
    matplotlib.path.Path
        A Path object defining a circle centered at (0.5, 0.5) with radius 0.5.
        The circle is composed of 100 evenly-spaced points along the circumference.
    
    Examples
    --------
    >>> circle = create_circle()
    >>> circle.vertices.shape
    (100, 2)
    """
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    return circle

def modify_map(ax, lat_north = -50):
    circle = create_circle()
    ax.set_extent([-180, 180, -90, lat_north], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, zorder=1, color = 'grey')
    # ax.add_feature(cfeature.COASTLINE, linewidth = 0.5)
    ax.add_feature(cfeature.OCEAN, alpha = 0.15)
    ax.set_boundary(circle, transform=ax.transAxes)
    # ax.spines['geo'].set_linewidth(0.5)
    ax.spines['geo'].set_edgecolor(None)
    # ax.gridlines(draw_labels=False, 
    #              ylocs=np.linspace(-90, 90, 7), 
    #              color = 'grey', linestyle = '-.', linewidth = 0.5, alpha = 0.8)

def plot_lagged_correlation(ax, x, y, max_lag, p_limit=0.05, color='b', label=None, lw=1.5, pls = ":", al = 0.5):
    from scipy import stats

    x_norm = (x - np.mean(x)) / np.std(x.astype(np.float64))
    y_norm = (y - np.mean(y)) / np.std(y.astype(np.float64))

    x_slice = x_norm[max_lag : len(x_norm) - max_lag]
    window_len = len(x_slice)
    
    correlations = []
    p_values = []

    for i in range(2 * max_lag+1):
        y_slice = y_norm[i : i + window_len]
        reg = stats.linregress(x_slice, y_slice)
        correlations.append(reg.rvalue)
        p_values.append(reg.pvalue)

    correlations = np.array(correlations)
    p_values = np.array(p_values)
    lags = np.arange(-max_lag, max_lag+1)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    ax.plot(lags, correlations, color=color, linestyle=pls, linewidth=lw, alpha=al)
    sig_mask = np.where(p_values < p_limit, correlations, np.nan)
    ax.plot(lags, sig_mask, color=color, linestyle='-', linewidth=lw, label=label)

def mask_time_range(p1, p2, ds):
    if "time" in ds.dims:
        ds_masked = ds.isel({"time": slice(p1, p2)})
    elif "year" in ds.dims:
        ds_masked = ds.isel({"year": slice(p1, p2)})
    return ds_masked

def plot_multi_lagged_correlation(
        max_lag, lag_center, lag_data_dict, 
        ax = None, fig_size=(8, 5), pls = ':',
        title=None, set_xlim=None, set_ylim=None, 
        colors=None, legend_col=1, lg_out=False,
        lw = 1.5, xl = True, mask=None,
        savename=None, 
):
    if ax is None:
        fig = plt.figure(figsize = fig_size) # Creates the figure 
        ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, which="both", ls="-", color='0.65', alpha=0.5)
    if colors is not None:
        colors_list = colors
    else:
        colors_list = plt.cm.tab10.colors
    i = 0
    if mask is not None:
        lag_center = mask_time_range(mask['p1'], mask['p2'], lag_center)
    for data_label, lag_data in lag_data_dict.items():
        if mask is not None:
            lag_data = mask_time_range(mask['p1'], mask['p2'], lag_data)
        plot_lagged_correlation(ax, lag_center, lag_data, max_lag, 
                                color=colors_list[i], label=data_label, lw=lw, pls=pls)
        i+=1
    # plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    if xl:
        ax.set_xlabel('lag (year)')
    ax.set_ylabel('correlation coefficient (r)')
    if title is not None:
        ax.set_title(title)
    if set_xlim is not None:
        ax.set_xlim(set_xlim[0], set_xlim[1])
    if set_ylim is not None:
        ax.set_ylim(set_ylim[0], set_ylim[1])
    if lg_out:
        ax.legend(bbox_to_anchor=(1.005, 1), loc='upper left',frameon=False)
    else:
        ax.legend(frameon=False, ncol = legend_col)
    if savename is not None:
        fig.savefig(savename, format="pdf")