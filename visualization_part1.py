"""
PlotRoss.py
Module for visualizing Ross Sea convection related results.
"""
import xarray as xr
import numpy as np
import gsw
from myfunctions import openpickle, running_mean, ReadDataFromNCAR, normalize_timeseries, connect_dask_cluster

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from myplotfunctions import plot_multi_lagged_correlation, modify_map, mask_time_range

from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_rs_lag(data_center, 
                ross_lag_cor_ann_dict1, 
                ross_lag_cor_ann_dict2, 
                mask = None, pls =':',
                sfname="RossCrossLag.pdf"):
    max_lag = 60
    fig = plt.figure(figsize = (6,5.5))
    ax1 = fig.add_subplot(2,1,1)
    plt.subplots_adjust(left=0.125,
                        bottom=0.08, 
                        right=0.75, 
                        top=0.955, 
                        wspace=0.04, 
                        hspace=0.10)
    
    plot_multi_lagged_correlation(
        max_lag, data_center, ross_lag_cor_ann_dict1, ax1,
        xl = False,
        title="time-lagged cross-correlation with Ross convection index",
        lg_out=True,
        colors=plt.cm.tab10.colors, pls=pls,
        set_xlim=(-50, 50), mask=mask,
        # set_ylim=(-1, 1)
    )
    ax1.text(-47,0.85,"a.", fontsize=10, fontweight='bold')
    ax2 = fig.add_subplot(2,1,2)
    
    plot_multi_lagged_correlation(
        max_lag, data_center, ross_lag_cor_ann_dict2, ax2,
        # title="Properties (ann) Lagged Correlation with Ross convection index",
        lg_out=True,
        colors=plt.cm.tab10.colors, pls=pls,
        set_xlim=(-50, 50), mask=mask
        # set_ylim=(-1, 1)
    )
    ax2.text(-47,0.82,"b.", fontsize=10, fontweight='bold')
    if sfname is not None:
        fig.savefig(sfname, format="pdf")

def add_timeseries_and_modify(ax, pltx, pdict, fs=8):
    yb = pdict['ybounds']
    defaults = {"ls": "-", "lw": 0.8, "al": 0.75, 
                "yticks": [yb[0], (yb[0]+yb[1])/2, yb[1]]}
    pld = defaults | pdict

    p2, = ax.plot(
        pltx, pld['data'], color = pld['color'], 
        linestyle = pld['ls'], linewidth = pld['lw'], alpha = pld['al']
    )
    ax.spines[:].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='y', colors=p2.get_color())#, labelrotation=90)
    ax.set_ylabel(pld['ylabel'], fontsize = fs, color = p2.get_color())
    ax.yaxis.set_label_coords(pld['ylcoords'][0], pld['ylcoords'][1])
    twinyticks = ax.get_yticklabels()
    for ytl in twinyticks:
        ytl.set_verticalalignment('center')
    ax.set_yticks(pld['yticks'])
    ax.set_ylim(pld['ylim'][0], pld['ylim'][1])
    offax_set = pld['offax']
    
    if offax_set<0:
        ax.spines.left.set_visible(True)
        ax.spines.left.set_color(p2.get_color())
        ax.spines.left.set_position(("axes", offax_set))
        ax.tick_params(left=True, labelleft=True, right=False, labelright=False)
        ax.yaxis.set_label_position("left")
        ax.spines.left.set_bounds(yb[0], yb[1])
    else:
        ax.spines.right.set_visible(True)
        ax.spines.right.set_color(p2.get_color())
        if offax_set > 0:
            ax.spines.right.set_position(("axes", offax_set))
        ax.tick_params(left=False, labelleft=False, right=True, labelright=True)
        ax.yaxis.set_label_position("right")
        ax.spines.right.set_bounds(yb[0], yb[1])

def add_vspans(ax, vs, vl, c1='0.9', c2='darkgrey') -> None:
    for vi in vs:
        ax.axvspan(vi[0], vi[1], color=c1)
    for vli in vl:
        ax.axvline(vli, color=c2, linestyle='--', linewidth=0.75)

def add_profile_ts(fig, ax, pxs, profile_data_dict, mfs):
    defaults = {"cmap": cmocean.cm.thermal, "zname": "lev", "ylabel": "depth (m)"}
    cfg = defaults | profile_data_dict 
    pdata = cfg['data']
    ys = pdata[cfg['zname']]
    m = ax.contourf(
        pxs, ys, 
        pdata.transpose(), 
        cmap = cfg['cmap']
    )
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="3%", pad="2%")
    cbar = fig.colorbar(m, cax=cax1)
    cbar.ax.tick_params(labelsize=mfs) 
    cbar.outline.set_visible(False)
    cbar.set_label(cfg['clabel'], fontsize = mfs)
    ax.spines[:].set_visible(False)
    ax.set_ylim(0, 5100)
    ax.invert_yaxis()
    ax.set_ylabel(cfg['ylabel'], fontsize = mfs)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.tick_params(axis='both', which='major', labelsize=mfs)
    return divider

def plot_rs_ts(
    profile_data_dict, ts_data_dict, 
    plt_xs = range(1, 501), xlim = (-5, 505),
    mfs=8, fsize=(6,6), mask=None,
    conv_phase = [(81, 90), (180, 193), (292, 300), (386, 403)],
    conv_peak = [86, 184, 295, 398], 
    figname="RossTimeseries.pdf"
) -> None:
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1,1,1)
    fig.subplots_adjust(bottom=0.02, top=0.93, left = 0.135, right = 0.89, hspace=0.1)

    divider = add_profile_ts(fig, ax, plt_xs, profile_data_dict, mfs)
    ax.plot(plt_xs, ts_data_dict['mld']['data'], 
        color = ts_data_dict['mld']['color'], linestyle = "-", linewidth = 1.2)
    ax.set_xlim(xlim[0], xlim[1])
    
    ax0 = divider.append_axes("top", 3, pad=0, sharex=ax)
    add_vspans(ax0, conv_phase, conv_peak, c1='wheat', c2='grey')

    add_timeseries_and_modify(ax0, plt_xs, ts_data_dict['sic'], mfs)
    
    ax0.xaxis.set_label_position("top")
    ax0.set_xlabel("year", fontsize = 8)
    ax0.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    
    twin2 = ax0.twinx()
    add_timeseries_and_modify(twin2, plt_xs, ts_data_dict['hfds'], mfs)
    twin1 = ax0.twinx()
    add_timeseries_and_modify(twin1, plt_xs, ts_data_dict['thick'], mfs)
    twin3 = ax0.twinx()
    add_timeseries_and_modify(twin3, plt_xs, ts_data_dict['gyre'], mfs)
    twin4 = ax0.twinx()
    add_timeseries_and_modify(twin4, plt_xs, ts_data_dict['heat'], mfs)
    
    if mask is not None:
        mask_default = {'color': '0.6', 'alpha': 0.7, 'start': plt_xs[0], 'end': plt_xs[-1]}
        msk = mask_default | mask
        ax0.axvspan(msk['start'], msk['p1'], color=msk['color'], alpha=msk['alpha'])
        ax0.axvspan(msk['p2'], msk['end'], color=msk['color'], alpha=msk['alpha'])
        ax.axvspan(msk['start'], msk['p1'], color=msk['color'], alpha=msk['alpha'])
        ax.axvspan(mask['p2'], msk['end'], color=msk['color'], alpha=msk['alpha'])
    if figname is not None:
        fig.savefig(figname, format="pdf")

def plot_region_box(ax, lons, lats, c, lw=1.0, n=50, tf=ccrs.PlateCarree()) -> None:
    lon_top = np.linspace(lons[0], lons[1], n)
    lat_top = np.full(n, lats[1])
    lon_bottom = np.linspace(lons[0], lons[1], n)
    lat_bottom = np.full(n, lats[0])
    lat_left = np.linspace(lats[0], lats[1], n)
    lon_left = np.full(n, lons[0])
    lat_right = np.linspace(lats[0], lats[1], n)
    lon_right = np.full(n, lons[1])
    ax.plot(lon_top, lat_top, transform=tf, color=c, linewidth=lw)
    ax.plot(lon_bottom, lat_bottom, transform=tf, color=c, linewidth=lw)
    ax.plot(lon_left, lat_left, transform=tf, color=c, linewidth=lw)
    ax.plot(lon_right, lat_right, transform=tf, color=c, linewidth=lw)

def plot_region_map(ax, Psi_plot, conv, lons, lats):
    ax.add_feature(cfeature.LAND, facecolor='darkgray')
    gl = ax.gridlines(
        xlocs=np.arange(-180, 180, 20),
        ylocs=np.arange(-85, 85, 5),
        draw_labels=True
    )
    gl.bottom_labels = True
    gl.right_labels = True
    gl.left_labels = False
    gl.top_labels = False

    im = ax.pcolormesh(Psi_plot.lon, Psi_plot.lat, Psi_plot, vmin=-50, vmax=50,
                transform=ccrs.PlateCarree(), cmap = cmocean.cm.delta)
    ax.set_extent([lons[0], lons[1], lats[0], lats[1]], ccrs.PlateCarree())
    ax.contour(Psi_plot.lon, Psi_plot.lat, Psi_plot, linewidths = 1,
                levels = [-20], colors = 'purple', linestyles='solid',
                transform=ccrs.PlateCarree())
    conv_plt = xr.where(conv>0, 1.0, 0.0)
    ax.contour(conv.lon, conv.lat, conv_plt, 
               colors = 'coral', linewidths = 0.2, #alpha = 0.7,
               transform=ccrs.PlateCarree())
    c1 = ax.contourf(conv.lon, conv.lat, conv_plt, 
                levels=[0.5, 1.5],  colors='none', 
                hatches=['\\\\\\\\'], transform=ccrs.PlateCarree())
    c1._hatch_color = (1, .4, .4, 1.0)
    return im

def add_arrows(ax,xy1,xy2,c):
    ax.annotate(
        "", xytext=xy1, xy=xy2, transform=ccrs.PlateCarree(), 
        arrowprops=dict(arrowstyle="-|>", color = c)
    )

def plot_study_area(
    depth_ocean, Psi_SO_avg, conv_ws, conv_rs, dspolynya,
    figname="StudyAreaMaps.png",
    ws_lats = [-80, -50], ws_lons = [-60, 60],
    rs_lats = [-78, -58], rs_lons = [-205, -125]
) -> None:
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2,2, width_ratios=[1, 2])

    ax1 = fig.add_subplot(gs[:, 0], projection=ccrs.SouthPolarStereo())
    plt.subplots_adjust(
        left=0.04, bottom=0.01, 
        right=0.96, top=0.99, 
        wspace=0.06, hspace=0.05
    )
    modify_map(ax1, -45)
    im1 = ax1.pcolormesh(depth_ocean.lon, depth_ocean.lat, 
                        depth_ocean, 
                        cmap = cmocean.cm.deep,
                        alpha = 0.8,
                        transform=ccrs.PlateCarree())
    ax1.gridlines(
        xlocs = np.arange(-180,180,45), 
        ylocs = np.arange(-70,90,10),
        draw_labels=True
    )

    Psi_plot = Psi_SO_avg['all'].where(depth_ocean>0)
    ax1.contour(Psi_SO_avg.lon, Psi_SO_avg.lat, Psi_plot, 
                levels = [-20], colors = 'gold',
                linewidths=0.8, linestyles='solid',
                transform=ccrs.PlateCarree())
    cs = ax1.contour(Psi_SO_avg.lon, Psi_SO_avg.lat, Psi_plot, 
                levels = [10, 60, 110], colors = 'orangered',
                linewidths=0.8, linestyles='solid',
                transform=ccrs.PlateCarree())
    ax1.clabel(cs, cs.levels, fontsize=8)
    add_arrows(ax1,(-7, -58.28),(-6, -58.25),'gold')
    add_arrows(ax1,(4, -68.2),(3, -68.6),'gold')
    add_arrows(ax1,(-92, -68.4),(-91, -68.4),'orangered')
    add_arrows(ax1,(-103, -59.95),(-102, -60.1),'orangered')
    add_arrows(ax1,(-103, -53.9),(-102, -53.9),'orangered')
    add_arrows(ax1,(-144, -68),(-144, -69),'gold')
    ax1.contour(Psi_SO_avg.lon, Psi_SO_avg.lat, Psi_plot, 
                levels = [0], colors = 'black',
                linewidths=0.8, linestyles='solid',
                transform=ccrs.PlateCarree())
    ax1.text(-55, -33, "a.", transform=ccrs.PlateCarree(), fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(
        im1, orientation='horizontal', 
        aspect=25, pad=0.12, spacing='proportional',
        extend='max'
    )
    cbar1.set_label('Ocean Topography (m)', fontsize=10)
    cbar1.ax.tick_params(labelsize=8, direction='in')
    plot_region_box(ax1, ws_lons, ws_lats, c='lime')
    plot_region_box(ax1, rs_lons, rs_lats, c='cyan')

    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    im2 = plot_region_map(ax2, Psi_plot, conv_ws, ws_lons, ws_lats)
    ax2.plot([-20, -20], [-72, -63], color='yellow', linewidth=1.5)
    cax = plt.axes((0.56, 0.5, 0.40, 0.02))
    cbar2 = plt.colorbar(im2, cax=cax,  orientation='horizontal', extend='both')
    cbar2.set_label('barotropic streamfunction (Sv)', fontsize=10)
    cbar2.ax.tick_params(labelsize=8, direction='in')
    cbar2.outline.set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree(central_longitude=-180))
    im3 = plot_region_map(ax3, Psi_plot, conv_rs, rs_lons, rs_lats)
    polynya_plot = xr.where(dspolynya.mean('time')>0, 1, 0)
    polynya_plot = polynya_plot.where(polynya_plot.lat>-75, drop=True)
    ax3.contourf(
        polynya_plot.lon, polynya_plot.lat, polynya_plot, levels=[0.5, 1.5], 
        colors='none', hatches=['////'], transform=ccrs.PlateCarree(),
    ) 

    ax2.text(0.95, 0.05, "b.", transform=ax2.transAxes, fontsize=12, fontweight='bold')
    ax3.text(0.95, 0.05, "c.", transform=ax3.transAxes, fontsize=12, fontweight='bold')
    # ax2.spines['geo'].set_edgecolor("lime")
    # ax3.spines['geo'].set_edgecolor("cyan")

    proxy_artists = [Rectangle((0,0),1,1, hatch = '\\\\\\\\', facecolor='none', edgecolor='coral'),
                    Rectangle((0,0),1,1, hatch = '////', facecolor='none', edgecolor='black', linewidth=0)]
    fig.legend(proxy_artists, ['open-ocean convection', 'open-ocean polynya'],
                frameon = False,
                bbox_to_anchor= (0.55, 0.54),
                fontsize = 10)

    fig.savefig(figname, format="png", dpi=300)

def get_tsgram_points(ds_st, conv_idx, phase_value, n=500, rnseed=42):
    ds_st_stack = ds_st.stack(xy=("x", "y")).dropna(dim="xy", how="all")
    rng = np.random.default_rng(seed=rnseed)
    locations = rng.choice(ds_st_stack.xy.values, size=n, replace=False)
    ds_st_small = ds_st_stack.sel(xy=locations)
    ds_st_small_0 = ds_st_small.reset_index("y", drop=False)
    ds_st_clean = ds_st_small_0.drop_vars("y")

    df_non = ds_st_clean.where(conv_idx < -phase_value, drop=True).mean("time").to_dataframe().reset_index()
    df_non_clean = df_non.dropna(subset=["so", "thetao"]).drop(columns=["lon", "lat"])

    df_conv = ds_st_clean.where(conv_idx > phase_value, drop=True).mean("time").to_dataframe().reset_index()
    df_conv_clean = df_conv.dropna(subset=["so", "thetao"]).drop(columns=["lon", "lat"])
    return df_non_clean, df_conv_clean

def get_tsgram_points_mean(ds_st, conv_idx, phase_value):
    ds_st_mean = ds_st.mean(('x','y'))
    df_non = ds_st_mean.where(conv_idx < -phase_value, drop=True).to_dataframe().reset_index()
    df_non_clean = df_non.dropna(subset=["so", "thetao"]).drop(columns=["time"])

    df_conv = ds_st_mean.where(conv_idx > phase_value, drop=True).to_dataframe().reset_index()
    df_conv_clean = df_conv.dropna(subset=["so", "thetao"]).drop(columns=["time"])
    return df_non_clean, df_conv_clean

def get_tsgram_points_all(ds_st, conv_idx, phase_value, n=10, rnseed=42):
    ds_st_stack = ds_st.stack(xy=("x", "y")).dropna(dim="xy", how="all")
    rng = np.random.default_rng(seed=rnseed)
    locations = rng.choice(ds_st_stack.xy.values, size=n, replace=False)
    ds_st_small = ds_st_stack.sel(xy=locations)
    ds_st_small_0 = ds_st_small.reset_index("y", drop=False)
    ds_st_clean = ds_st_small_0.drop_vars("y")

    df_non = ds_st_clean.where(conv_idx < -phase_value, drop=True).to_dataframe().reset_index()
    df_non_clean = df_non.dropna(subset=["so", "thetao"]).drop(columns=["time", "lon", "lat"])

    df_conv = ds_st_clean.where(conv_idx > phase_value, drop=True).to_dataframe().reset_index()
    df_conv_clean = df_conv.dropna(subset=["so", "thetao"]).drop(columns=["time", "lon", "lat"])
    return df_non_clean, df_conv_clean    

def get_tsgram_data(ds_st, buff = 0.02, n=100):
    s_min = ds_st.so.min().load().item()
    s_max = ds_st.so.max().load().item()
    s_diff = s_max - s_min
    t_min = ds_st.thetao.min().load().item()
    t_max = ds_st.thetao.max().load().item()
    t_diff = t_max - t_min

    s_range = np.linspace(s_min-buff*s_diff, s_max+buff*s_diff, n)
    t_range = np.linspace(t_min-buff*t_diff, t_max+buff*t_diff, n)
    density = gsw.sigma0(s_range[:, np.newaxis], t_range[np.newaxis, :])
    tsd_dict = {
        "salinity": s_range,
        "temperature": t_range,
        "density": density
    }
    return tsd_dict

def get_conv_idx(ds_t, mask=None, modify_time=None):
    if mask is not None:
        ds_t = mask_time_range(mask['p1'], mask['p2'], ds_t)
        if modify_time is not None:
            modify_time = mask_time_range(mask['p1'], mask['p2'], modify_time)
    conv_idx = -normalize_timeseries(ds_t)
    if modify_time is not None:
        conv_idx = conv_idx.rename({'year':'time'})
        conv_idx['time'] = modify_time
    return conv_idx
    

def add_ts_diagram(ax, df_sample, tsd_dict, title=None, subname=None, 
                   ylabel=True, xls=(34.01, 34.77), yls=(-1.95, 2.2)):
    sc = ax.scatter(
        df_sample['so'], df_sample['thetao'], 
        c = df_sample['lev'], cmap = 'nipy_spectral', s=5
    )
    s_range = tsd_dict['salinity']
    t_range = tsd_dict['temperature']
    density = tsd_dict['density']
    cs = ax.contour(
        s_range, t_range, density.T, levels=np.arange(26.5, 28, 0.1), 
        colors = 'grey', linewidths=0.8, alpha=0.5
    )
    ax.clabel(cs, cs.levels, fmt='%1.1f', fontsize=8)
    ax.set_xlabel("Salinity (psu)", fontsize=10)
    if ylabel:
        ax.set_ylabel("Potential Temperature ($^{\circ}$C)", fontsize=10)
    if title is not None:
        ax.set_title(title)
    ax.tick_params(axis='both', which='major', labelsize=8, 
                   direction='in', top=True, bottom=True, 
                   left=True, right=True)
    ax.set_xlim(xls[0], xls[1])
    ax.set_ylim(yls[0], yls[1])
    if subname is not None:
        ax.text(0.05, 0.93, subname, transform=ax.transAxes, fontsize=12, fontweight='bold')
    return sc

def add_hlines(ax, hl1=1.0, hl2=0.75, c="grey", ls1='--', ls2=':'):
    ax.axhline(0, color = c, alpha = 0.75)
    if hl1 > 0:
        ax.axhline(hl1, color = c, linestyle=ls1, alpha = 0.75)
        ax.axhline(-hl1, color = c, linestyle=ls1, alpha = 0.75)
    if hl2 > 0:
        ax.axhline(hl2, color = c, linestyle=ls2, alpha = 0.75)
        ax.axhline(-hl2, color = c, linestyle=ls2, alpha = 0.75)

def plot_ts_conv_diff(
    tsd_dict, ds_t, dfs=None, da = None, get_points=None,
    mask=None, modify_time=None,
    phase_value=1, ccolor='C3', nccolor='C0',
    xls=(33.91, 34.77), yls=(-1.95, 2.2),
    figsize = (5, 5), **params,
):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2,2, height_ratios=[1, 2.2])
    plt.subplots_adjust(
        left=0.11, bottom=0.08, 
        right=0.92, top=0.92, 
        wspace=0.05, hspace=0.01
    )
    ax1 = fig.add_subplot(gs[0,:])
    conv_idx = get_conv_idx(ds_t, mask, modify_time)
    tr = {'p1':0, 'p2':len(conv_idx)}
    if mask is not None:
        tr = tr | mask
    timerange = range(tr['p1'], tr['p2'])
    ax1.plot(timerange, conv_idx, color='dimgrey')
    if phase_value == 1:
        add_hlines(ax1, hl2=0, c="silver")
    elif phase_value == 0.75:
        add_hlines(ax1, hl1=0.75, hl2=0, c="silver")
    ax1.fill_between(timerange, phase_value, conv_idx, 
                     where=(conv_idx > phase_value), color=ccolor, 
                     alpha=0.3, interpolate=True)
    ax1.fill_between(timerange, -phase_value, conv_idx, 
                     where=(conv_idx < -phase_value), color=nccolor, 
                     alpha=0.3, interpolate=True)
    ax1.tick_params(axis='both', which='major', labelsize=8, 
                    direction='in', top=True, labeltop=True, 
                    bottom=False, labelbottom=False)
    ax1.set_xlabel("year", labelpad=5)
    ax1.xaxis.set_label_position('top')
    ax1.set_yticks([-phase_value, 0, phase_value])
    ax1.spines[:].set_visible(False)
    ax1.text(0.02, 0.88, "a.", 
             transform=ax1.transAxes, 
             fontsize=12, fontweight='bold')
    
    if dfs is not None:
        df_non_clean, df_conv_clean = dfs[0], dfs[1]
    else:
        if da is None:
            raise ValueError("Need st data if no df provided.")
        else:
            df_non_clean, df_conv_clean = get_points(da, conv_idx, phase_value, **params)

    ax2 = fig.add_subplot(gs[1,0])
    sc1 = add_ts_diagram(
        ax2, df_non_clean, tsd_dict, 
        subname = 'b.', xls=xls, yls=yls
    )
    ax2.text(0.17, 0.93, "Non-convection", 
             transform=ax2.transAxes, color=nccolor)

    ax3 = fig.add_subplot(gs[1,1])
    sc2 = add_ts_diagram(
        ax3, df_conv_clean, tsd_dict, 
        subname = 'c.', xls=xls, yls=yls,ylabel=False
    )
    ax3.tick_params(labelright=True, 
                    labelleft=False)
    ax3.text(0.17, 0.93, "Convection", 
             transform=ax3.transAxes, color=ccolor)
    
    cax = plt.axes((0.56, 0.32, 0.02, 0.26))
    cbar2 = plt.colorbar(sc2, cax=cax, orientation='vertical', pad=0.02, extend='max')
    cbar2.ax.tick_params(labelsize=8, direction='in')
    cbar2.set_label("depth (m)", fontsize=8)
    cbar2.outline.set_visible(False)
    cbar2.ax.invert_yaxis()
    return fig


def plot_timeseries(ax, da, c, ls='solid', lw=1.5, alpha = 1):
    ax.plot(da, color = c, linestyle = ls, linewidth = lw, alpha = alpha)

def plot_separation(ax, y1, y2, y3, ylabel = 1, subname=None, legend=True, yls=None):
    plot_timeseries(ax, y1, 'k', lw=1)#(.106, .62, .467))
    plot_timeseries(ax, y2, 'red', alpha=0.5)#(.459, .439, .702), lw=1.5)
    plot_timeseries(ax, y3, 'blue', alpha=0.5)#(.851, .373, .008), lw=1.5)
    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=8, 
                   direction='in')
    if ylabel == 1:
        ax.set_ylabel("potential density\n anomaly (kg/m$^3$)")
    elif ylabel == 2:
        ax.set_ylabel("density difference \n (kg/m$^3$)")
    if subname is not None:
        ax.text(0.96, 0.90, subname, transform=ax.transAxes, fontsize=12, fontweight='bold')
    if legend:
        ax.legend(["density", "density (varied s)", "density (varied t)"], frameon=False, ncol=3)
    if yls is not None:
        ax.set_ylim(yls[0], yls[1])

def get_ys(das, lev1, lev2, zname):
    y1 = das.sel({zname: lev1}, method = 'nearest')
    y2 = das.sel({zname: lev2}, method = 'nearest')
    y3 = y2 - y1
    return y1, y2, y3

def sigma_separation_plot(das, lev1, lev2, zname='lev', figsize = (10, 10)):
    fig, axs = plt.subplots(3, 1, figsize = figsize, sharex=True)
    fig.subplots_adjust(bottom=0.07, top=0.975, left = 0.12, right = 0.98, hspace=0.05)
    vns = list(das.data_vars)
    y11, y21, y31 = get_ys(das[vns[0]], lev1, lev2, zname)
    y12, y22, y32 = get_ys(das[vns[1]], lev1, lev2, zname)
    y13, y23, y33 = get_ys(das[vns[2]], lev1, lev2, zname)

    lev1_value = das.sel({zname: lev1}, method = 'nearest')[zname].item()
    lev2_value = das.sel({zname: lev2}, method = 'nearest')[zname].item()
    
    plot_separation(axs[0], y11, y12, y13, legend=False, yls=(27.32, 27.67))
    axs[0].text(0.014, 0.94, r"$\bf{{{}}}$ surface [{} m]".format("a.",lev1_value), transform=axs[0].transAxes)
    plot_separation(axs[1], y21, y22, y23, yls=(27.48, 27.83))
    axs[1].text(0.014, 0.94, r"$\bf{{{}}}$ subsurface [{} m]".format("b.", lev2_value), transform=axs[1].transAxes)
    plot_separation(axs[2], y31, y32, y33, legend=False, ylabel=2, yls=(-0.035, 0.315))
    axs[2].text(0.014, 0.94, r"$\bf{c.}$ difference", transform=axs[2].transAxes)
    axs[2].set_xlabel("year")
    return fig

def main():
    # filter some warning messages
    import warnings
    warnings.filterwarnings("ignore")

    ds_deptho = ReadDataFromNCAR(variable_id = ['deptho'], grid_label = 'gn', table_id="Ofx")
    depth_ocean = ds_deptho.deptho.load()

    Psi_SO_avg = openpickle('Psi_SO_avg', 'data/')
    # ds_mld = openpickle('GFDL-CM4', '../../SO_data/data_mld/')
    conv_rs = openpickle('conv_rs', 'data/')
    conv_ws = openpickle('conv_ws', 'data/')
    dspolynya = openpickle('GFDL-CM4', '../../SO_data/data_polynya_mean/')

    # gyres_sep_lonmin = openpickle('gyres_sep_lonmin', 'data/')
    gyres_ann_lonmin = openpickle('gyres_ann_lonmin', 'data/')

    # heat_content_rs_ann = openpickle('heat_content_rs_ann', 'data/')
    # heat_content_rs_sep = openpickle('heat_content_rs_sep', 'data/')

    heat_content_2000_rs_sep = openpickle('heat_content_2000_rs_sep', 'data/')
    heat_content_2000_rs_ann = openpickle('heat_content_2000_rs_ann', 'data/')

    conv_properties_rs_ann = openpickle('conv_properties_rs_ann', 'data/')
    conv_properties_rs_sep = openpickle('conv_properties_rs_sep', 'data/')

    fw_properties_rs_ann = openpickle('fw_properties_rs_ann', 'data/')
    # fw_properties_rs_sep = openpickle('fw_properties_rs_sep', 'data/')

    # adv_heat_rs_sep = openpickle('adv_heat_rs_sep', 'data/')
    # adv_heat_rs_ann = openpickle('adv_heat_rs_ann', 'data/')
    # adv_salt_rs_sep = openpickle('adv_salt_rs_sep', 'data/')
    adv_salt_rs_ann = openpickle('adv_salt_rs_ann', 'data/')

    # adv_heat_horiz_rs_sep_100 = adv_heat_rs_sep['adv_horiz'].sel(lev = slice(0,100)).sum('lev')
    # adv_salt_horiz_rs_sep_100 = adv_salt_rs_sep['adv_horiz'].sel(lev = slice(0,100)).sum('lev')
    # adv_heat_horiz_rs_ann_100 = adv_heat_rs_ann['adv_horiz'].sel(lev = slice(0,100)).sum('lev')
    adv_salt_horiz_rs_ann_100 = adv_salt_rs_ann['adv_horiz'].sel(lev = slice(0,100)).sum('lev')

    # curl_sep_timeseries = openpickle('curl_sep_timeseries', 'data/')
    curl_ann_timeseries = openpickle('curl_ann_timeseries', 'data/')

    # ACC_drake_69_sep = openpickle('ACC_drake_69_sep', 'data/')
    ACC_drake_69_ann = openpickle('ACC_drake_69_ann', 'data/')

    # conv_ind_ws = openpickle('conv_ind_ws', 'data/')
    # conv_ind_rs = openpickle('conv_ind_rs', 'data/')

    # uas_ann_maxmin = openpickle('uas_ann_maxmin', 'data/')
    # uas_sep_maxmin = openpickle('uas_sep_maxmin', 'data/')

    # easterlies_ann = uas_ann_maxmin.minwind.sel(lat = slice(-70, -65)).mean('lat')
    # easterlies_sep = uas_sep_maxmin.minwind.sel(lat = slice(-70, -65)).mean('lat')
    ds_st = xr.open_dataset('data/ts_rs_sep.nc', chunks={'time': 50, 'x': 30, 'y': 15})

    ross_lag_cor_ann_dict1 = {
        'SSS': conv_properties_rs_ann['so'].isel(lev=0),
        'precipitation': fw_properties_rs_ann['pr'],
        'evaporation': fw_properties_rs_ann['evs'],
        'salt adv.': adv_salt_horiz_rs_ann_100, 
        'fresh water': fw_properties_rs_ann['wfo'],
        'E-P': fw_properties_rs_ann['emp'],
        'ice melt': fw_properties_rs_ann['ice'],
    }
    ross_lag_cor_ann_dict2 = {
        'SST': conv_properties_rs_ann['thetao'].isel(lev=0),
        'MLD': conv_properties_rs_ann['mld'],
        'heat flux loss': -conv_properties_rs_ann['hfds'],
        'ice conc.': conv_properties_rs_ann['siconc'],
        'ice thick.': conv_properties_rs_ann['sithick'],
        # 'heat adv.': adv_heat_horiz_rs_ann_100, 
        'curl/f': -running_mean(curl_ann_timeseries['rs']),
        # 'easterlies': -running_mean(easterlies_ann),
        'ACC': ACC_drake_69_ann,
        'gyre': -gyres_ann_lonmin['rs'],
    }

    plot_timeseries_rs = openpickle('plotdata_timeseries_rs', 'data/')
    mask = {'p1': 20, 'p2':420, 'color': '0.7', 'alpha': 0.5}

    plot_rs_lag(-heat_content_2000_rs_sep, ross_lag_cor_ann_dict1, ross_lag_cor_ann_dict2, mask=mask)

    ts_data_dict = {
        'mld': {
            'data': conv_properties_rs_sep['mld'],
            'color': 'w',
        },
        'sic': {
            'data': conv_properties_rs_sep['siconc'],
            'color': 'k',
            'offax': -0.01,
            'ylabel': "SIC (%)",
            'ylcoords': (-0.09, 0.57),
            'ylim': (-85, 190),
            'ybounds': (20, 100),
        },
        'hfds': {
            'data': conv_properties_rs_sep['hfds'],
            'color': 'r',
            'offax': -0.01,
            'ylabel': "heat flux \n into the ocean($W/m^2$)",
            'ylcoords': (-0.10, 0.2), 
            'ylim': (-165, 285),
            'ybounds': (-150, -10),         
        },
        'thick': {
            'data': conv_properties_rs_sep['sithick'],
            'color': 'b',
            'offax': 1.01,
            'ylabel': "thickness (m)",
            'ylcoords': (1.08, 0.43), 
            'ylim': (-0.43, 1.8),
            'ybounds': (0.1, 0.9),    
        },
        'gyre': {
            'data': -gyres_ann_lonmin['rs'],
            'color': 'g',
            'offax': 1.01,
            'ylabel': "gyre strength \n (Sv)",
            'ylcoords': (1.075, 0.85), 
            'ylim': (-75, 55),
            'ybounds': (15, 55),    
        },
        'heat': {
            'data': heat_content_2000_rs_ann,
            'color': 'purple',
            'offax': -0.01,
            'ylabel': "heat content \n (10$^{12}$ J/m$^2$)",
            'ylcoords': (-0.1, 0.85), 
            'ylim': (2.0715, 2.108),
            'ybounds': (2.098, 2.108),    
        },
    }
    profile_data_dict = {
        'data': conv_properties_rs_sep['thetao'],
        'clabel': "potential temperature (°C)",
    }
    
    plot_rs_ts(profile_data_dict, ts_data_dict, mask=mask)
    plot_study_area(depth_ocean, Psi_SO_avg, conv_ws, conv_rs, dspolynya)

    phase_value = 0.75

    try:
        client, cluster = connect_dask_cluster()
        tsd_dict = get_tsgram_data(ds_st)
        fig = plot_ts_conv_diff(tsd_dict, heat_content_2000_rs_ann, da=ds_st, 
                        mask=mask, modify_time=ds_st.time, phase_value=phase_value, 
                        get_points=get_tsgram_points)
        fig.savefig("RossTSgram.pdf", format="pdf")

        density_all = gsw.sigma0(ds_st.so, ds_st.thetao)
        density_all_meant = gsw.sigma0(ds_st.so, ds_st.thetao.mean("time"))
        density_all_means = gsw.sigma0(ds_st.so.mean("time"), ds_st.thetao)

        ds_density_all = xr.Dataset({
            "sigma0": density_all,
            "sigma0_t": density_all_meant,
            "sigma0_s": density_all_means,
        })
        density_timeseries = density_all.mean(('x','y'))
        density_meant_ts = density_all_meant.mean(('x','y'))
        density_means_ts = density_all_means.mean(('x','y'))

        ds_density = xr.Dataset({
            "sigma0": density_timeseries.compute(),
            "sigma0_t": density_meant_ts.compute(),
            "sigma0_s": density_means_ts.compute(),
        })
        fig_str = sigma_separation_plot(ds_density, 0, 500, figsize=(6,5))
        fig_str.savefig("RossStratification.pdf", format="pdf")
    finally:
        client.shutdown()
        cluster.close()    
    

if __name__ == "__main__":
    main()