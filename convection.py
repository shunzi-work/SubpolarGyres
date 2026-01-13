# filter some warning messages
import warnings
warnings.filterwarnings("ignore")

import xesmf as xe
import xarray as xr
import numpy as np
import gsw

from myfunctions import ReadDataFromNCAR, compute_and_save, connect_dask_cluster, openpickle, select_month, cal_mld, ispickleexists, open_from_cloud

def find_conv(ds, zname: str, conv_depth: float = 2000):
    sigma0 = gsw.sigma0(ds['so'], ds['thetao'])
    mld_r = cal_mld(sigma0, zname)
    conv = mld_r.where(mld_r>=conv_depth).mean("time", skipna = True)
    c = conv.reset_coords(zname, drop = True)
    return c

def cal_st_to_density(ds, xname: str, yname: str):
    sigma0 = gsw.sigma0(ds.so, ds.thetao)
    sigma0_avgt = gsw.sigma0(ds.so, ds.thetao.mean("time"))
    sigma0_avgs = gsw.sigma0(ds.so.mean("time"), ds.thetao)
    return sigma0.mean((xname, yname)).load(), sigma0_avgt.mean((xname, yname)).load(), sigma0_avgs.mean((xname, yname)).load()
    
def cal_heat_content(ds, conv, lev1=0, lev2=2000):
    dz = ds.lev_bnds[:, 1] - ds.lev_bnds[:, 0]
    conv_t = ds.thetao.where(conv>0).mean(('x','y'))
    conv_t_dz = conv_t * dz
    heat_content = conv_t_dz.sel(lev=slice(lev1, lev2)).sum('lev')
    heat_sep = select_month(heat_content, 9)
    heat_ann = heat_content.groupby("time.year").mean("time")
    return heat_sep, heat_ann

def get_conv_property_mean(ds, conv, xname = 'x', yname = 'y'):
    ds_conv = ds.where(conv>0, drop = True).mean((xname, yname))
    ds_conv_sep = select_month(ds_conv, 9)
    ds_conv_ann = ds_conv.groupby("time.year").mean("time")
    return ds_conv_sep, ds_conv_ann

def compute_convection_index(thetao_convregion, depth = 2000, timedim = 'time'):
    thetao_conv = thetao_convregion.sel(lev=depth, method='nearest')
    conv_index = -(thetao_conv - thetao_conv.mean(timedim)) / thetao_conv.std(timedim)
    return conv_index

def main_convection() -> None:
    ds_t = ReadDataFromNCAR(variable_id=["thetao"], grid_label = 'gn', table_id="Omon")
    ds_s = ReadDataFromNCAR(variable_id=["so"], grid_label = 'gn', table_id="Omon")
    ds_st = xr.merge([ds_t, ds_s])

    ds_hfds = ReadDataFromNCAR(variable_id=["hfds"], grid_label = 'gn', table_id="Omon")
    ds_siconc = ReadDataFromNCAR(variable_id=["siconc"], grid_label = 'gn', table_id="SImon")
    ds_sithick = ReadDataFromNCAR(variable_id=["sithick"], grid_label = 'gn', table_id="SImon")

    ds_sep = select_month(ds_st, 9)
    ds_sep_SO = ds_sep.sel({'y': slice(-90, -40)})
    ds_sep_ws = ds_sep_SO.sel(x = slice(-60, 60))
    ds_sep_rs = ds_sep_SO.sel(x = slice(-210, -135))

    sigma0 = gsw.sigma0(ds_st['so'], ds_st['thetao'])
    mld = cal_mld(sigma0, 'lev')

    if ispickleexists('conv_ws', 'data/'):
        conv_ws = openpickle('conv_ws', 'data/')
    else:
        conv_ws = find_conv(ds_sep_ws, 'lev')

    if ispickleexists('conv_rs', 'data/'):
        conv_rs = openpickle('conv_rs', 'data/')
    else:
        conv_rs = find_conv(ds_sep_rs, 'lev')

    heat_content_ws_sep, heat_content_ws_ann = cal_heat_content(ds_sep_ws, conv_ws)
    heat_content_rs_sep, heat_content_rs_ann = cal_heat_content(ds_sep_rs, conv_rs)

    mld_rs_sep, mld_rs_ann = get_conv_property_mean(mld, conv_rs)
    mld_ws_sep, mld_ws_ann = get_conv_property_mean(mld, conv_ws)

    so_rs_sep, so_rs_ann = get_conv_property_mean(ds_st.so, conv_rs)
    so_ws_sep, so_ws_ann = get_conv_property_mean(ds_st.so, conv_ws)

    thetao_rs_sep, thetao_rs_ann = get_conv_property_mean(ds_st.thetao, conv_rs)
    thetao_ws_sep, thetao_ws_ann = get_conv_property_mean(ds_st.thetao, conv_ws)

    hfds_rs_sep, hfds_rs_ann = get_conv_property_mean(ds_hfds.hfds, conv_rs)
    hfds_ws_sep, hfds_ws_ann = get_conv_property_mean(ds_hfds.hfds, conv_ws)

    siconc_rs_sep, siconc_rs_ann = get_conv_property_mean(ds_siconc.siconc, conv_rs)
    siconc_ws_sep, siconc_ws_ann = get_conv_property_mean(ds_siconc.siconc, conv_ws)

    sithick_rs_sep, sithick_rs_ann = get_conv_property_mean(ds_sithick.sithick, conv_rs)
    sithick_ws_sep, sithick_ws_ann = get_conv_property_mean(ds_sithick.sithick, conv_ws)

    conv_properties_ws_ann = xr.Dataset({
        'so': so_ws_ann,
        'thetao': thetao_ws_ann,
        'hfds': hfds_ws_ann,
        'siconc': siconc_ws_ann,
        'sithick': sithick_ws_ann,
        'mld': mld_ws_ann,
    })
    conv_properties_rs_ann = xr.Dataset({
        'so': so_rs_ann,
        'thetao': thetao_rs_ann,
        'hfds': hfds_rs_ann,
        'siconc': siconc_rs_ann,
        'sithick': sithick_rs_ann,
        'mld': mld_rs_ann,
    })
    conv_properties_ws_sep = xr.Dataset({
        'so': so_ws_sep,
        'thetao': thetao_ws_sep,
        'hfds': hfds_ws_sep,
        'siconc': siconc_ws_sep,
        'sithick': sithick_ws_sep,
        'mld': mld_ws_sep,
    })
    conv_properties_rs_sep = xr.Dataset({
        'so': so_rs_sep,
        'thetao': thetao_rs_sep,
        'hfds': hfds_rs_sep,
        'siconc': siconc_rs_sep,
        'sithick': sithick_rs_sep,
        'mld': mld_rs_sep,
    })

    conv_ind_ws = compute_convection_index(thetao_ws_sep)
    conv_ind_rs = compute_convection_index(thetao_rs_sep)

    st_dict = {
        'conv_ind_ws': conv_ind_ws,
        'conv_ind_rs': conv_ind_rs,
        'heat_content_ws_sep': heat_content_ws_sep,
        'heat_content_ws_ann': heat_content_ws_ann,
        'heat_content_rs_sep': heat_content_rs_sep,
        'heat_content_rs_ann': heat_content_rs_ann,
    }

    try:
        client, cluster = connect_dask_cluster()
        print(f"Dask Dashboard link: {client.dashboard_link}")
        compute_and_save('conv_ws', 'data/', conv_ws)
        compute_and_save('conv_rs', 'data/', conv_rs)
        for key, value in st_dict.items():
            compute_and_save(key, 'data/', value)
        compute_and_save('conv_properties_ws_ann', 'data/', conv_properties_ws_ann)
        compute_and_save('conv_properties_rs_ann', 'data/', conv_properties_rs_ann)
        compute_and_save('conv_properties_ws_sep', 'data/', conv_properties_ws_sep)
        compute_and_save('conv_properties_rs_sep', 'data/', conv_properties_rs_sep)
    finally:
        client.close()
        cluster.close()

def main_freshwater() -> None:
    ds_pr = ReadDataFromNCAR(variable_id = 'pr', table_id = 'Amon')
    ds_evs = ReadDataFromNCAR(variable_id = 'evs', table_id = 'Omon')
    ds_wfo = open_from_cloud("gs://cmip6/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1/Omon/wfo/gn/v20180701")

    grid_out = xr.Dataset({"lon": ds_wfo.lon.load(), "lat": ds_wfo.lat.load()})
    grid_pr = xr.Dataset({"lon": (["lon"], ds_pr.lon.values), "lat": (["lat"], ds_pr.lat.values)})
    grid_evs = xr.Dataset({"lon": (["lon"], ds_evs.lon.values), "lat": (["lat"], ds_evs.lat.values)})
    
    regridder_pr = xe.Regridder(grid_pr, grid_out, "bilinear")
    regridder_evs = xe.Regridder(grid_evs, grid_out, "bilinear")
    
    pr_gn = regridder_pr(ds_pr.pr)       # positive values
    evs_gn = regridder_evs(ds_evs.evs)   # negative values
    emp_gn = evs_gn + pr_gn 
    ice_gn = ds_wfo.wfo - emp_gn         # wfo: positive 

    conv_rs = openpickle("conv_rs", "data/")
    conv_ws = openpickle("conv_ws", "data/")

    pr_rs_sep, pr_rs_ann = get_conv_property_mean(pr_gn, conv_rs)
    pr_ws_sep, pr_ws_ann = get_conv_property_mean(pr_gn, conv_ws)

    evs_rs_sep, evs_rs_ann = get_conv_property_mean(evs_gn, conv_rs)
    evs_ws_sep, evs_ws_ann = get_conv_property_mean(evs_gn, conv_ws)

    wfo_rs_sep, wfo_rs_ann = get_conv_property_mean(ds_wfo.wfo, conv_rs)
    wfo_ws_sep, wfo_ws_ann = get_conv_property_mean(ds_wfo.wfo, conv_ws)

    emp_ws_sep, emp_ws_ann = get_conv_property_mean(emp_gn, conv_ws)
    emp_rs_sep, emp_rs_ann = get_conv_property_mean(emp_gn, conv_rs)

    ice_ws_sep, ice_ws_ann = get_conv_property_mean(ice_gn, conv_ws)
    ice_rs_sep, ice_rs_ann = get_conv_property_mean(ice_gn, conv_rs)

    fw_properties_ws_ann = xr.Dataset({
        'pr': pr_ws_ann,
        'evs': evs_ws_ann,
        'wfo': wfo_ws_ann,
        'emp': emp_ws_ann,
        'ice': ice_ws_ann,
    })
    fw_properties_rs_ann = xr.Dataset({
        'pr': pr_rs_ann,
        'evs': evs_rs_ann,
        'wfo': wfo_rs_ann,
        'emp': emp_rs_ann,
        'ice': ice_rs_ann,
    })
    fw_properties_ws_sep = xr.Dataset({
        'pr': pr_ws_sep,
        'evs': evs_ws_sep,
        'wfo': wfo_ws_sep,
        'emp': emp_ws_sep,
        'ice': ice_ws_sep,
    })
    fw_properties_rs_sep = xr.Dataset({
        'pr': pr_rs_sep,
        'evs': evs_rs_sep,
        'wfo': wfo_rs_sep,
        'emp': emp_rs_sep,
        'ice': ice_rs_sep,
    })
    fw_dict = {
        'fw_properties_ws_ann': fw_properties_ws_ann,
        'fw_properties_rs_ann': fw_properties_rs_ann,
        'fw_properties_ws_sep': fw_properties_ws_sep,
        'fw_properties_rs_sep': fw_properties_rs_sep,
    }
    try:
        client, cluster = connect_dask_cluster()
        print(f"Dask Dashboard link: {client.dashboard_link}")
        for key, value in fw_dict.items():
            compute_and_save(key, 'data/', value)
    finally:
        client.close()
        cluster.close()


def main() -> None:
    main_convection()
    main_freshwater()

if __name__ == "__main__":
    main()