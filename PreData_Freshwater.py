# filter some warning messages
import warnings
warnings.filterwarnings("ignore")

import xesmf as xe
import xarray as xr
from myfunctions import ReadDataFromNCAR, open_from_cloud, openpickle, select_month, connect_dask_cluster, ispickleexists, savepickle

def main():
    ds_pr = ReadDataFromNCAR(variable_id = 'pr', table_id = 'Amon')
    ds_evs = ReadDataFromNCAR(variable_id = 'evs', table_id = 'Omon')
    ds_wfo = open_from_cloud("gs://cmip6/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1/Omon/wfo/gn/v20180701")

    grid_out = xr.Dataset({"lon": ds_wfo.lon.load(), "lat": ds_wfo.lat.load()})
    grid_pr = xr.Dataset({"lon": (["lon"], ds_pr.lon.values), "lat": (["lat"], ds_pr.lat.values)})
    grid_evs = xr.Dataset({"lon": (["lon"], ds_evs.lon.values), "lat": (["lat"], ds_evs.lat.values)})
    
    regridder_pr = xe.Regridder(grid_pr, grid_out, "bilinear")
    regridder_evs = xe.Regridder(grid_evs, grid_out, "bilinear")
    
    pr_gn = regridder_pr(ds_pr.pr)
    evs_gn = regridder_evs(ds_evs.evs)

    conv_rs = openpickle("conv_rs", "data/")
    wfo_rs = ds_wfo.wfo.where(conv_rs>0).mean(('x','y'))
    pr_rs = pr_gn.where(conv_rs>0).mean(('x','y'))
    evs_rs = evs_gn.where(conv_rs>0).mean(('x','y'))

    conv_ws = openpickle("conv_ws", "data/")
    wfo_ws = ds_wfo.wfo.where(conv_ws>0).mean(('x','y'))
    pr_ws = pr_gn.where(conv_ws>0).mean(('x','y'))
    evs_ws = evs_gn.where(conv_ws>0).mean(('x','y'))
    
    pr_rs_sep = select_month(pr_rs, 9)
    pr_ws_sep = select_month(pr_ws, 9)

    evs_rs_sep = select_month(evs_rs, 9)
    evs_ws_sep = select_month(evs_ws, 9)

    wfo_rs_sep = select_month(wfo_rs, 9)
    wfo_ws_sep = select_month(wfo_ws, 9)

    pr_ann_rs = pr_rs.groupby("time.year").mean("time")
    evs_ann_rs = evs_rs.groupby("time.year").mean("time")
    wfo_ann_rs = wfo_rs.groupby("time.year").mean("time")

    pr_ann_ws = pr_ws.groupby("time.year").mean("time")
    evs_ann_ws = evs_ws.groupby("time.year").mean("time")
    wfo_ann_ws = wfo_ws.groupby("time.year").mean("time")

    fw_dict = {
        'pr_rs_sep': pr_rs_sep, 
        'pr_ws_sep': pr_ws_sep, 
        'evs_rs_sep': evs_rs_sep,
        'evs_ws_sep': evs_ws_sep,
        'wfo_rs_sep': wfo_rs_sep,
        'wfo_ws_sep': wfo_ws_sep,
        'pr_ann_rs': pr_ann_rs,
        'pr_ann_ws': pr_ann_ws,
        'evs_ann_rs': evs_ann_rs,
        'evs_ann_ws': evs_ann_ws,
        'wfo_ann_rs': wfo_ann_rs,
        'wfo_ann_ws': wfo_ann_ws,
    }

    client, cluster = connect_dask_cluster()
    print(f"Dask Dashboard link: {client.dashboard_link}")
    try:
        for key in fw_dict.keys():
            print(f"Processing {key}...")
            if not ispickleexists(key, 'data/'):
                data = fw_dict[key].compute()
                savepickle(key, 'data/', data)
    finally:
        client.close()
        cluster.close()
    

if __name__ == "__main__":
    main()
    






