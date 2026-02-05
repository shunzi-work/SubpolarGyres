import numpy as np
import xarray as xr
from myfunctions import openpickle, ReadDataFromNCAR, connect_dask_cluster, ispickleexists, savepickle

def try_save(save_name, save_path, data):
    if not ispickleexists(save_name, save_path):
        data = data.compute()
        savepickle(save_name, save_path, data)

def Weight_Calculation() -> None:
    conv_rs = openpickle('conv_rs', 'data/')
    ds_thetao = ReadDataFromNCAR(variable_id = "thetao", grid_label = 'gn', table_id="Omon")
    ds_volcello = ReadDataFromNCAR(variable_id = "volcello", grid_label = 'gn')
    ds_areacello = ReadDataFromNCAR(variable_id = "areacello", grid_label = 'gn')
    dz = openpickle('dz', 'data/')

    cp=3991.86795711963
    dens=1026.0
    c2k=273.15

    tk_rs = (ds_thetao.thetao + c2k).where(conv_rs > 0, drop = True) 
    v_rs = ds_volcello.volcello.where(conv_rs > 0, drop = True)
    a_rs = ds_areacello.areacello.where(conv_rs > 0, drop = True)
    dz_rs = dz.where(conv_rs > 0, drop = True)

    # J/(K*kg) * kg/m3 * K * m3 = J
    heat_time_vol = (cp * dens * tk_rs * v_rs).sel(lev = slice(0, 2001)).sum('lev')
    heat_fx_vol = (cp * dens * tk_rs * dz_rs * a_rs).sel(lev = slice(0, 2001)).sum('lev')

    # J / m2
    heat_time_vol_mean = heat_time_vol.sum(('x','y')) / a_rs.sum(('x','y'))
    heat_fx_vol_mean = heat_fx_vol.sum(('x','y')) / a_rs.sum(('x','y'))
    heat_time_vol_ann = heat_time_vol_mean.groupby("time.year").mean('time')
    heat_fx_vol_ann = heat_fx_vol_mean.groupby("time.year").mean('time')

    heat_no_weight = (cp * dens * tk_rs * dz_rs).sel(lev = slice(0, 2001)).sum('lev')
    heat_no_weight_mean = heat_no_weight.mean(('x','y'), skipna=True)
    heat_no_weight_ann = heat_no_weight_mean.groupby("time.year").mean('time')

    weight_rs = a_rs.fillna(0)
    heat_weighted = heat_no_weight.weighted(weight_rs)
    heat_weighted_mean = heat_weighted.mean(('x','y'))
    heat_weighted_ann = heat_weighted_mean.groupby("time.year").mean('time')

    try:
        client, cluster = connect_dask_cluster()
        print(f"Dask Dashboard link: {client.dashboard_link}")
        try_save('heat_time_vol_ann', 'data/test/', heat_time_vol_ann)
        try_save('heat_fx_vol_ann', 'data/test/', heat_fx_vol_ann)
        try_save('heat_no_weight_ann', 'data/test/', heat_no_weight_ann)
        try_save('heat_weighted_ann', 'data/test/', heat_weighted_ann)
    finally:
        client.close()
        cluster.close()

def main():
    # filter some warning messages
    import warnings
    warnings.filterwarnings("ignore")
    Weight_Calculation()

if __name__ == "__main__":
    main()
