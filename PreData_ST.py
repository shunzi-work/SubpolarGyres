# filter some warning messages
import warnings
warnings.filterwarnings("ignore")

import xarray as xr
from myfunctions import ReadDataFromNCAR, openpickle, select_month, connect_dask_cluster, ispickleexists, savepickle

def main():
    ds_s = ReadDataFromNCAR(variable_id = ['so'], table_id = 'Omon', grid_label = 'gn')
    ds_t = ReadDataFromNCAR(variable_id = ['thetao'], table_id = 'Omon', grid_label = 'gn')

    conv_rs = openpickle("conv_rs", "data/")
    so_rs = ds_s.so.where(conv_rs>0, drop = True)
    thetao_rs = ds_t.thetao.where(conv_rs>0, drop = True)

    conv_ws = openpickle("conv_ws", "data/")
    so_ws = ds_s.so.where(conv_ws>0, drop = True)
    thetao_ws = ds_t.thetao.where(conv_ws>0, drop = True)

    so_rs_sep = select_month(so_rs, 9)
    so_ws_sep = select_month(so_ws, 9)

    thetao_rs_sep = select_month(thetao_rs, 9)
    thetao_ws_sep = select_month(thetao_ws, 9)

    # thetao_rs_ann = thetao_rs.groupby("time.year").mean("time")
    # thetao_ws_ann = thetao_ws.groupby("time.year").mean("time")

    # so_rs_ann = so_rs.groupby("time.year").mean("time")
    # so_ws_ann = so_ws.groupby("time.year").mean("time")

    # st_dict = {
    #     'so_rs_sep': so_rs_sep, 
    #     'so_ws_sep': so_ws_sep, 
    #     'thetao_rs_sep': thetao_rs_sep,
    #     'thetao_ws_sep': thetao_ws_sep,
    #     'thetao_rs_ann': thetao_rs_ann,
    #     'thetao_ws_ann': thetao_ws_ann,
    #     'so_rs_ann': so_rs_ann,
    #     'so_ws_ann': so_ws_ann,
    # }

    ds_ws = xr.Dataset({
        "so": so_ws_sep,
        "thetao": thetao_ws_sep
    })

    ds_rs = xr.Dataset({
        "so": so_rs_sep,
        "thetao": thetao_rs_sep
    })

    client, cluster = connect_dask_cluster()
    print(f"Dask Dashboard link: {client.dashboard_link}")
    try:
        ds_ws = ds_ws.load()
        ds_ws.to_netcdf('data/ts_ws_sep.nc')
        ds_rs = ds_rs.load()
        ds_rs.to_netcdf('data/ts_rs_sep.nc')
        # for key in st_dict.keys():
        #     print(f"Processing {key}...")
        #     if not ispickleexists(key, 'data/'):
        #         data = st_dict[key].compute()
        #         savepickle(key, 'data/', data)
    finally:
        client.close()
        cluster.close()
    

if __name__ == "__main__":
    main()
    






