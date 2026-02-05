# advection

import os, glob
import xarray as xr
import numpy as np
from myfunctions import ReadDataFromNCAR, openpickle, connect_dask_cluster, ispickleexists, savepickle, select_month
from xgcm import Grid

def replace_uvgrid(da, dsgrid, dimname, renamedim, latname = 'lat', lonname = 'lon', renamelat = 'lat_r', renamelon = 'lon_r'):
    da[dimname] = dsgrid[dimname]
    da[latname] = dsgrid[latname]
    da[lonname] = dsgrid[lonname]
    da = da.rename({dimname: renamedim, latname: renamelat, lonname: renamelon})
    return da

def calc_dx(lon, lat, periodic=True):
    ''' This definition calculates the distance 
        between grid points that are in
        a latitude/longitude format.
        
        Using pyproj GEOD; different Earth Shapes 
        https://jswhit.github.io/pyproj/pyproj.Geod-class.html
        Common shapes: 'sphere', 'WGS84', 'GRS80'
        
        Accepts, 1D arrays for latitude and longitude
        
        Returns: dx, dy; 2D arrays of distances 
                       between grid points in the x and y direction in meters
        ------------
        modified from:
        https://github.com/Unidata/MetPy/issues/288#issuecomment-279481555
    '''
    from pyproj import Geod
    
    g = Geod(ellps='sphere')
    
    lon = lon.values
    lat = lat.values
    
    dx = np.empty(lon.shape)
    dy = np.zeros(lat.shape)
    
    for i in range(dx.shape[0]-1):
        for j in range(dx.shape[1]):
            _, _, dy[i,j] = g.inv(lon[i,j], lat[i,j], lon[i+1,j], lat[i+1,j])
    dy[i+1,:] = dy[i,:]
            
    for i in range(dx.shape[0]):
        for j in range(dy.shape[1]):
            if j < dy.shape[1]-1:
                _, _, dx[i,j] = g.inv(lon[i,j], lat[i,j], lon[i,j+1], lat[i,j+1])
            else:
                if periodic:
                    _, _, dx[i,j] = g.inv(lon[i,j], lat[i,j], lon[i,0], lat[i,0])
                else:
                    dx[i,j] = dx[i,j-1]
    
    return dx, dy

def calculate_advection(da, grid, ax, dav, dx):
    flux = grid.interp(da, axis=ax, to="right") * dav
    adv = grid.diff(flux, axis = ax) / dx
    return adv


def cal_adv_small_region(da, da_vo, da_uo, ds_vgrid, ds_ugrid, xy_dict):
    ds = xr.Dataset({
        "da": da,
        "uo": da_uo,
        "vo": da_vo,
    })
    ds_sr = ds.sel(xy_dict)
    dsv_sr = ds_vgrid.sel(xy_dict)
    dsu_sr = ds_ugrid.sel(xy_dict)
    dav = replace_uvgrid(ds_sr['vo'], dsv_sr, 'y', 'y_r',
                         renamelat='lat_v', renamelon='lon_v')
    dau = replace_uvgrid(ds_sr['uo'], dsu_sr, 'x', 'x_r',
                         renamelat='lat_u', renamelon='lon_u')
    __, dyv = calc_dx(dsv_sr.lon, dsv_sr.lat, periodic=False)
    dxu, __ = calc_dx(dsu_sr.lon, dsu_sr.lat, periodic=False)
    ds_sr_all = xr.Dataset({
        'da': ds_sr["da"],
        'vo': dav,
        'uo': dau,
        'dyv': xr.DataArray(data=dyv, dims=["y", "x"]),
        'dxu': xr.DataArray(data=dxu, dims=["y", "x"]),
    })
    grid = Grid(ds_sr_all, coords={'X': {'center': 'x', 'right': 'x_r'},
                                   'Y': {'center': 'y', 'right': 'y_r'}},
                periodic=['X'])
    adv_v = calculate_advection(ds_sr_all['da'], grid, "Y", dav,
                                ds_sr_all.dyv)
    adv_u = calculate_advection(ds_sr_all['da'], grid, "X", dau,
                                ds_sr_all.dxu)
    return adv_v, adv_u

# def try_compute(savedata):
#     try:
#         client, cluster = connect_dask_cluster()
#         print(f"Dask Dashboard link: {client.dashboard_link}")
#         savedata = savedata.compute()
#         return savedata
#     except Exception as e:
#         # Code that runs if an exception occurs
#         print(f"An error occurred: {e}")
#         client.shutdown()
#         cluster.close()
#         return None
#     finally:
#         client.shutdown()
#         cluster.close()


def try_compute_and_save(savename, savedata, savepath='data/'):
    if not ispickleexists(savename, savepath):
        try:
            client, cluster = connect_dask_cluster()
            print(f"Dask Dashboard link: {client.dashboard_link}")
            savedata = savedata.compute()
            savepickle(savename, savepath, savedata)
            print(savename)
        except Exception as e:
            # Code that runs if an exception occurs
            print(f"An error occurred: {e}")
            client.shutdown()
            cluster.close()
        finally:
            client.shutdown()
            cluster.close()
            
def combine_adv_timeseries(varname, timename, rgname, advnames, fpath='data/'):
    ds_timeseries = xr.Dataset()
    for advname in advnames:
        filename = advname + '_' + varname + '_' + rgname + '_' + timename
        if ispickleexists(filename, fpath):
            ts = openpickle(filename, fpath)
            ds_timeseries[advname] = ts
        else:
            print("files imcomplete.")
            ds_timeseries = None
            break
    if ds_timeseries is not None:
        savename = 'adv_' + varname +'_' + rgname + '_' + timename
        savepickle(savename, fpath, ds_timeseries)
        for advname in advnames:
            fp = fpath + advname + '_' + varname + '_' + rgname + '_' + timename + '.pickle'
            os.remove(fp)
        



def main():
    # filter some warning messages
    import warnings
    warnings.filterwarnings("ignore")

    ds_vo = ReadDataFromNCAR(variable_id=["vo"], grid_label = 'gn', table_id="Omon")
    ds_uo = ReadDataFromNCAR(variable_id=["uo"], grid_label = 'gn', table_id="Omon")
    ds_so = ReadDataFromNCAR(variable_id = ['so'], grid_label = 'gn', table_id="Omon")
    ds_thetao = ReadDataFromNCAR(variable_id = ['thetao'], grid_label = 'gn', table_id="Omon")

    ds_vgrid = xr.open_dataset("data/ocean_static_0p25_vgrid.nc")
    ds_ugrid = xr.open_dataset("data/ocean_static_0p25_ugrid.nc")

    dz = openpickle('dz', 'data/')
    conv_ws = openpickle('conv_ws', 'data/')
    conv_rs = openpickle('conv_rs', 'data/')

    wb_xy_dict = {"x": slice(-53, 21.05), "y": slice(-75, -60)}
    rb_xy_dict = {"x": slice(-210, -140), "y": slice(-76.95, -63.05)}

    variable_dict = {'heat': ds_thetao.thetao, 'salt': ds_so.so}
    conv_dict = {
        'ws': {'box': wb_xy_dict, 'conv': conv_ws},
        'rs': {'box': rb_xy_dict, 'conv': conv_rs}
    }

    time_group = np.arange(0, len(ds_so.time)+1, step = 1200)

    for varname, vardata in variable_dict.items():
        for rgname in conv_dict.keys():
            adv_v_rg, adv_u_rg = cal_adv_small_region(
                vardata, ds_vo.vo, ds_uo.uo,
                ds_vgrid, ds_ugrid, conv_dict[rgname]['box']
            )
            adv_horiz_rg = adv_v_rg + adv_u_rg
            adv_dict = {
                'adv_u': adv_u_rg * dz,
                'adv_v': adv_v_rg * dz,
                'adv_horiz': adv_horiz_rg * dz,
            }

            
            for advname in adv_dict.keys():
                adv_conv = adv_dict[advname].where(conv_dict[rgname]['conv']>0, drop=True)
                adv_conv_mean = adv_conv.mean(('x','y'))
                
                ffn1 = 'adv_' + varname +'_' + rgname + '_sep'
                if not ispickleexists(ffn1, 'data/'):
                    adv_conv_sep = select_month(adv_conv_mean, 9)
                    sn_sep = advname + '_' + varname + '_' + rgname + '_sep'
                    try_compute_and_save(sn_sep, adv_conv_sep)                
                
                ffn2 = 'adv_' + varname +'_' + rgname + '_ann'
                if not ispickleexists(ffn2, 'data/'):
                    sn_ann = advname + '_' + varname + '_' + rgname + '_ann'
                    sp_temp = 'data/temp/'
                    if not ispickleexists(sn_ann, 'data/'):
                        adv_conv_ann_list = []
                        for i in range(0, len(time_group)-1):
                            adv_conv_mean_i = adv_conv_mean.isel(time = slice(time_group[i], time_group[i+1]))
                            adv_conv_ann_i = adv_conv_mean_i.groupby("time.year").mean("time")
                            svname_i = f"{sn_ann}_part{i}"
                            try_compute_and_save(svname_i, adv_conv_ann_i, sp_temp)
                        for i in range(0, len(time_group)-1):
                            fname_i = f"{sn_ann}_part{i}"
                            if ispickleexists(fname_i, sp_temp):
                                adv_conv_ann_i = openpickle(fname_i, sp_temp)
                                adv_conv_ann_list.append(adv_conv_ann_i)
                            else:
                                adv_conv_ann_list = None
                                break
                        if adv_conv_ann_list is not None:
                            adv_conv_ann = xr.combine_by_coords(adv_conv_ann_list)
                            savepickle(sn_ann, 'data/', adv_conv_ann)
                            print(sn_ann)
                            fnames = sp_temp + sn_ann + '_part*'
                            matching_temp_files = glob.glob(fnames)
                            if len(matching_temp_files) > 0:
                                for file_path in matching_temp_files:
                                    os.remove(file_path)
                        else:
                            print(f"Failed to compute all parts for {sn_ann}")

            combine_adv_timeseries(varname, 'sep', rgname, adv_dict.keys(), fpath='data/')
            combine_adv_timeseries(varname, 'ann', rgname, adv_dict.keys(), fpath='data/')
    

if __name__ == "__main__":
    main()
