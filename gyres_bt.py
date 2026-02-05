# filter some warning messages
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
from myfunctions import ReadDataFromNCAR, ispickleexists, openpickle, compute_and_save, connect_dask_cluster, select_month

def calculate_dz_with_bottom_depth(levbnds, depths):
    dz_full = levbnds.isel(bnds=1, drop = True) - levbnds.isel(bnds=0, drop = True)
    mask_above = levbnds.isel(bnds=0, drop = True) < depths
    dz_at_bottom = depths - levbnds.isel(bnds=0, drop = True)
    dz = xr.where(levbnds.isel(bnds=1, drop = True) < depths, 
                dz_full, 
                dz_at_bottom)
    dz = dz.where(mask_above)
    return dz

def BarotropicStreamfunction(dsv, uv, lonbnds, latbnds, depths = None, basin = None, month = None):
    from pyproj import Geod
    g = Geod(ellps='sphere')
    # Using pyproj GEOD to calculates the distance between 
    # grid points that are in a latitude/longitude format.
    
    if depths is not None:
        dz = calculate_dz_with_bottom_depth(dsv.lev_bnds, depths)
    else:
        dz = dsv.lev_bnds[:, 1] - dsv.lev_bnds[:, 0]
    U_bt = (dsv[uv] * dz).sum(dz.dims[0])
    """ vertices of the cells follow the order below
    (1)####(v)####(2)
     #             #
     #             #
    (u)    (T)    (u)
     #             #
     #             #
    (4)####(v)####(3)
    """
    if uv == 'uo':
        _, _, dx = g.inv(lonbnds[:,:,1], latbnds[:,:,1], lonbnds[:,:,2], latbnds[:,:,2])
        sdim = lonbnds.dims[0]
    elif uv == 'vo':
        _, _, dx = g.inv(lonbnds[:,:,0], latbnds[:,:,0], lonbnds[:,:,1], latbnds[:,:,1])
        sdim = lonbnds.dims[1]
        
    Udx = U_bt * dx / 1e6
    if isinstance(basin, xr.DataArray):
        basin = basin.compute()            
        Udx = Udx.where(basin>0, drop = True)
    Psi = Udx.cumsum(dim = sdim, skipna = True)
    if month is not None:
        if month > 0:
            Psi = Psi.sel(time = Psi.time.dt.month == month)
        elif month == 0:
            Psi = Psi.groupby("time.year").mean("time")
    return Psi

def compute_gyre_strength_region_min(Psi, region_lonmin, region_lonmax):
    Psi_region = Psi.sel(x = slice(region_lonmin, region_lonmax))
    Psi_region_sep = select_month(Psi_region, 9)
    Psi_region_ann = Psi_region.groupby('time.year').mean('time')
    gyre_sep = Psi_region_sep.min(('x','y'))
    gyre_ann = Psi_region_ann.min(('x','y'))
    return gyre_sep, gyre_ann

def compute_gyre_strength_lon_min(Psi, lon):
    Psi_transect = Psi.sel(x = lon, method='nearest', drop=True)
    Psi_transect_sep = select_month(Psi_transect, 9)
    Psi_transect_ann = Psi_transect.groupby('time.year').mean('time')
    gyre_sep = Psi_transect_sep.min('y')
    gyre_ann = Psi_transect_ann.min('y')
    return gyre_sep, gyre_ann

def compute_gyre_strength_region_mean(Psi, Psi_mean, lonmin, lonmax):
    Psi_region = Psi_mean.sel(x = slice(lonmin, lonmax))
    Psi_region_masked = Psi.where(Psi_region < -20).mean(('x','y'), skipna=True)
    Psi_region_sep = select_month(Psi_region_masked, 9)
    Psi_region_ann = Psi_region_masked.groupby('time.year').mean('time')
    return Psi_region_sep, Psi_region_ann

def compute_u_transects_mean(ds_v, lon, latmin=-90, latmax=-65):
    u_transect = ds_v.sel(x = lon, method = 'nearest', drop=True).sel(y = slice(latmin, latmax)).mean('time')
    return u_transect

def compute_shelf_Psi(Psi, lon, lat_offshelf, lat_shelf=-90):
    if lat_shelf > -90:
        Psi_shelf_1 = compute_shelf_Psi(Psi, lon, lat_offshelf=lat_shelf)
        Psi_shelf_2 = compute_shelf_Psi(Psi, lon, lat_offshelf=lat_offshelf)
        Psi_shelf = Psi_shelf_1 - Psi_shelf_2
    else:
        Psi_shelf = Psi.sel(x = lon, method='nearest', drop=True).sel(y=lat_offshelf, method='nearest')
    return Psi_shelf

def compute_shelf_current_strength(Psi, lon, lat_offshelf, lat_shelf=-90):
    Psi_shelf = compute_shelf_Psi(Psi, lon, lat_offshelf, lat_shelf)
    Psi_shelf_sep = select_month(Psi_shelf, 9)
    Psi_shelf_ann = Psi_shelf.groupby('time.year').mean('time')
    return Psi_shelf_sep, Psi_shelf_ann

def main():
    ds_v = ReadDataFromNCAR(variable_id=["uo"], grid_label = 'gn', table_id="Omon")
    ds_basin = ReadDataFromNCAR(variable_id=["basin"], grid_label = 'gn')
    ds_areacello = ReadDataFromNCAR(variable_id=["areacello"], grid_label = 'gn')
    ds_deptho = ReadDataFromNCAR(variable_id = ['deptho'], grid_label = 'gn', table_id="Ofx")
    depth_ocean = ds_deptho.deptho.load()

    # Psi_SO_sep = BarotropicStreamfunction(ds_v, "uo", ds_areacello.lon_bnds, ds_areacello.lat_bnds, depth_ocean, ds_basin.basin.where(ds_basin.basin==1), month=9)
    Psi_SO = BarotropicStreamfunction(ds_v, "uo", ds_areacello.lon_bnds, ds_areacello.lat_bnds, depth_ocean, ds_basin.basin.where(ds_basin.basin==1))
    Psi_SO_sep = select_month(Psi_SO, 9)
    dz = calculate_dz_with_bottom_depth(ds_v.lev_bnds, depth_ocean)

    Psi_weddell_gyre_sep, Psi_weddell_gyre_ann = compute_gyre_strength_region_min(Psi_SO, -60, 60)
    Psi_ross_gyre_sep, Psi_ross_gyre_ann = compute_gyre_strength_region_min(Psi_SO, -210, -135)
    gyres_ann_regionmin = xr.Dataset({'ws': Psi_weddell_gyre_ann, 'rs': Psi_ross_gyre_ann})
    gyres_sep_regionmin = xr.Dataset({'ws': Psi_weddell_gyre_sep, 'rs': Psi_ross_gyre_sep})

    Psi_ross_gyre_lon160_sep, Psi_ross_gyre_lon160_ann = compute_gyre_strength_lon_min(Psi_SO, -160)
    Psi_weddell_gyre_lon0_sep, Psi_weddell_gyre_lon0_ann = compute_gyre_strength_lon_min(Psi_SO, 0)

    gyres_sep_lonmin = xr.Dataset({'ws': Psi_weddell_gyre_lon0_sep, 'rs': Psi_ross_gyre_lon160_sep})
    gyres_ann_lonmin = xr.Dataset({'ws': Psi_weddell_gyre_lon0_ann, 'rs': Psi_ross_gyre_lon160_ann})

    Psi_SO_sep_ocean = Psi_SO_sep.where(ds_basin.basin.where(ds_basin.basin==1)>0)
    Psi_SO_ann_ocean = Psi_SO.where(ds_basin.basin.where(ds_basin.basin==1)>0).groupby("time.year").mean("time")

    ACC_drake_69_sep = Psi_SO_sep_ocean.sel(x = -69, method='nearest').sel(y = -55.635, method = 'nearest')
    ACC_drake_69_ann = Psi_SO_ann_ocean.sel(x = -69, method='nearest').sel(y = -55.635, method = 'nearest')

    if ispickleexists('Psi_SO_avg', 'data/'):
        Psi_SO_avg = openpickle('Psi_SO_avg', 'data/')
        Psi_SO_mean = Psi_SO_avg['all']
        Psi_SO_mean_sep = Psi_SO_avg['sep']
    else:
        Psi_SO_mean = Psi_SO.mean("time")
        Psi_SO_mean_sep = Psi_SO_sep.mean("time")
        Psi_SO_avg = xr.Dataset({'all': Psi_SO_mean, 'sep': Psi_SO_mean_sep})

    Psi_weddell_gyre_mean_sep, Psi_weddell_gyre_mean_ann = compute_gyre_strength_region_mean(Psi_SO, Psi_SO_mean, -60, 60)
    Psi_ross_gyre_mean_sep, Psi_ross_gyre_mean_ann = compute_gyre_strength_region_mean(Psi_SO, Psi_SO_mean, -210, -135)

    gyres_sep_meanregion = xr.Dataset({'ws': Psi_weddell_gyre_mean_sep, 'rs': Psi_ross_gyre_mean_sep})
    gyres_ann_meanregion = xr.Dataset({'ws': Psi_weddell_gyre_mean_ann, 'rs': Psi_ross_gyre_mean_ann})

    u_0 = compute_u_transects_mean(ds_v.uo, 0)
    u_20W = compute_u_transects_mean(ds_v.uo, -20)
    u_20E = compute_u_transects_mean(ds_v.uo, 20)
    u_210W = compute_u_transects_mean(ds_v.uo, -210)
    u_130W = compute_u_transects_mean(ds_v.uo, -130)
    u_180W = compute_u_transects_mean(ds_v.uo, -180)
    u_transects = {'u_0': u_0, 'u_20W': u_20W, 'u_20E': u_20E, 'u_210W': u_210W, 'u_130W': u_130W, 'u_180W': u_180W}

    Psi_shelf_0_sep, Psi_shelf_0_ann = compute_shelf_current_strength(Psi_SO, 0, -67.8)
    Psi_shelf_20W_sep, Psi_shelf_20W_ann = compute_shelf_current_strength(Psi_SO, -20, -71.5)
    Psi_shelf_20E_sep, Psi_shelf_20E_ann = compute_shelf_current_strength(Psi_SO, 20, -69)
    Psi_shelf_130W_sep, Psi_shelf_130W_ann = compute_shelf_current_strength(Psi_SO, -130, -72)
    Psi_shelf_130W_n_sep, Psi_shelf_130W_n_ann = compute_shelf_current_strength(Psi_SO, -130, -65, lat_shelf= -68.5)
    Psi_shelf_210W_sep, Psi_shelf_210W_ann = compute_shelf_current_strength(Psi_SO, -210, -65.5)
    Psi_shelf_180W_sep, Psi_shelf_180W_ann = compute_shelf_current_strength(Psi_SO, -180, -68)

    shelf_Psi = {
        'Psi_shelf_lon_0_sep': Psi_shelf_0_sep,
        'Psi_shelf_lon_20W_sep': Psi_shelf_20W_sep,
        'Psi_shelf_lon_20E_sep': Psi_shelf_20E_sep,
        'Psi_shelf_lon_130W_sep': Psi_shelf_130W_sep,
        'Psi_shelf_lon_130W_n_sep': Psi_shelf_130W_n_sep,
        'Psi_shelf_lon_210W_sep': Psi_shelf_210W_sep,
        'Psi_shelf_lon_180W_sep': Psi_shelf_180W_sep,
        'Psi_shelf_lon_0_ann': Psi_shelf_0_ann,
        'Psi_shelf_lon_20W_ann': Psi_shelf_20W_ann,
        'Psi_shelf_lon_20E_ann': Psi_shelf_20E_ann,
        'Psi_shelf_lon_130W_ann': Psi_shelf_130W_ann,
        'Psi_shelf_lon_130W_n_ann': Psi_shelf_130W_n_ann,
        'Psi_shelf_lon_210W_ann': Psi_shelf_210W_ann,
        'Psi_shelf_lon_180W_ann': Psi_shelf_180W_ann
    }

    try:
        client, cluster = connect_dask_cluster()
        print(f"Dask Dashboard link: {client.dashboard_link}")
        compute_and_save('dz', 'data/', dz)
        compute_and_save('gyres_sep_regionmin', 'data/', gyres_sep_regionmin)
        compute_and_save('gyres_ann_regionmin', 'data/', gyres_ann_regionmin)
        compute_and_save('gyres_sep_meanregion', 'data/', gyres_sep_meanregion)
        compute_and_save('gyres_ann_meanregion', 'data/', gyres_ann_meanregion)
        compute_and_save('gyres_sep_lonmin', 'data/', gyres_sep_lonmin)
        compute_and_save('gyres_ann_lonmin', 'data/', gyres_ann_lonmin)
        compute_and_save('Psi_SO_avg', 'data/', Psi_SO_avg)
        compute_and_save('ACC_drake_69_sep', 'data/', ACC_drake_69_sep)
        compute_and_save('ACC_drake_69_ann', 'data/', ACC_drake_69_ann)
        for transect_name, transect_data in u_transects.items():
            compute_and_save(transect_name, 'data/', transect_data)
        for shelf_name, shelf_data in shelf_Psi.items():
            compute_and_save(shelf_name, 'data/', shelf_data)
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()

