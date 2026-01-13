# filter some warning messages
import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import numpy as np
from myfunctions import compute_and_save, select_month, ReadDataFromNCAR, openpickle, connect_dask_cluster
from xgcm import Grid

def calc_dx(lon, lat):
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
                _, _, dx[i,j] = g.inv(lon[i,j], lat[i,j], lon[i,0], lat[i,0])
    
    return dx, dy

def calculate_advection(da, grid, ax, dav, dx):
    flux = grid.interp(da, axis=ax, to="right") * dav
    adv = grid.diff(flux, axis = ax) / dx
    return adv


def select_conv_sep(da, conv):
    return select_month(da, 9).where(conv > 0, drop = True)


def main():
    ds_t = ReadDataFromNCAR(variable_id=["thetao"], grid_label = 'gn', table_id="Omon")
    ds_s = ReadDataFromNCAR(variable_id=["so"], grid_label = 'gn', table_id="Omon")
    ds_st = xr.merge([ds_t, ds_s])

    ds_basin = ReadDataFromNCAR(variable_id=["basin"], grid_label = 'gn')
    ds_areacello = ReadDataFromNCAR(variable_id=["areacello"], grid_label = 'gn')
    ds_tauuo = ReadDataFromNCAR(variable_id = ['tauuo'])
    ds_tauvo = ReadDataFromNCAR(variable_id = ['tauvo'])
    ds_tau = xr.merge([ds_tauuo, ds_tauvo])

    ds_vgrid = xr.open_dataset("data/ocean_static_0p25_vgrid.nc")
    ds_ugrid = xr.open_dataset("data/ocean_static_0p25_ugrid.nc")



    conv_ws = openpickle('conv_ws', 'data/')
    conv_rs = openpickle('conv_rs', 'data/')

    conv_dict = {'conv_rs': conv_rs, 'conv_ws': conv_ws}

    
    
    try:
        client, cluster = connect_dask_cluster()
        print(f"Dask Dashboard link: {client.dashboard_link}")

    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()

