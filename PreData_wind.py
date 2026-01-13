# filter some warning messages
import warnings
warnings.filterwarnings("ignore")

import xesmf as xe
import xarray as xr
import numpy as np

from xgcm import Grid

from myfunctions import ReadDataFromNCAR, open_from_cloud, openpickle, select_month, connect_dask_cluster, compute_and_save
from convection import get_conv_property_mean

def replace_uvgrid(da, dsgrid, dimname, renamedim, latname = 'lat', lonname = 'lon', renamelat = 'lat_r', renamelon = 'lon_r'):
    da[dimname] = dsgrid[dimname]
    da[latname] = dsgrid[latname]
    da[lonname] = dsgrid[lonname]
    da = da.rename({dimname: renamedim, latname: renamelat, lonname: renamelon})
    return da

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


def get_sep_boxmean(da, box):
    da_box = da.sel(lon=slice(box[0], box[1]), lat=slice(box[2], box[3])).mean(('lat','lon'))
    return select_month(da_box, 9)

def get_wind_boxmean(ds_vas, ds_uas, ds_sfcWind, box):
    vas_box = get_sep_boxmean(ds_vas.vas, box)
    uas_box = get_sep_boxmean(ds_uas.uas, box)
    sfcWind_box = get_sep_boxmean(ds_sfcWind.sfcWind, box)
    wind_box_sep = xr.Dataset({'vas': vas_box, 'uas': uas_box, 'sfcWind': sfcWind_box})
    return wind_box_sep

def get_wind_bandmean(ds_vas, ds_uas, ds_sfcWind, lat1, lat2):
    vas_band = ds_vas.vas.sel(lat=slice(lat1, lat2)).mean('lat')
    uas_band = ds_uas.uas.sel(lat=slice(lat1, lat2)).mean('lat')
    sfcWind_band = ds_sfcWind.sfcWind.sel(lat=slice(lat1, lat2)).mean('lat')
    wind_band_sep = xr.Dataset({'vas': select_month(vas_band, 9), 'uas': select_month(uas_band, 9), 'sfcWind': select_month(sfcWind_band, 9)})
    return wind_band_sep


def get_wind_band_maxmin(wind, yname = 'lat', xname = 'lon', lat_south=-90, lat_north=90):
    wind_band = wind.sel({yname:slice(lat_south, lat_north)})
    wind_band_sep = select_month(wind_band, 9)
    wind_band_ann = wind_band.groupby("time.year").mean("time")
    wind_band_sep_min = wind_band_sep.min(xname)
    wind_band_sep_max = wind_band_sep.max(xname)
    wind_band_ann_min = wind_band_ann.min(xname)
    wind_band_ann_max = wind_band_ann.max(xname)
    wind_band_sep_maxmin = xr.Dataset({
        'maxwind': wind_band_sep_max,
        'minwind': wind_band_sep_min,
    })
    wind_band_ann_maxmin = xr.Dataset({
        'maxwind': wind_band_ann_max,
        'minwind': wind_band_ann_min,
    })
    return wind_band_sep_maxmin, wind_band_ann_maxmin

def main():
    ds_vas = open_from_cloud("gs://cmip6/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1/Amon/vas/gr1/v20180701")
    ds_uas = open_from_cloud("gs://cmip6/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1/Amon/uas/gr1/v20180701")
    ds_sfcWind = open_from_cloud("gs://cmip6/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1/Amon/sfcWind/gr1/v20180701")

    ds_tauuo = ReadDataFromNCAR(variable_id = ['tauuo'])
    ds_tauvo = ReadDataFromNCAR(variable_id = ['tauvo'])
    ds_tau = xr.merge([ds_tauuo, ds_tauvo])

    ds_vgrid = xr.open_dataset("data/ocean_static_0p25_vgrid.nc")
    ds_ugrid = xr.open_dataset("data/ocean_static_0p25_ugrid.nc")

    conv_rs = openpickle('conv_rs', 'data/')
    conv_ws = openpickle('conv_ws', 'data/')

    wsbox = conv_ws.where(conv_ws > 0, drop = True).x.min().item(), conv_ws.where(conv_ws > 0, drop = True).x.max().item(), conv_ws.where(conv_ws > 0, drop = True).y.min().item(), conv_ws.where(conv_ws > 0, drop = True).y.max().item()
    rsbox = conv_rs.where(conv_rs > 0, drop = True).x.min().item(), conv_rs.where(conv_rs > 0, drop = True).x.max().item(), conv_rs.where(conv_rs > 0, drop = True).y.min().item(), conv_rs.where(conv_rs > 0, drop = True).y.max().item()

    wind_ws_sep = get_wind_boxmean(ds_vas, ds_uas, ds_sfcWind, wsbox)
    wind_rs_sep = get_wind_boxmean(ds_vas, ds_uas, ds_sfcWind, rsbox)

    wind_65_sep = get_wind_bandmean(ds_vas, ds_uas, ds_sfcWind, -65, -64)
    wind_50_sep = get_wind_bandmean(ds_vas, ds_uas, ds_sfcWind, -50, -49)
    wind_40_60_sep = get_wind_bandmean(ds_vas, ds_uas, ds_sfcWind, -60, -40)

    uas_sep_maxmin, uas_ann_maxmin = get_wind_band_maxmin(ds_uas.uas, lat_north=-40)

    datatauvo = replace_uvgrid(ds_tau.tauvo, ds_vgrid, 'y', 'y_r', renamelat = 'lat_v', renamelon = 'lon_v') 
    datatauuo = replace_uvgrid(ds_tau.tauuo, ds_ugrid, 'x', 'x_r', renamelat = 'lat_u', renamelon = 'lon_u')

    ds_all = xr.Dataset({'tauuo': datatauuo, 'tauvo':datatauvo})

    dxv, dyv = calc_dx(ds_vgrid.lon, ds_vgrid.lat)
    dxu, dyu = calc_dx(ds_ugrid.lon, ds_ugrid.lat)

    ds_all['dyv'] = xr.DataArray(data=dyv, dims=["y", "x"])
    ds_all['dxu'] = xr.DataArray(data=dxu, dims=["y", "x"])
    ds_all['dxv'] = xr.DataArray(data=dxv, dims=["y_r", "x_r"])
    ds_all['dyu'] = xr.DataArray(data=dyu, dims=["y_r", "x_r"])

    grid = Grid(ds_all, coords={'X': {'center': 'x', 'right': 'x_r'},
                            'Y': {'center': 'y', 'right': 'y_r'}}, periodic=['X'])
    
    # curl = d tauy / dx - d taux / dy
    curl = grid.diff(ds_all.tauvo, axis = 'X')/ds_all.dxv - grid.diff(ds_all.tauuo, axis = 'Y')/ds_all.dyu
    curl['x_r'] = ds_tau.x.values
    curl['y_r'] = ds_tau.y.values
    curl = curl.rename({'x_r': 'x', 'y_r': 'y'})

    curl_rs_sep, curl_rs_ann = get_conv_property_mean(curl, conv_rs)
    curl_ws_sep, curl_ws_ann = get_conv_property_mean(curl, conv_ws)


    curl_rs_box = curl.sel(x=slice(rsbox[0], rsbox[1]), y=slice(rsbox[2], rsbox[3]))
    curl_ws_box = curl.sel(x=slice(wsbox[0], wsbox[1]), y=slice(wsbox[2], wsbox[3]))

    curl_rs_box_sep = select_month(curl_rs_box, 9).mean(('x','y'))
    curl_ws_box_sep = select_month(curl_ws_box, 9).mean(('x','y'))
    curl_rs_box_ann = curl_rs_box.groupby("time.year").mean("time").mean(('x','y'))
    curl_ws_box_ann = curl_ws_box.groupby("time.year").mean("time").mean(('x','y'))

    curl_sep_timeseries = xr.Dataset({
        'rs': curl_rs_sep, 
        'ws': curl_ws_sep, 
        'rs_box': curl_rs_box_sep, 
        'ws_box': curl_ws_box_sep,
    })  

    curl_ann_timeseries = xr.Dataset({
        'rs': curl_rs_ann, 
        'ws': curl_ws_ann, 
        'rs_box': curl_rs_box_ann, 
        'ws_box': curl_ws_box_ann,
    })  

    wind_dict = {
        'curl_sep_timeseries': curl_sep_timeseries,
        'curl_ann_timeseries': curl_ann_timeseries,
        'wind_ws_sep': wind_ws_sep,
        'wind_rs_sep': wind_rs_sep,
        'wind_65_sep': wind_65_sep,
        'wind_50_sep': wind_50_sep,
        'wind_40_60_sep': wind_40_60_sep,
        'uas_sep_maxmin': uas_sep_maxmin, 
        'uas_ann_maxmin': uas_ann_maxmin,
    }

    try:
        client, cluster = connect_dask_cluster()
        print(f"Dask Dashboard link: {client.dashboard_link}")

        for key, value in wind_dict.items():
            compute_and_save(key, 'data/', value)
    finally:
        client.close()
        cluster.close()
    

if __name__ == "__main__":
    main()
    






