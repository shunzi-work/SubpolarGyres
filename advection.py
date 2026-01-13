# filter some warning messages
import warnings
warnings.filterwarnings("ignore")

from myfunctions import *
from xgcm import Grid

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

def calculate_advection(da, grid, ax, dav, dx):
    flux = grid.interp(da, axis=ax, to="right") * dav
    adv = grid.diff(flux, axis = ax) / dx
    return adv

def select_conv_sep(da, conv):
    return select_month(da, 9).where(conv > 0, drop = True)

def main():
    ds_vo = ReadDataFromNCAR(variable_id=["vo"], grid_label = 'gn', table_id="Omon")
    ds_uo = ReadDataFromNCAR(variable_id=["uo"], grid_label = 'gn', table_id="Omon")
    # ds_basin = ReadDataFromNCAR(variable_id=["basin"], grid_label = 'gn')
    # ds_areacello = ReadDataFromNCAR(variable_id=["areacello"], grid_label = 'gn')
    ds_so = ReadDataFromNCAR(variable_id = ['so'], grid_label = 'gn', table_id="Omon")
    ds_thetao = ReadDataFromNCAR(variable_id = ['thetao'], grid_label = 'gn', table_id="Omon")

    ds_vgrid = xr.open_dataset("data/ocean_static_0p25_vgrid.nc")
    ds_ugrid = xr.open_dataset("data/ocean_static_0p25_ugrid.nc")

    dav = replace_uvgrid(ds_vo.vo, ds_vgrid, 'y', 'y_r', renamelat = 'lat_v', renamelon = 'lon_v') 
    dau = replace_uvgrid(ds_uo.uo, ds_ugrid, 'x', 'x_r', renamelat = 'lat_u', renamelon = 'lon_u') 

    ds_all = xr.Dataset({'so': ds_so.so, 'thetao': ds_thetao.thetao, 'vo': dav, 'uo': dau})
    __, dyv = calc_dx(ds_vgrid.lon, ds_vgrid.lat)
    dxu, __ = calc_dx(ds_ugrid.lon, ds_ugrid.lat)
    ds_all['dyv'] = xr.DataArray(data=dyv, dims=["y", "x"])
    ds_all['dxu'] = xr.DataArray(data=dxu, dims=["y", "x"])

    grid = Grid(ds_all, coords={'X': {'center': 'x', 'right': 'x_r'},
                        'Y': {'center': 'y', 'right': 'y_r'}}, periodic=['X'])

    adv_salt_v = calculate_advection(ds_all.so, grid, "Y", dav, ds_all.dyv)
    adv_salt_u = calculate_advection(ds_all.so, grid, "X", dau, ds_all.dxu)
    adv_salt_horiz = adv_salt_u + adv_salt_v
    adv_heat_v = calculate_advection(ds_all.thetao, grid, "Y", dav, ds_all.dyv)
    adv_heat_u = calculate_advection(ds_all.thetao, grid, "X", dau, ds_all.dxu)
    adv_heat_horiz = adv_heat_u + adv_heat_v

    conv_ws = openpickle('conv_ws', 'data/')
    conv_rs = openpickle('conv_rs', 'data/')

    conv_dict = {'conv_rs': conv_rs, 'conv_ws': conv_ws}

    adv_dict = {'adv_salt_v': adv_salt_v, 'adv_salt_u': adv_salt_u, 'adv_heat_horiz': adv_heat_horiz,
                'adv_heat_v': adv_heat_v, 'adv_heat_u': adv_heat_u, 'adv_salt_horiz': adv_salt_horiz}
    
    client, cluster = connect_dask_cluster()
    print(f"Dask Dashboard link: {client.dashboard_link}")
    try:
        for a in adv_dict:
            for c in conv_dict:
                savename = a +'_'+ c[-2:]
                print(savename)
                if not ispickleexists(savename, 'data/'):
                    adv_conv_sep = select_conv_sep(adv_dict[a], conv_dict[c])
                    adv_conv_sep = adv_conv_sep.compute()
                    savepickle(savename, 'data/', adv_conv_sep)    
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()

