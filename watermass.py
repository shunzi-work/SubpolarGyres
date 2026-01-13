# filter some warning messages
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import gsw

from myfunctions import ReadDataFromNCAR, select_month, connect_dask_cluster, savepickle, ispickleexists

def find_AMOC_rho(da_msftyrho, y = -34, rho_sel = 1035):
    rhos_up = []
    rhos_low = []
    da_y = da_msftyrho.sel(y = y, method = 'nearest')
    da_ynew = da_y.where(da_y.rho2_i >= rho_sel)
    AMOC = da_ynew.max("rho2_i")
    for t in AMOC.time:
        da_tmp = da_y.sel(time = t)
        ind = np.where(da_tmp == AMOC.sel(time = t))
        rho2_i0 = da_tmp.rho2_i.isel(rho2_i = ind[0][0]).values
        rhos_up.append(rho2_i0)
        da_tmp2 = da_y.sel(rho2_i = slice(rho2_i0, None))
        ind_low = np.where(da_tmp2 == da_tmp2.min("rho2_i"))
        rho2_i1 = da_tmp2.rho2_i.isel(rho2_i = ind_low[0][0]).values
        rhos_low.append(rho2_i1)
    AMOC['rho_up_time'] = (('time'), rhos_up)
    AMOC['rho_low_time'] = (('time'), rhos_low)
    return AMOC

def compute_AABW_sv_rho(damsfyrho, rho_time):
    results = []
    for t in rho_time.time:
        da_temp = damsfyrho.sel(time = t).sel(rho2_i = rho_time.sel(time = t), method = 'nearest') / rho_time.mean().item() /1e6 
        results.append(da_temp)
    return xr.concat(results, dim='time')

def streamfunction_v(dsv, vname, lonbnds, latbnds, lonrange = None, basin = None, month = None):
    from pyproj import Geod
    g = Geod(ellps='sphere')
    # Using pyproj GEOD to calculates the distance between 
    # grid points that are in a latitude/longitude format.
    
    dz = dsv.lev_bnds[:, 1] - dsv.lev_bnds[:, 0]

    """ vertices of the cells follow the order below
    (1)####(v)####(2)
     #             #
     #             #
    (u)    (T)    (u)
     #             #
     #             #
    (4)####(v)####(3)
    """
    _, _, dx = g.inv(lonbnds[:,:,0], latbnds[:,:,0], lonbnds[:,:,1], latbnds[:,:,1])
    sdim = lonbnds.dims[1]

    vdx = dsv[vname] * dx / 1e6
    if isinstance(basin, xr.DataArray):
        basin = basin.compute()            
        vdx = vdx.where(basin>0, drop = True)
    if lonrange is not None:
        vdx = vdx.sel(x = slice(lonrange[0], lonrange[1]))
    vdx_sum = vdx.sum(dim = sdim)

    Psi = (vdx_sum * dz).cumsum(dim = dsv.lev_bnds.dims[0]) - (vdx_sum * dz).sum(dim = dsv.lev_bnds.dims[0])

    if month is not None:
        if month > 0:
            Psi = Psi.sel(time = Psi.time.dt.month == month)
        elif month == 0:
            Psi = Psi.groupby("time.year").mean("time")

    Psi = Psi / 1e6  # m3/s to Sv
    return Psi


def transport_v(dsv, vname, lonbnds, latbnds, dstans, lonrange = None, basin = None, month = None):
    from pyproj import Geod
    g = Geod(ellps='sphere')
    # Using pyproj GEOD to calculates the distance between 
    # grid points that are in a latitude/longitude format.
    
    dz = dsv.lev_bnds[:, 1] - dsv.lev_bnds[:, 0]

    """ vertices of the cells follow the order below
    (1)####(v)####(2)
     #             #
     #             #
    (u)    (T)    (u)
     #             #
     #             #
    (4)####(v)####(3)
    """
    _, _, dx = g.inv(lonbnds[:,:,0], latbnds[:,:,0], lonbnds[:,:,1], latbnds[:,:,1])
    sdim = lonbnds.dims[1]

    vdx = dsv[vname] * dx / 1e6
    if isinstance(basin, xr.DataArray):
        basin = basin.compute()            
        vdx = vdx.where(basin>0, drop = True)
    if lonrange is not None:
        vdx = vdx.sel(x = slice(lonrange[0], lonrange[1]))
    
    transport = vdx * dz * dstans
    
    transport_sum = transport.sum(dim = sdim)

    if month is not None:
        if month > 0:
            transport_sum = transport_sum.sel(time = transport_sum.time.dt.month == month)
        elif month == 0:
            transport_sum = transport_sum.groupby("time.year").mean("time")

    transport_sum = transport_sum / 1e6  # m3/s to Sv
    return transport_sum



def main():
    ds_msftyrho = ReadDataFromNCAR(variable_id = ["msftyrho"], table_id = "Omon")

    ds_fx = ReadDataFromNCAR(variable_id = ["volcello"], grid_label = "gn")

    da_msftyrho_AL = ds_msftyrho.msftyrho.isel(basin = 0).load()
    da_msftyrho_all = ds_msftyrho.msftyrho.isel(basin = 2).load()

    AMOC_34S = find_AMOC_rho(da_msftyrho_AL, y = -34)

    ds_s = ReadDataFromNCAR(variable_id = ["so"], table_id = "Omon", grid_label = "gn")
    ds_t = ReadDataFromNCAR(variable_id = ["thetao"], table_id = "Omon", grid_label = "gn")
    da_sigma2 = gsw.sigma2(ds_s.so, ds_t.thetao)

    sigma2_weddell = da_sigma2.sel(y = slice(-90, -70), x = slice(-60, 0))
    sigma2_weddell_o = da_sigma2.sel(y = slice(-90, -60), x = slice(-60, 0))

    AABW_weddell = ds_fx.volcello.where(sigma2_weddell >= AMOC_34S.rho_low_time-1000).sum(('x','y','lev')) / 1e9 # m3 to km3
    AABW_weddell_o = ds_fx.volcello.where(sigma2_weddell_o >= AMOC_34S.rho_low_time-1000).sum(('x','y','lev')) / 1e9

    sigma2_ross = da_sigma2.sel(y = slice(-90, -70), x = slice(-210, -150))
    sigma2_ross_o = da_sigma2.sel(y = slice(-90, -60), x = slice(-210, -150))
    AABW_ross = ds_fx.volcello.where(sigma2_ross >= AMOC_34S.rho_low_time-1000).sum(('x','y','lev')) / 1e9 # m3 to km3
    AABW_ross_o = ds_fx.volcello.where(sigma2_ross_o >= AMOC_34S.rho_low_time-1000).sum(('x','y','lev')) / 1e9

    ds_v = ReadDataFromNCAR(variable_id = ["vo"], grid_label = 'gn', table_id="Omon")
    ds_basin = ReadDataFromNCAR(variable_id = ["basin"], grid_label = 'gn')
    Psi_v_w_sep = streamfunction_v(ds_v, 'vo', ds_fx.lon_bnds, ds_fx.lat_bnds, lonrange=(-60, 30), basin=ds_basin.basin.where(ds_basin.basin==1), month=9)
    Psi_v_r_sep = streamfunction_v(ds_v, 'vo', ds_fx.lon_bnds, ds_fx.lat_bnds, lonrange=(-210, -150), basin=ds_basin.basin.where(ds_basin.basin==1), month=9)

    Psi_v_AL = streamfunction_v(ds_v, 'vo', ds_fx.lon_bnds, ds_fx.lat_bnds, basin=ds_basin.basin.where(ds_basin.basin==2), month=0)
    Psi_v_PA = streamfunction_v(ds_v, 'vo', ds_fx.lon_bnds, ds_fx.lat_bnds, basin=ds_basin.basin.where(ds_basin.basin==3), month=0)
    Psi_v_IN = streamfunction_v(ds_v, 'vo', ds_fx.lon_bnds, ds_fx.lat_bnds, basin=ds_basin.basin.where(ds_basin.basin==5), month=0)

    Psi_v_AL_34 = Psi_v_AL.sel(y = -34, method='nearest')
    Psi_v_PA_34 = Psi_v_PA.sel(y = -34, method='nearest')
    Psi_v_IN_34 = Psi_v_IN.sel(y = -34, method='nearest')

    heat_trans_v_AL_ann = transport_v(ds_v, 'vo', ds_fx.lon_bnds, ds_fx.lat_bnds, ds_t.thetao, basin=ds_basin.basin.where(ds_basin.basin==2), month=0)
    heat_trans_v_AL_sep = transport_v(ds_v, 'vo', ds_fx.lon_bnds, ds_fx.lat_bnds, ds_t.thetao, basin=ds_basin.basin.where(ds_basin.basin==2), month=9)
    salt_trans_v_AL_ann = transport_v(ds_v, 'vo', ds_fx.lon_bnds, ds_fx.lat_bnds, ds_s.so, basin=ds_basin.basin.where(ds_basin.basin==2), month=0)
    salt_trans_v_AL_sep = transport_v(ds_v, 'vo', ds_fx.lon_bnds, ds_fx.lat_bnds, ds_s.so, basin=ds_basin.basin.where(ds_basin.basin==2), month=9)

    heat_trans_v_AL_ann_34 = heat_trans_v_AL_ann.sel(y = -34, method='nearest')
    heat_trans_v_AL_sep_34 = heat_trans_v_AL_sep.sel(y = -34, method='nearest')
    salt_trans_v_AL_ann_34 = salt_trans_v_AL_ann.sel(y = -34, method='nearest')
    salt_trans_v_AL_sep_34 = salt_trans_v_AL_sep.sel(y = -34, method='nearest')

    AABW_w_psi = Psi_v_w_sep.min(('y','lev'))
    AABW_r_psi = Psi_v_r_sep.min(('y','lev'))

    AABW_weddell_sep = select_month(AABW_weddell,9)
    AABW_weddell_o_sep = select_month(AABW_weddell_o,9)

    AABW_ross_sep = select_month(AABW_ross,9)
    AABW_ross_o_sep = select_month(AABW_ross_o,9)

    AABW_sv_rho_sep = select_month(compute_AABW_sv_rho(da_msftyrho_all.sel(y = -70, method='nearest'), AMOC_34S.rho_low_time), 9)

    wm_dict = {
        'AMOC_34S': AMOC_34S,
        'AABW_weddell_sep': AABW_weddell_sep,
        'AABW_weddell_o_sep': AABW_weddell_o_sep,
        'AABW_ross_sep': AABW_ross_sep,
        'AABW_ross_o_sep': AABW_ross_o_sep,
        'AABW_sv_rho_sep': AABW_sv_rho_sep,
        'AABW_w_psi': AABW_w_psi,
        'AABW_r_psi': AABW_r_psi,
        'heat_trans_v_AL_sep_34': heat_trans_v_AL_sep_34,
        'salt_trans_v_AL_sep_34': salt_trans_v_AL_sep_34,
        'heat_trans_v_AL_ann_34': heat_trans_v_AL_ann_34,
        'salt_trans_v_AL_ann_34': salt_trans_v_AL_ann_34,
    }

    client, cluster = connect_dask_cluster()
    print(f"Dask Dashboard link: {client.dashboard_link}")

    try:
        for wm in wm_dict:
            print(wm)
            if not ispickleexists(wm, 'data/'):
                wm_data = wm_dict[wm].compute()
                savepickle(wm, 'data/', wm_data)
                
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()

