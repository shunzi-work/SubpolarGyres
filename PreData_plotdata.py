# filter some warning messages
import warnings
warnings.filterwarnings("ignore")

import xarray as xr
from myfunctions import openpickle, savepickle

def main():
    m = "GFDL-CM4"
    p_ice = "../../SO_data/data_siconc_w_area/"
    p_mld = "../../SO_data/data_mld/"
    p_thick = "../../SO_data/data_thick/"
    p_hfds = "../../SO_data/data_hfds/"

    dsmld = openpickle(m, p_mld)
    damld = dsmld.mld
    dsice = openpickle(m, p_ice)
    ds_hfds = openpickle(m, p_hfds)
    ds_thick = openpickle(m, p_thick)

    conv_rs = openpickle('conv_rs', 'data/')
    conv_ws = openpickle('conv_ws', 'data/')

    ds_thetao_ws = openpickle('thetao_ws_sep','data/')
    ds_thetao_rs = openpickle('thetao_rs_sep','data/')

    Psi_gyres_ann = openpickle('Psi_gyres_ann', 'data/')

    gyres_ws = Psi_gyres_ann['ws']
    gyres_rs = Psi_gyres_ann['rs']

    ice_ws = dsice.siconc.where(conv_ws>0).mean(('x','y'))
    ice_rs = dsice.siconc.where(conv_rs>0).mean(('x','y'))

    thetao_ws = ds_thetao_ws.mean(('x','y'))
    thetao_rs = ds_thetao_rs.mean(('x','y'))

    hfds_ws = ds_hfds.hfds.where(conv_ws>0).mean(('x','y'))
    hfds_rs = ds_hfds.hfds.where(conv_rs>0).mean(('x','y'))

    thick_ws = ds_thick.sithick.where(conv_ws>0).mean(('x','y'))
    thick_rs = ds_thick.sithick.where(conv_rs>0).mean(('x','y'))

    mld_ws = damld.where(conv_ws>0).mean(('x','y'))
    mld_rs = damld.where(conv_rs>0).mean(('x','y'))

    pltd_ws = {
        "ice": ice_ws,
        "thick": thick_ws,
        "hfds": hfds_ws,
        "mld": mld_ws,
        "thetao": thetao_ws,
        "gyres": gyres_ws
    }
    pltd_rs = {
        "ice": ice_rs,
        "thick": thick_rs,
        "hfds": hfds_rs,
        "mld": mld_rs,
        "thetao": thetao_rs,
        "gyres": gyres_rs
    }
    
    savepickle('plotdata_timeseries_ws', 'data/', pltd_ws)
    savepickle('plotdata_timeseries_rs', 'data/', pltd_rs)

if __name__ == "__main__":
    main()
    






