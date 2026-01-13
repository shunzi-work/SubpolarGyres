import xarray as xr
import pandas as pd
import numpy as np

import os
import pickle
import glob
import gsw

from pyproj import Geod

def lighten_color(color, amount=0.5):
    """
    from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mcolors
    import colorsys
    c = np.array(colorsys.rgb_to_hls(*mcolors.to_rgb(color)))
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])

def ispickleexists(n, p0):
    """
    Check if a pickle file exists.

    Args:
        n (str): The name of the pickle file without the extension.
        p0 (str): The base path where the pickle file is located.

    Returns:
        bool: True if the pickle file exists, False otherwise.
    """
    p = p0 + n + '.pickle'
    if os.path.exists(p):
        # print('    [o] {} exists.'.format(p))
        return True
    else:
        return False

def openpickle(n, p0):
    """
    Load a pickle file and return its contents.

    Args:
        n (str): The name of the pickle file without the extension.
        p0 (str): The path to the directory containing the pickle file.

    Returns:
        object: The contents of the pickle file.

    Example:
        data = openpickle('datafile', '/path/to/directory/')
    """
    p = p0 + n + '.pickle'
    d = pd.read_pickle(p)
    # with open(p, 'rb') as df:
    #     d = pickle.load(df)
    return d

def savepickle(n, p0, sf):
    """
    Save an object to a pickle file.

    Args:
        n (str): The name of the file (without extension).
        p0 (str): The path or prefix to the directory where the file will be saved.
        sf (object): The object to be serialized and saved.

    Returns:
        None
    """
    p = p0 + n + '.pickle'
    with open(p, 'wb') as wf:
        pickle.dump(sf, wf, pickle.HIGHEST_PROTOCOL)

def open_from_cloud(link):
    """
    Open a dataset from a cloud storage link using xarray and zarr.

    Parameters:
    link (str): The cloud storage link to the dataset.

    Returns:
    xarray.Dataset: The opened dataset.

    Notes:
    - This function uses the gcsfs library to access Google Cloud Storage.
    - The dataset is opened using xarray's open_zarr method.
    - If the dataset contains a coordinate named 'type', it is removed.
    """
    import gcsfs
    gcs = gcsfs.GCSFileSystem(token='anon')
    mapper = gcs.get_mapper(link)
    ## open it using xarray and zarr
    ds = xr.open_zarr(mapper, consolidated=True)
    if 'type' in ds.coords:
        ds = ds.reset_coords('type', drop = True)
    return ds

def open_nc(mf):
    """
    Open multiple NetCDF files and concatenate them along the time dimension.

    Parameters:
    mf (list of str): List of file paths to NetCDF files. If the list contains more than 50 files,
                      the files are opened and concatenated one by one to avoid memory issues.

    Returns:
    xarray.Dataset: The concatenated dataset from the provided NetCDF files.

    Notes:
    - If the 'type' coordinate exists in the dataset, it will be dropped.
    - The function uses `use_cftime=True` to handle non-standard calendars in the NetCDF files.
    """
    if len(mf)>50:
        ds = xr.open_mfdataset(mf[0], use_cftime=True)
        for i in range(1, len(mf)):
            ds0 = xr.open_mfdataset(mf[i], use_cftime=True)
            ds = xr.concat([ds, ds0], dim="time")
    else:
        ds = xr.open_mfdataset(mf, use_cftime=True)
    if 'type' in ds.coords:
        ds = ds.reset_coords('type', drop = True)
    return ds

def select_month(da, n):
    """
    Selects data from an xarray DataArray for a specific month.

    Parameters:
    da (xarray.DataArray): The input data array with a time dimension.
    n (int): The month to select (1 for January, 2 for February, etc.).

    Returns:
    xarray.DataArray: A DataArray containing data only for the specified month.
    """
    return da.isel(time=(da.time.dt.month == n))

def open_nc_month(mf, month):
    """
    Open and concatenate NetCDF files for a specific month.

    Parameters:
    mf (list of str): List of file paths to NetCDF files.
    month (int): The month to select from the dataset (1 for January, 2 for February, etc.).

    Returns:
    xarray.Dataset: The concatenated dataset for the specified month.

    Notes:
    - If the length of `mf` is greater than 5, the function will open each file individually,
      select the specified month, and concatenate them along the time dimension.
    - If the length of `mf` is 5 or less, the function will open all files at once with chunking.
    - If the dataset contains a coordinate named 'type', it will be dropped.
    """
    if len(mf)>5:
        ds = xr.open_mfdataset(mf[0], use_cftime=True)
        ds = select_month(ds, month)
        for i in range(1, len(mf)):
            ds0 = xr.open_mfdataset(mf[i], use_cftime=True)
            ds0 = select_month(ds0, month)
            ds = xr.concat([ds, ds0], dim="time")
        # ds = xr.open_mfdataset(mf, use_cftime=True)
    else:
        ds = xr.open_mfdataset(mf, use_cftime=True, chunks={'time': 12})
        ds = select_month(ds, month)
    if 'type' in ds.coords:
        ds = ds.reset_coords('type', drop = True)
    return ds

def open_nc_month_save_temp(mf, month):
    """
    Opens a list of NetCDF files, selects data for a specific month, and saves the modified data to new NetCDF files.

    Parameters:
    mf (list of str): List of file paths to the NetCDF files.
    month (int): The month to select from the data (1 for January, 2 for February, etc.).

    Returns:
    None
    """
    for i in range(0, len(mf)):
        ds0 = xr.open_mfdataset(mf[i], use_cftime=True)
        ds0 = select_month(ds0, month)
        ds0.to_netcdf(mf[i].replace('.nc', '_temp.nc'))
        ds0.close()
        del ds0

def read_nc_files(p_nc, name, selected_month, dataname):
    """
    Reads NetCDF files based on the provided parameters and returns the dataset.

    Parameters:
    p_nc (str): The base path to the NetCDF files.
    name (str): The name identifier for the NetCDF files.
    selected_month (int or None): The specific month to filter the data. If None, no month filtering is applied.
    dataname (str): The data name identifier for the NetCDF files.

    Returns:
    xarray.Dataset: The dataset containing the data from the NetCDF files.

    Raises:
    ValueError: If no matching NetCDF files are found.

    Notes:
    - If `selected_month` is provided and `name` is 'CanESM5-1', a temporary file is created and used for further processing.
    - The function uses `glob` to find matching files based on the constructed file path pattern.
    """
    import glob, os
    data_path = p_nc + dataname + '_*' + name + '_piControl_' + '*' + '.nc'
    matching_files = glob.glob(data_path)
    if len(matching_files) > 0:
        if selected_month:
            if  name == 'CanESM5-1':
                open_nc_month_save_temp(matching_files, selected_month)
                data_path_new = p_nc + dataname + '_*' + name + '_piControl_' + '*' + '_temp.nc'
                matching_files_new = glob.glob(data_path_new)
                ds = open_nc_month(matching_files_new, selected_month)
            else:
                ds = open_nc_month(matching_files, selected_month) 
        else:
            ds = open_nc(matching_files)
    else:
        raise ValueError("    [x] no {} data.".format(dataname))
    return ds    

def get_latlon(data_info, ds0, newlatlon=False, nolatlon=False):
    """
    Retrieve or generate latitude and longitude coordinates from a dataset.

    Parameters:
    data_info (dict): Dictionary containing metadata about the dataset, including keys for latitude and longitude names.
    ds0 (xarray.Dataset): The dataset from which to retrieve or generate latitude and longitude coordinates.
    newlatlon (tuple, optional): Tuple containing new latitude and longitude names to use if lat/lon are not found in data_info. Default is False.
    nolatlon (bool, optional): Flag to indicate whether to skip checking for existing lat/lon in data_info. Default is False.

    Returns:
    tuple: A tuple containing two xarray.DataArray objects for latitude and longitude coordinates.
    """
    # Check if latitude and longitude names are not provided in data_info
    if not nolatlon:  # default is False (by default, will check again, but if set to true, won't check the table)
        nolatlon = pd.isna(data_info['latname'])  # check again if lat lon name exist

    # If latitude and longitude names do not exist
    if nolatlon:  # if True, means lat lon doesn't exist
        if newlatlon:
            latname, lonname = newlatlon  # Use provided new latitude and longitude names
        else:
            latname = data_info['yname']  # Use yname from data_info as latitude name
            lonname = data_info['xname']  # Use xname from data_info as longitude name

        # Extract longitude and latitude values from the dataset
        x = ds0[lonname]
        y = ds0[latname]

        # Create new longitude and latitude meshgrid
        newlon, newlat = np.meshgrid(x, y)

        # Convert meshgrid to xarray DataArray
        dlon = xr.DataArray(newlon, dims={latname: y.values, lonname: x.values})
        dlat = xr.DataArray(newlat, dims={latname: y.values, lonname: x.values})
    else:
        if newlatlon:
            latname, lonname = newlatlon  # Use provided new latitude and longitude names
        else:
            latname = data_info['latname']  # Use latname from data_info as latitude name
            lonname = data_info['lonname']  # Use lonname from data_info as longitude name

        # Load latitude and longitude data from the dataset
        dlat = ds0[latname].load()
        dlon = ds0[lonname].load()

        # Filter latitude and longitude values to valid ranges
        dlat = dlat.where(dlat < 91).where(dlat > -91)
        dlon = dlon.where(dlon < 361).where(dlon > -361)

    # Remove 'time' coordinate if it exists
    if 'time' in dlat.coords:
        dlat = dlat.reset_coords('time', drop=True)
        dlon = dlon.reset_coords('time', drop=True)

    # Remove 'time_bounds' coordinate if it exists
    if 'time_bounds' in dlat.coords:
        dlat = dlat.reset_coords('time_bounds', drop=True)
        dlon = dlon.reset_coords('time_bounds', drop=True)

    return dlat, dlon

def read_areacello(p_nc, name, var, data_info):
    """
    Reads the cell area netCDF file based on the provided path, variable, and name.

    Parameters:
    p_nc (str): The base path to the netCDF files.
    name (str): The name identifier for the netCDF files.
    var (str): The variable name to be used in the file path.

    Returns:
    xarray.Dataset: The dataset containing the cell area information.

    Raises:
    ValueError: If no matching files or multiple matching files are found.
    """
    import glob
    grid_path = p_nc + var + '_Ofx_' + name + '_*' + '.nc'
    matching_gfiles = glob.glob(grid_path)
    if len(matching_gfiles)==1:
        dsg = xr.open_mfdataset(matching_gfiles)
    elif len(matching_gfiles)>1:
        grid_path = p_nc + var + '_Ofx_' + name + '_*' + data_info['grid_label'] + '*.nc'
        matching_gfiles = glob.glob(grid_path)
        if len(matching_gfiles)==1:
            dsg = xr.open_mfdataset(matching_gfiles)
        else:
            raise ValueError("    [x] cell area data error.")
    else:
        raise ValueError("    [x] cell area data error.")
    return dsg

def create_new_ds(da, area, lat, lon, daname):
    """
    Create a new xarray Dataset with the given data and coordinates.

    Parameters:
    da (xarray.DataArray): The data array to be included in the dataset.
    area (xarray.DataArray): The area data array.
    lat (xarray.DataArray): The latitude data array.
    lon (xarray.DataArray): The longitude data array.
    daname (str): The name to be assigned to the data array in the dataset.

    Returns:
    xarray.Dataset: A new dataset containing the provided data and coordinates.

    Raises:
    Exception: If an error occurs during the creation of the dataset.
    """
    area_data = area.values
    try:
        new_ds = xr.Dataset(
            data_vars = {
                daname: da,
                'areacello': (lat.dims, area_data),
                'newlat': lat,
                'newlon': lon,
            }
        )
        return new_ds
    except Exception as error:
        print("    An exception occurred:", error)

def calculate_area_xy(ds, dataname):
    """
    Calculate the area of each grid cell in a dataset.
    Parameters:
    ds (xarray.Dataset): The dataset containing the data.
    dataname (str): The name of the data variable for which to calculate the area.
    Returns:
    xarray.DataArray: A DataArray containing the area of each grid cell.
    Raises:
    TypeError: If the boundary coordinates are not found in the dataset.
    """
    # Initialize Geod object for geodesic calculations
    g = Geod(ellps='sphere')
    
    # Extract the data array from the dataset
    dataarray = ds[dataname]
    
    # Define the names of the boundary coordinates
    ybnds_name = dataarray.dims[-2] + '_bnds'
    xbnds_name = dataarray.dims[-1] + '_bnds'
    
    # Check if boundary coordinates exist in the dataset
    if (ybnds_name in ds.data_vars) or (ybnds_name in ds.coords):
        ybnds = ds[ybnds_name].values
        xbnds = ds[xbnds_name].values
    else:
        raise TypeError("No x/y bnds")
    
    # Extract the x and y coordinates from the dataset
    y = ds[dataarray.dims[-2]].values
    x = ds[dataarray.dims[-1]].values
    
    # Initialize arrays to store the distances
    dx = np.empty((dataarray.shape[-2], dataarray.shape[-1])) * np.nan
    dy = np.empty((dataarray.shape[-2], dataarray.shape[-1])) * np.nan
    
    # Calculate the distance between x boundaries for each y coordinate
    for i in range(len(x)):
        for j in range(len(y)):
            _, _, dx[j, i] = g.inv(xbnds[i, 0], y[j], xbnds[i, 1], y[j])
    
    # Calculate the distance between y boundaries for each x coordinate
    for j in range(len(y)):
        _, _, dy[j, :] = g.inv(x[0], ybnds[j, 0], x[0], ybnds[j, 1])
    
    # Create a DataArray for the area of each grid cell
    areadata = xr.DataArray(
        data=dx * dy,
        dims=dataarray.dims[-2:],
        coords={list(dataarray.dims)[-1]: dataarray[list(dataarray.dims)[-1]], list(dataarray.dims)[-2]: dataarray[list(dataarray.dims)[-2]]}
    )
    
    # Return the area DataArray
    return areadata.where(areadata>0)


def calculate_area_latlon(ds, dsinfo):
    """
    Calculate the area of grid cells in a dataset with latitude and longitude coordinates.
    Parameters:
    ds (xarray.Dataset): The dataset containing the latitude and longitude coordinates and their vertices.
    dsinfo (dict): A dictionary containing the names of the latitude and longitude variables in the dataset.
        - 'latname': The name of the latitude variable.
        - 'lonname': The name of the longitude variable.
    Returns:
    xarray.DataArray: A DataArray containing the calculated area of each grid cell, with the same dimensions as the latitude and longitude variables in the dataset. The area is calculated in square meters and cells with non-positive area are masked out.
    Raises:
    TypeError: If the dataset does not contain the latitude and longitude vertices.
    """
    ####### vertices of the cells follow the order below
    #(1)#########(2)
    #            #
    #            #
    #            #
    #            #
    #(4)#########(3)
    g = Geod(ellps='sphere')
    latvertices_name = 'vertices_' + dsinfo['latname']
    lonvertices_name = 'vertices_' + dsinfo['lonname']
    if (latvertices_name in ds.data_vars) or (latvertices_name in ds.coords):
        latb = ds[latvertices_name].values
        lonb = ds[lonvertices_name].values
    else:
        raise TypeError("No lat/lon bnds")

    lat = ds[dsinfo['latname']].values
    lon = ds[dsinfo['lonname']].values

    dx = np.empty((lon.shape[0], lon.shape[1]))*np.nan
    dy = np.empty((lat.shape[0], lat.shape[1]))*np.nan

    for i in range(dy.shape[0]):
        for j in range(dy.shape[1]):
            _, _, dy[i,j] = g.inv(lonb[i,j,1], latb[i,j,1], lonb[i,j,2], latb[i,j,2])
            
    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            _, _, dx[i,j] = g.inv(lonb[i,j,0], latb[i,j,0], lonb[i,j,1], latb[i,j,1])
            
    areadata = xr.DataArray(
        data = dx*dy,
        dims = ds[dsinfo['latname']].dims,
        coords={ds[dsinfo['latname']].dims[0]: ds[dsinfo['latname']][ds[dsinfo['latname']].dims[0]], 
                ds[dsinfo['latname']].dims[1]: ds[dsinfo['latname']][ds[dsinfo['latname']].dims[1]], }
    )
    return areadata.where(areadata>0)


def newxy_fmissingxy(dx, dy):
    """
    Replace missing values in the input arrays `dx` and `dy` with new values.

    Parameters:
    dx (xarray.DataArray): Input data array for x-coordinates.
    dy (xarray.DataArray): Input data array for y-coordinates.

    Returns:
    tuple: A tuple containing two xarray.DataArray objects (dsx, dsy) with missing values replaced.
    """
    # Filter out invalid values in dx and dy arrays
    dxv = dx.where((dx > -361) & (dx < 361)).values
    dyv = dy.where((dy > -91) & (dy < 91)).values

    # Find the first non-NaN row in dx and dy arrays
    newx0 = dxv[~np.isnan(dxv).any(axis=1)][0]
    newy0 = dyv[:, ~np.isnan(dyv).any(axis=0)][:, 0]

    # Create a meshgrid from the non-NaN values
    newx, newy = np.meshgrid(newx0, newy0)

    # Replace NaN values in dx and dy with the corresponding values from the meshgrid
    x = np.where(np.isnan(dx), newx, dxv)
    y = np.where(np.isnan(dy), newy, dyv)

    # Create new DataArray objects for the modified dx and dy arrays
    dsx = xr.DataArray(x, dims=dx.dims, coords=dx.coords)
    dsy = xr.DataArray(y, dims=dy.dims, coords=dy.coords)

    return dsx, dsy


def find_first_non_nan_row(da):
    """
    Finds the first row in each column where the value is not NaN.

    Parameters:
    da (xarray.DataArray): The input DataArray to search for non-NaN values.

    Returns:
    xarray.DataArray: A DataArray containing the coordinates of the first non-NaN value in each column.
    """
    # Find non-NaN values
    da = da.where(da > 0)
    not_nan = ~da.isnull()
    # Get the indices of the first non-NaN value 
    first_non_nan_indices = not_nan.argmax(da.dims[-2])
    # Get the corresponding coordinate values
    first_non_nan_rows = da[da.dims[-2]][first_non_nan_indices]
    return first_non_nan_rows

def detect_polynya(da_ice, da_area, ice_threshold, area_threshold=(100, 1000), flood_points = [(0,0)], buffering = 15):
    """
    Detects polynyas (areas of open water surrounded by sea ice) in sea ice concentration data.
    Parameters:
    da_ice (xarray.DataArray): Sea ice concentration data array with 3 dimensions including 'time'.
    da_area (xarray.DataArray): Grid area data array with the 2 dimensions.
    ice_threshold (float): Threshold value for sea ice concentration to consider as open water.
    area_threshold (tuple): A tuple of two floats representing the minimum and maximum area thresholds 
                            (in 10^3 km^2) for detected polynyas.
    flood_points (list of tuples, optional): List of points to start flood fill to remove coastal areas. 
                                             Default is [(0,0)].
    buffering (float, optional): Tolerance value for the flood fill operation. Default is 15.
    Returns:
    xarray.DataArray: A masked data array where detected polynyas are retained and other areas are set to NaN.
    """
    from scipy import ndimage
    from skimage.segmentation import flood_fill
    # Copy the input data arrays to avoid modifying the original data
    daice = da_ice.copy()
    daarea = da_area.copy()

    # Fill NaN values in the area data array with the mean value
    daarea = daarea.fillna(daarea.mean().values.item())
    
    # Generate a binary structure for labeling connected components
    s = ndimage.generate_binary_structure(2,2)
    
    # Create an empty masked data array with the same dimensions as the input data
    da_masked = xr.DataArray(np.nan*np.empty_like(daice), dims = daice.dims, coords = daice.coords)
    
    # Loop through each time step in the data array
    for year in daice.time:
        # Copy the sea ice concentration data for the current time step and fill NaN values with 0
        ice0_flood = daice.sel(time = year).copy()
        ice0_flood = ice0_flood.fillna(0).values
        
        # Apply flood fill to remove coastal areas based on the provided flood points
        for flood_point in flood_points:
            ice0_flood = flood_fill(ice0_flood, flood_point, 0, tolerance=buffering)
        
        # Apply flood fill to remove areas outside the sea ice extent
        ice0_flood = flood_fill(ice0_flood, (ice0_flood.shape[0]-1, ice0_flood.shape[1]-1), 0, tolerance=buffering)
        
        # Identify areas with sea ice concentration below the threshold
        icenew = ice0_flood <= ice_threshold
        
        # Label connected components in the binary image
        labeled_image, num_features = ndimage.label(icenew, structure = s)
        
        # If there are less than 2 features, skip to the next time step
        if num_features < 2:
            continue
        
        # Initialize a mask to store the detected polynyas
        mask = np.zeros_like(labeled_image)
        
        # Loop through each labeled feature
        for i in range(1, num_features+1):
            # Calculate the area of the current feature
            area = daarea.where(labeled_image == i).sum()/1e9  # m2 -> 10^3 km2
            
            # If the area is within the specified thresholds, mark it as a polynya
            if (area > area_threshold[0]) and (area < area_threshold[1]):
                mask[labeled_image == i] = 1
        
        # Mask the sea ice concentration data for the current time step
        ice_value = daice.sel(time=year).values
        ice_value[mask == 0] = np.nan
        da_masked.loc[year] = ice_value
    
    # Return the masked data array
    return da_masked


def count_polynya_area(ds, ice_threshold, area_threshold, flood_points, buffering, re=False):
    """
    Calculate the total area of polynyas in a given dataset.

    Parameters:
    ds (xarray.Dataset): The dataset containing sea ice concentration ('siconc') and cell area ('areacello').
    ice_threshold (float): Threshold value for sea ice concentration to consider as open water.
    area_threshold (tuple): A tuple of two floats representing the minimum and maximum area thresholds 
                            (in 10^3 km^2) for detected polynyas.
    flood_points (list of tuples, optional): List of points to start flood fill to remove coastal areas. 
                                             Default is [(0,0)].
    buffering (float, optional): Tolerance value for the flood fill operation. Default is 15.
    re (int or bool, optional): If provided, only count polynyas that occur at least 're' times. Default is False.

    Returns:
    float: The total area of detected polynyas.
    """
    masked = detect_polynya(ds.siconc, ds.areacello, ice_threshold, area_threshold, flood_points = flood_points, buffering = buffering)
    polynya_count = masked.count('time')
    if re:
        polynya_count = polynya_count.where(polynya_count >= re)
    area_total = ds.areacello.where(polynya_count > 0).sum().values.item()
    area_max = ds.areacello.where(polynya_count > 0).sum((ds.areacello.dims[0], ds.areacello.dims[1])).max().values.item()
    area_mean = ds.areacello.where(polynya_count > 0).sum((ds.areacello.dims[0], ds.areacello.dims[1])).mean().values.item()
    return [area_total, area_max, area_mean]


def drop_coords(ds):
    """
    Drop coordinate variables from an xarray Dataset that are not also dimensions.

    Parameters:
    ds (xarray.Dataset): The input dataset from which coordinate variables will be dropped.

    Returns:
    xarray.Dataset: A new dataset with the specified coordinate variables removed.
    """
    for v in ds.coords:
        if v not in ds.dims:
            ds = ds.drop_vars(v)
    return ds


def copy_coords(copyfrom, copyto, latname, lonname):
    """
    Copy latitude and longitude coordinates from one dataset to another.

    Parameters:
    copyfrom (xarray.Dataset): The source dataset from which to copy the coordinates.
    copyto (xarray.Dataset): The target dataset to which the coordinates will be copied.
    latname (str): The name of the latitude coordinate variable.
    lonname (str): The name of the longitude coordinate variable.

    Returns:
    xarray.Dataset: A new dataset with the copied coordinates.
    """
    newd = copyto.assign_coords(
        {
            latname : copyfrom[latname],
            lonname : copyto[lonname],
        }
    )
    return newd


def rename_xy(data_copyfrom, data_copyto):
    """
    Renames the dimensions of data_copyto to match the dimensions of data_copyfrom.

    Parameters:
    data_copyfrom (xarray.DataArray or xarray.Dataset): The data structure from which to copy the dimension names.
    data_copyto (xarray.DataArray or xarray.Dataset): The data structure whose dimensions will be renamed.

    Returns:
    xarray.DataArray or xarray.Dataset: A new data structure with renamed dimensions.
    """
    new = data_copyto.rename(
        {
            data_copyto.dims[-1]:data_copyfrom.dims[-1], 
            data_copyto.dims[-2]:data_copyfrom.dims[-2], 
        }
    )
    return new


def copy_xy(data_copyfrom, data_copyto):
    """
    Copies the values of the last two dimensions from one xarray DataArray to another.

    Parameters:
    data_copyfrom (xarray.DataArray): The source DataArray from which to copy the values.
    data_copyto (xarray.DataArray): The target DataArray to which the values will be copied.

    Returns:
    xarray.DataArray: The target DataArray with the copied values from the source DataArray.
    """
    data_copyto = rename_xy(data_copyfrom, data_copyto)
    data_copyto[data_copyto.dims[-1]] = data_copyfrom[data_copyfrom.dims[-1]].values
    data_copyto[data_copyto.dims[-2]] = data_copyfrom[data_copyfrom.dims[-2]].values
    return data_copyto

def regrid_data(da, ds_in, ds_out):
    """
    Regrids the input data array from the source grid to the destination grid using bilinear interpolation.

    Parameters:
    da (xarray.DataArray): The data array to be regridded.
    ds_in (xarray.Dataset): The source grid dataset.
    ds_out (xarray.Dataset): The destination grid dataset.

    Returns:
    xarray.DataArray: The regridded data array.
    """
    import xesmf as xe
    regridder = xe.Regridder(ds_in, ds_out, "bilinear", periodic=True)
    da_out = regridder(da)
    return da_out

def regrid_based_on_dsgxy(da, dsg, dsinfo):
    """
    Regrids the input data array `da` based on the grid information provided in `dsg`.

    Parameters:
    da (xarray.DataArray): The input data array to be regridded.
    dsg (xarray.Dataset): The dataset containing the target grid information.
    dsinfo (dict): A dictionary containing the names of the coordinate variables. 
                   It should have keys 'xname' and 'yname' corresponding to the 
                   names of the x and y coordinates in `da` and `dsg`.

    Returns:
    xarray.DataArray: The regridded data array.
    """
    ds_in = {dsinfo['xname']: da[dsinfo['xname']].values, dsinfo['yname']: da[dsinfo['yname']].values}
    ds_out = {dsinfo['xname']: dsg[dsinfo['xname']].values, dsinfo['yname']: dsg[dsinfo['yname']].values}
    return regrid_data(da, ds_in, ds_out)    

def set_land_to_nan(ds):
    """
    Sets Antarctic land areas to NaN in the sea ice concentration data.
    Only for GISS, INM-CM4-8

    This function processes the sea ice concentration data (siconc) in the given dataset (ds)
    and sets the land areas to NaN. It uses a flood fill algorithm to identify the land areas
    based on the sea ice concentration values.

    Parameters:
    ds (xarray.Dataset): The input dataset containing sea ice concentration data.

    Returns:
    xarray.Dataset: The modified dataset with land areas set to NaN in the sea ice concentration data.
    """
    from skimage.segmentation import flood_fill
    ice = ds.siconc
    for t in ice.time:
        ice0 = ice.sel(time = t)
        dffill = flood_fill(ice0.values, (0, 0), np.nan, tolerance=0)
        if ~np.isnan(dffill[-1,-1]):
            break
    ice_mask = xr.DataArray(
        data = dffill,
        dims = ice0.dims, 
        coords=ice0.coords
    )
    newice = ice.where(~ice_mask.isnull())
    ds['siconc'] = newice
    return ds

def set_ocean_to_zero(ds):
    """
    Sets values to zero in the 'siconc' variable of the given dataset.

    This function fills NaN values in the 'siconc' variable of the dataset with zero,
    and then calls the set_land_to_nan function to process the dataset further.

    Parameters:
    ds (xarray.Dataset): The input dataset containing the 'siconc' variable.

    Returns:
    xarray.Dataset: The modified dataset with ocean values set to zero and further processed by set_land_to_nan.
    """
    ## for E3SM-2-0
    dsnew = ds.siconc.fillna(0)
    ds['siconc'] = dsnew
    newice = set_land_to_nan(ds)
    return newice

def shift_x(da):
    ## only for 'CAS-ESM2-0'
    ## area lon 0~359 ; ice lon 1~360 
    ## first modify the lon in areadata
    ## NOTE: no need, those two actually match
    a = da.isel({da.dims[1]:0})  # select lon=0
    a[da.dims[1]] = da[da.dims[1]][0].values+360  # resign lon = 360 to lon=0 
    b = da.sel({da.dims[1]:slice(1, None)})  # select lon = 1~359
    new = xr.concat([b, a], dim=da.dims[1])  # combine 1~359 & 360
    return new

def change_start_x(ds, newx):
    lon360 = (ds.newlon+360)%360 - newx
    if np.abs(lon360[0,0])>5:
        diffmin = np.argmin(np.abs(lon360[0,:]).values)
        part_a = ds.isel({lon360.dims[-1]: slice(diffmin, None)})
        part_b = ds.isel({lon360.dims[-1]: slice(0, diffmin)})
        newds = xr.concat([part_a, part_b], dim = lon360.dims[-1])
        return newds
    else:
        return ds

def flip_y(ds):
    dsa = ds.reindex({ds.dims[-2]: ds[ds.dims[-2]][::-1]})
    dsnew = dsa.assign_coords({dsa.dims[-2]: ds[ds.dims[-2]]})
    return dsnew

def cal_mld(sigma0, lev = 'lev'):
    """
    Calculate the mixed layer depth (MLD) from density difference.

    Parameters:
    sigma0 (xarray.DataArray): The density profile with depth.
    lev (str, optional): The name of the depth coordinate. Default is 'lev'.

    Returns:
    xarray.DataArray: The mixed layer depth.

    Notes:
    - The function calculates the MLD based on the depth where the density difference 
        from the density at 10 meters is less than 0.03 kg/m^3.
    - It performs a linear interpolation to find the exact depth where the density 
        difference equals 0.03 kg/m^3.
    - If the resulting depth is larger than the deepest depth of the ocean bottom, 
        the bottom layer depth is used.
    """
    # Function for calculate mld from density difference (den - den10) and depth
    # Return mixed layer depth 
    b0 = sigma0[lev].where(~sigma0.isnull()).max(dim = lev) ## get bottom topography
    sigma0_10 = sigma0.interp({lev: 10}) 
    ## find the deepest layer where den - den10 < 0.03
    mld0 = sigma0[lev].where(sigma0 - sigma0_10 < 0.03).max(dim = lev)
    ## find the next layer 
    mld1 = sigma0[lev].where(sigma0[lev]>mld0).min(lev)
    cal_min = sigma0.where(sigma0[lev]>=mld0).min(lev) ## density of the shallow layer (this layer den - den10 <= 0.03)
    cal_max = sigma0.where(sigma0[lev]>=mld1).min(lev) ## density of the deep layer (this layer den - den10 > 0.03)

    ## simple linear interpolation to get the depth where den - den10 = 0.03
    mld2 = (mld1 - mld0)/(cal_max - cal_min) * (sigma0_10 + 0.03 - cal_min) + mld0 
    ## if the resulting depth is larger than the deepest depth of the ocean bottom, use the bottom layer 
    mld = xr.where(mld0 >= b0, b0, mld2)
    return mld

def check_lev_unit(levname, da):
    # check unit of depth and convert 
    # unit: cm --> m
    if 'units' in da[levname].attrs:
        if da[levname].attrs['units'] == 'centimeters':
            da[levname] = da[levname]/100
    return da

def connect_dask_cluster():
    # Create a PBS cluster object
    import dask
    from dask_jobqueue import PBSCluster
    from dask.distributed import Client
    cluster = PBSCluster(
        job_name        = 'dask',
        queue           = 'casper',
        walltime        = '20:00:00',
        log_directory   = 'dask-logs',
        cores           = 1,
        memory          = '8GiB',
        resource_spec   = 'select=1:ncpus=1:mem=8GB',
        processes       = 1,
        local_directory = '${SCRATCH}/dask_scratch/pbs.$PBS_JOBID/dask/spill',
        interface       = 'ext',
        silence_logs    = 'error',
    )
    cluster.adapt(minimum=2, maximum=100)
    client = Client(cluster)
    return client, cluster

def ReadDataFromCat(cat, **kwargs):
    query = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        query[key] = value
    if 'experiment_id' not in kwargs:
        query['experiment_id'] = 'piControl'
    if 'source_id' not in kwargs:
        query['source_id'] = 'GFDL-CM4'
    cat_subset = cat.search(**query)
    return cat_subset


def ReadDataFromNCAR(url = '/glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json', show_only=False, **kwargs):
    import intake
    cat = intake.open_esm_datastore(url)
    cat_subset = ReadDataFromCat(cat, **kwargs)
    if show_only:
        return cat_subset.df
    else:
        if len(cat_subset.df) > 0:
            dataset = cat_subset.to_dataset_dict()
            ds = dataset[list(dataset)[0]]
            ds = ds.isel(member_id=0).reset_coords('member_id', drop = True)
            ds = ds.isel(dcpp_init_year=0).reset_coords('dcpp_init_year', drop = True)
            return ds
        else:
            print("No Data")
            return None

def compute_and_save(savename, savepath, savedata) -> None:
    if not ispickleexists(savename, savepath):
        savedata = savedata.compute()
        savepickle(savename, savepath, savedata)
        print(f"{savename} saved")