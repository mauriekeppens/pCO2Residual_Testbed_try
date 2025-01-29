# libraries
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from skimage.filters import sobel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, max_error, mean_squared_error, mean_absolute_error, median_absolute_error
import keras
from keras import Sequential, regularizers
from keras.layers import Dense, BatchNormalization, Dropout
import gcsfs
fs = gcsfs.GCSFileSystem()

#===============================================
# Masks
#===============================================

def network_mask(topo_path,lsmask_path):
    """
    Creates network mask. This masks out regions in the NCEP land-sea mask 
    (https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html)
    to define the open ocean. 
    
    Regions removed include:
    - Coast : defined by sobel filter
    - Batymetry less than 100m
    - Arctic ocean : defined as North of 79N
    - Hudson Bay
    - caspian sea, black sea, mediterranean sea, baltic sea, Java sea, Red sea
    
    Parameters
    ----------
    topo_path : str
        Path to topography map
    
    lsmask_path : str
        Path to land sea map
    
    Returns
    ----------
    data : xarray.Dataset
        xarray dataset with masked out regions as written above.
    
    """
    
    ### topography
    topo_file_ext = topo_path.split('.')[-1] # getting if zarr or nc
    ds_topo = xr.open_mfdataset(topo_path, engine=topo_file_ext)
    ds_topo = ds_topo.roll(lon=180, roll_coords='lon')
    ds_topo['lon'] = np.arange(0.5, 360, 1)

    ### Loads grids
    # land-sea mask
    # land=0, sea=1
    
    lsmask_file_ext = topo_path.split('.')[-1] # getting if zarr or nc
    ds_lsmask = xr.open_mfdataset(lsmask_path, engine=lsmask_file_ext).sortby('lat').squeeze().drop('time')
    data = ds_lsmask['mask'].where(ds_lsmask['mask']==1)
    
    ### Define Latitude and Longitude
    lon = ds_lsmask['lon']
    lat = ds_lsmask['lat']
    
    ### Remove coastal points, defined by sobel edge detection
    coast = (sobel(ds_lsmask['mask'])>0)
    data = data.where(coast==0)
    
    ### Remove show sea, less than 100m
    ### This picks out the Solomon islands and Somoa
    data = data.where(ds_topo['Height']<-100)
    
    ### remove arctic
    data = data.where(~((lat>79)))
    data = data.where(~((lat>67) & (lat<80) & (lon>20) & (lon<180)))
    data = data.where(~((lat>67) & (lat<80) & (lon>-180+360) & (lon<-100+360)))

    ### remove caspian sea, black sea, mediterranean sea, and baltic sea
    data = data.where(~((lat>24) & (lat<70) & (lon>14) & (lon<70)))
    
    ### remove hudson bay
    data = data.where(~((lat>50) & (lat<70) & (lon>-100+360) & (lon<-70+360)))
    data = data.where(~((lat>70) & (lat<80) & (lon>-130+360) & (lon<-80+360)))
    
    ### Remove Red sea
    data = data.where(~((lat>10) & (lat<25) & (lon>10) & (lon<45)))
    data = data.where(~((lat>20) & (lat<50) & (lon>0) & (lon<20)))
    
    ### remove Okhtosk
    data = data.where(~((lat>50) & (lat<66) & (lon>136) & (lon<159)))

    data = data.roll(lon=180, roll_coords='lon')
    data['lon'] = np.arange(-179.5,180.5,1)
    
    return data

#===============================================
# Data prep functions
#===============================================

def calc_anom(array_1d, N_time, N_batch, array_mask0=None):
    
    """
    Anomaly calculation for variable. Assumes 2d array can be filled in C order (i.e. time was the first dimension that generated the 1d array).
    Note: can include an extra array to use to adjust for values that should be set to 0 (refers to array_mask0 parameter).
    
    Parameters
    ----------
    array_1d : np.array
        1d array
        
    N_time : int
        Length that the time dimension should be
        
    N_batch : int
         Number of months window for averaging  
         
    array_mask0 : boolean
        Extra array to use for adjusting values that should be set to 0. Defaults to None.
        
    Returns
    ----------
    output : np.array
        Calculated anomalies for selected variable (chl, mld, etc)
            
    """
    array_2d = array_1d.copy()
    if array_mask0 is not None:
        nan_mask = np.isnan(array_2d)
        mask0 = np.nan_to_num(array_mask0, nan=-1.0) <= 0
        array_2d[mask0] = np.nan
    array_2d = array_2d.reshape(N_time,-1,order='C')

    for i in range(-(-N_time//N_batch)):
        avg_val = np.nanmean(array_2d[(i*N_batch):((i+1)*N_batch),:])
        array_2d[(i*N_batch):((i+1)*N_batch),:] = array_2d[(i*N_batch):((i+1)*N_batch),:] - avg_val
    
    output = array_2d.flatten(order='C')
    if array_mask0 is not None:
        output[~nan_mask & mask0] = 0
    
    return output

#===============================================
# Calculate anoms from a mean seasonal cycle:
#===============================================

def calc_interannual_anom(df):
    
    """
    Calculate anomalies of a driver variable from the mean seasonal cycle.
    
    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe of feature variable to calculate anomalies for
        
    Returns
    ----------
    df_anom : pandas.Dataframe
        Dataframe of anomalies (monthly values minus mean climatology of that month)
    
    """
    
    # chl, sst, sss, xco2 all may have seasonal cycles in them
    DS = df.to_xarray() # get from multi-index back to an xarray for calculating mean seasonal cycle:
    DS_cycle = DS.groupby("time.month").mean("time")
    # Now get anomalies from this mean seasonal cycle:
    DS_anom = DS.groupby("time.month") - DS_cycle
    
    DS2 = xr.Dataset(
        {
        'anom':(['time','xlon','ylat'], DS_anom.data                    
        )},

        coords={
        'xlon': (['xlon'],DS.xlon.data),
        'ylat': (['ylat'],DS.ylat.data),
        'time': (['time'],DS.time.data)
        })
        
    df_anom = DS2.to_dataframe()

    return df_anom

#==============================================================================================

def log_or_0(array_1d):
  
    """
    Returns the log of input-1d array values, or 0 for any values less than or equal to 0.
    
    Parameters
    ----------
    1d array : np.array
        1d array
        
    Returns
    ----------
    output: log of 1d array or 0 for values <= 0
    
    """
    output_ma, output = array_1d.copy(), array_1d.copy()
    output_ma = np.ma.masked_array(output_ma, np.isnan(output_ma))
    output_ma = np.ma.log10(output_ma)
    output[~output_ma.mask] = output_ma[~output_ma.mask]
    output[output_ma.mask] = np.maximum(output[output_ma.mask],0)
    return output

#===============================================
# Loading in data and creating features
#===============================================

def import_member_data(ensemble_dir_head, ens, member,
                       xco2_path, socat_path, dates, files_ext='zarr'):
    
    """
    Preps member data and other driver variables for machine learning.
    
    Parameters
    ----------
    ensemble_dir_head : str
        Path to ensemble directory
    
    ens : str
        Name of model 
    
    member : str
        Name of member
    
    dates_str : str
        String made up of user-specified start and end dates to look for files (salinity, chlorophyll, etc) corresponding to those dates.
        
    files_ext : str
        String of either 'nc' or 'zarr' for proper file opening. Defaults to 'zarr'.
    
    xco2_path : str
        Path to xco2 file

    socat_path: str
        Path to socat file

    dates: np.array of pd.datetime objects
        Date range to clip all data to

    files_ext: str
        Either 'nc' or 'zarr'
    
    Returns
    ----------
    DS : xarray.Dataset
        Dataset for machine learning
    
    inputs['xco2'] : xarray.DataArray
        atmospheric co2 (xco2) concentrations for machine learning
    """
    
    member_dir = f"{ensemble_dir_head}/{ens}/{member}"
    if files_ext == 'nc':
        file_engine = 'netcdf4'
    elif files_ext == 'zarr':
        file_engine = 'zarr'

    member_path = fs.glob(f"{ensemble_dir_head}/{ens}/{member}/{ens}*.{files_ext}")[0]

    chl_clim_path = fs.glob(f"{ensemble_dir_head}/{ens}/{member}/chlclim*.{files_ext}")[0]

    member_data = xr.open_mfdataset('gs://'+member_path, engine=file_engine).sel(time=slice(str(dates[0]),str(dates[-1])))
    socat_mask_data = xr.open_mfdataset(socat_path, engine=file_engine).sel(time=slice(str(dates[0]),str(dates[-1])))
    tmp = xr.open_mfdataset('gs://'+chl_clim_path, engine=file_engine).chl_clim
    xco2 = xr.open_mfdataset(xco2_path, engine=file_engine).sel(time=slice(str(dates[0]),str(dates[-1])))
     
    inputs = {}
    
    time = member_data.time
    inputs['socat_mask'] = socat_mask_data.socat_mask
    inputs['sss'] = member_data.sss
    inputs['sst'] = member_data.sst
    inputs['chl'] = member_data.chl
    inputs['mld'] = member_data.mld
    inputs['pCO2_DIC'] = member_data.pco2_residual # non temperature component of pCO2 (what we will reconstruct)
    inputs['pCO2'] = member_data.spco2 # Reconstruct pCO2-pCO2T (difference) # actual pco2
    inputs['xco2'] = xco2.xco2
    
    # Create Chl Clim 1982-1997 and then 1998-2017 time varying CHL:

    ### THIS ISNT ACTUALLY BEING USED RIGHT NOW FOR TESTBED IN NOTEBOOK 2 
    tmp2 = member_data.chl

    chl_sat = np.empty(shape=(len(time),180,360))
    init_year = time.dt.year.values[0]
    
    for no,yr in enumerate(time.dt.year.values):
        if yr == 1998:
            idx_1998 = no
            break
            
    for yr in range(init_year,1998):
        chl_sat[(yr-init_year)*12:(yr-(init_year-1))*12,:,:]=tmp
    
    chl_sat[idx_1998:len(time),:,:]=tmp2[idx_1998:len(time),:,:]
    
    chl2 = xr.Dataset({'chl_sat':(["time","ylat","xlon"],chl_sat.data)},
                    coords={'time': (['time'],tmp2.time.data),
                    'ylat': (['ylat'],tmp2.ylat.data),
                    'xlon':(['xlon'],tmp2.xlon.data)})
    inputs['chl_sat'] = chl2.chl_sat

    for i in inputs:        
        if i != 'xco2':
            # inputs[i] = inputs[i].transpose('time', 'xlon', 'ylat')
            time_len = len(time)
            inputs[i].assign_coords(time=time[0:time_len])

    DS = xr.merge([inputs['sss'], inputs['sst'], inputs['mld'], inputs['chl'], inputs['pCO2_DIC'], inputs['pCO2'], inputs['socat_mask'],
                   inputs['chl_sat']], compat='override', join='override') 

    return DS, inputs['xco2']

def create_features(df, N_time, N_batch = 12):
    
    """
    Calculate anomalies and create time/positional driver variables for ML.
    
    Parameters
    ----------
    df: pd.Dataframe
        In-progress ML dataframe 

    N_time: int
        Number of months of data we use. For example, Jan1982-Dec2022 is 492 months.

    N_batch: int
        Number of months in a year to use (defaults to 12, 12 months in a year)
    
    Returns
    ----------
    df: pd.Dataframe
        Dataframe for machine learning
    
    """
    
    df['mld_log'] = log_or_0(df['mld'].values)
    # print(df['mld_log'],N_time)
    df['mld_anom'] = calc_anom(df['mld_log'].values, N_time, N_batch,array_mask0=df['mld'].values)
    df['mld_log_anom'] = calc_interannual_anom(df['mld_log'])
    
    df_mld = df.loc[(df['mld']>0),'mld']
    mld_grouped = df_mld.groupby(by=[df_mld.index.get_level_values('time').month, 'xlon','ylat']).mean()
    df = df.join(mld_grouped, on = [df.index.get_level_values('time').month, 'xlon','ylat'], rsuffix="_clim")
    df['mld_clim_log'] = log_or_0(df['mld_clim'].values)

    df['chl_log'] = log_or_0(df['chl'].values)
    df['chl_log_anom'] = calc_interannual_anom(df['chl_log'])
    df['chl_sat_log'] = log_or_0(df['chl_sat'].values)
    df['chl_sat_anom'] = calc_interannual_anom(df['chl_sat_log'])
    
    df.rename(columns={'SSS':'sss'}, inplace=True)
    df['sss_anom'] = calc_anom(df['sss'].values, N_time, N_batch)

    df.rename(columns={'SST':'sst'}, inplace=True)
    df['sst_anom'] = calc_interannual_anom(df['sst'])
    
    days_idx = df.index.get_level_values('time').dayofyear
    lon_rad = np.radians(df.index.get_level_values('xlon').to_numpy())
    lat_rad = np.radians(df.index.get_level_values('ylat').to_numpy())
    df['T0'], df['T1'] = [np.cos(days_idx * 2 * np.pi / 365), np.sin(days_idx * 2 * np.pi / 365)]
    df['A'], df['B'], df['C'] = [np.sin(lat_rad), np.sin(lon_rad)*np.cos(lat_rad), -np.cos(lon_rad)*np.cos(lat_rad)]
    return df

def create_inputs(ensemble_dir_head, ens, member, dates, N_time,
                  xco2_path, socat_path, topo_path, lsmask_path,
                  N_batch = 12): #N_batch should be number of months
    """
    Creates inputs for machine learning in notebook 02.

    Parameters
    ----------
    ensemble_dir_head : str
        Path to testbed members directory

    ens : str
        "Ensemble", aka Earth System Model (ESM) name

    member : str
        Experiment member ID 

    dates : pandas.DatetimeIndex
        List of datetimes in dataset

    N_time : int
        Number of months in dataset

    xco2_path : str
        Path to atmospheric co2 (xco2) projections file

    socat_path : str
        Path to SOCAT mask

    topo_path : str
        Path to topography mask

    lsmask_path : str
        Path to land-sea mask

    N_batch : int
        Number of months in a year period to average over
    
    """
    DS, DS_xco2 = import_member_data(ensemble_dir_head,ens,member,xco2_path,socat_path,dates)
    df = DS.to_dataframe()
    df = create_features(df, N_time, N_batch = N_batch)
    net_mask = np.tile(network_mask(topo_path,lsmask_path).transpose('lon','lat').to_dataframe()['mask'].to_numpy(),len(dates))
    df['net_mask'] = net_mask

    df['xco2'] = np.repeat(DS_xco2.values,180*360)

    return df

#===============================================
# Evaluation functions
#===============================================

def centered_rmse(y,pred):
    """
    Calculates "centered" root mean square error.
    
    Parameters
    ----------
    y : pandas.Dataframe
        "Truth" target values
    
    pred : pandas.Dataframe
        Predicted target values from ML
    
    Returns
    ----------
    np.sqrt(np.square((pred - pred_mean) - (y - y_mean)).sum()/pred.size) : float
        Centered RMSE
    
    """
    y_mean = np.mean(y)
    pred_mean = np.mean(pred)
    return np.sqrt(np.square((pred - pred_mean) - (y - y_mean)).sum()/pred.size)

def evaluate_test(y, pred):
    """
    Calculates ML test metrics/scores.
    
    Parameters
    ----------
    y : pandas.Dataframe
        Truth values
    
    pred : pandas.Dataframe
        Predicted values from ML
    
    Returns
    ----------
    scores : dict
        Dictionary of performance metrics and their values
        
    """
    scores = {
        'mse':mean_squared_error(y, pred),
        'mae':mean_absolute_error(y, pred),
        'medae':median_absolute_error(y, pred),
        'max_error':max_error(y, pred),
        'bias':pred.mean() - y.mean(),
        'r2':r2_score(y, pred),
        'corr':np.corrcoef(y,pred)[0,1],
        'cent_rmse':centered_rmse(y,pred),
        'stdev' :np.std(pred),
        'amp_ratio':(np.max(pred)-np.min(pred))/(np.max(y)-np.min(y)), # added when doing temporal decomposition
        'stdev_ref':np.std(y),
        'range_ref':np.max(y)-np.min(y),
        'iqr_ref':np.subtract(*np.percentile(y, [75, 25]))
        }
    return scores

#===============================================
# Train test split functions
#===============================================

def train_val_test_split(N, test_prop, val_prop, random_seeds, ens_count):
    
    """
    Get indeces for splitting training and test sets for ML.
    
    Parameters
    ----------
    N : Number of months
    
    test_prop : float
        Proportion of data to use for testing, percentage
    
    val_prop : float
        Proportion of data to use for validation, percentage
    
    random_seeds : list
        Random numbers/seeds for partitioning randomly
    
    ens_count : int
        Random seed stop point for ensemble member
    
    Returns
    ----------
    intermediate_idx : list
        Indeces for combined training and validation dataset
        
    train_idx : list
        Indeces for training dataset
        
    val_idx : list
        Indeces for validation dataset
        
    test_idx : list
        Indeces for testing dataset
    
    """
    
    intermediate_idx, test_idx = train_test_split(range(N), test_size=test_prop, random_state=random_seeds[0,ens_count])
    train_idx, val_idx = train_test_split(intermediate_idx, test_size=val_prop/(1-test_prop), random_state=random_seeds[1,ens_count])
    return intermediate_idx, train_idx, val_idx, test_idx

def apply_splits(X, y, train_val_idx, train_idx, val_idx, test_idx):
    
    """
    Uses splitting indeces found in 'train_val_test_split' to apply splits. 
    
    Parameters
    ----------
    X : pandas.Dataframe
        Dataframe of feature data
    
    y : pandas.Dataframe
        Dataframe of target data
    
    train_val_idx : list
        Indeces for combined training and validation dataset
    
    train_idx : list
        Indeces for training dataset
    
    val_idx : list
        Indeces for validation dataset
    
    test_idx : list
        Indeces for testing dataset
    
    Returns
    ----------
    X_train_val : pandas.Dataframe
        Combined train and validation set
    X_train : pandas.Dataframe
        Training set
    X_val : pandas.Dataframe
        Validation set
    X_test : pandas.Dataframe
        Test set
    y_train_val : pandas.Dataframe
        Target values for train and validation set
    y_train : pandas.Dataframe
        Target values for training set
    y_val : pandas.Dataframe
        Target values for validation set
    y_test : pandas.Dataframe
        Target values for test set
    """
    
    X_train_val = X[train_val_idx,:]
    X_train = X[train_idx,:]
    X_val = X[val_idx,:]
    X_test = X[test_idx,:]

    y_train_val = y[train_val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    return X_train_val, X_train, X_val, X_test, y_train_val, y_train, y_val, y_test

#===============================================
# Saving functions
#===============================================

def save_clean_data(df, data_output_dir, ens, member, dates):
    
    """
    Saves clean ML dataframe to be fed into ML algorithm
    
    Parameters
    ----------
    df : pd.Dataframe
        Dataframe for ML algo
    
    data_output_dir: str
        Path to directory to save dataframe for ML
        
    """
    
    print("Starting data saving process")

    init_date = str(dates[0].year) + format(dates[0].month,'02d')
    fin_date = str(dates[-1].year) + format(dates[-1].month,'02d')
    
    output_dir = f"{data_output_dir}/{ens}/{member}"
    fname = f"{output_dir}/MLinput_{ens}_{member.split('_')[-1]}_mon_1x1_{init_date}_{fin_date}.pkl"
    df.to_pickle(fname)
    print(f"{member} save complete")

def save_model(model, dates, model_output_dir, ens, member):
    
    """
    Saves ML model.
    
    Parameters
    ----------
    model: xgboost.sklearn.XGBRegressor
        XGBoost model once trained
        
    dates: pandas.DatetimeIndex
        List of dates of dataset
        
    model_output_dir: str
        Directory to store model
        
    ens: str
        Earth System Model name
        
    member: str
        Member index (e.g. 'member_r1i1p1f1')
    
    """
    
    print("Starting model saving process")

    init_date = str(dates[0].year) + format(dates[0].month,'02d')
    fin_date = str(dates[-1].year) + format(dates[-1].month,'02d')
    
    model_dir = f"{model_output_dir}/{ens}/{member}"

    model_fname = f"{model_dir}/model_pC02_2D_{ens}_{member.split('_')[-1]}_mon_1x1_{init_date}_{fin_date}.json"

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save_model(model_fname)
    print("Save complete")

def save_recon(DS_recon, dates, recon_output_dir, ens, member):
    
    """
    Saves ML reconstruction.
    
    Parameters
    ----------
    DS_recon: xarray.Dataset
        ML reconstruction, turned into an xarray from a pandas dataframe (to be saved as a .zarr file)
    
    dates: pandas.DatetimeIndex
        List of dates of dataset  
        
    recon_output_dir: str
        Directory to save reconstruction in
        
    ens: str
        Earth System Model name
        
    member: str
        Member index (e.g. 'member_r1i1p1f1')
    
    """
    
    print("Starting reconstruction saving process")

    init_date = str(dates[0].year) + format(dates[0].month,'02d')
    fin_date = str(dates[-1].year) + format(dates[-1].month,'02d')
    
    recon_dir = f"{recon_output_dir}/{ens}/{member}"
    
    recon_fname = f"{recon_dir}/recon_pC02residual_{ens}_{member}_mon_1x1_{init_date}_{fin_date}.zarr"

    print(recon_fname)
    DS_recon.to_zarr(f'{recon_fname}', mode='w')
    print("Save complete")
