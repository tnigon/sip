# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:35:45 2020

@author: nigo0024
"""
from ast import literal_eval
from copy import deepcopy
import fnmatch
import itertools as it
import math
import numpy as np
import os
import geopandas as gpd
import pandas as pd
import pathlib
import sys
import time

from hs_process import batch
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score

from scipy.stats import rankdata
import warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Plotting
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime
import subprocess
import globus_sdk

from extended_text_box import BoxStyle
from extended_text_box import ExtendedTextBox

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

# In[File management functions]
def hs_grid_search(hs_settings, msi_run_id, dir_out=None):
    '''
    Reads ``hs_settings`` and returns a dataframe with all the necessary
    information to execute each specific processing scenario. This enables
    searching over any number of image processsing scenarios.

    Folder name will be the index of df_grid for each set of outputs, so
    df_grid must be referenced to know which folder corresponds to which
    scenario.
    '''
    df_grid = pd.DataFrame(columns=hs_settings.keys())
    keys = hs_settings.keys()
    values = (hs_settings[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in it.product(*values)]
    for i in combinations:
        data = []
        for col in df_grid.columns:
            data.append(i[col])
        df_temp = pd.DataFrame(data=[data], columns=df_grid.columns)
        df_grid = df_grid.append(df_temp)
    df_grid = df_grid.reset_index(drop=True)
    # if csv is True:
    if dir_out is not None and os.path.isdir(dir_out):
        df_grid.to_csv(os.path.join(dir_out, 'msi_' + str(msi_run_id) + '_hs_settings.csv'), index=True)
    return df_grid

def get_idx_grid(dir_results_msi, msi_run_id, idx_min=0):
    '''
    Finds the index of the processing scenario based on files written to disk

    The problem I have, is that after 10 loops, I am running into a
    memoryerror. I am not sure why this is, but one thing I can try is to
    restart the Python instance and begin the script from the beginning after
    every main loop. However, I must determine which processing scenario I
    am currently on based on the files written to disk.

    Parameters:
        dir_results_msi: directory to search
        msi_run_id:
        start: The minimum idx_grid to return (e.g., if start=100, then
            idx_grid will be forced to be at least 100; it will be higher if
            other folders already exist and processing as been performed)
    '''
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    folders_all = [f for f in os.listdir(dir_results_msi) if os.path.isdir(os.path.join(dir_results_msi, f))]
    folders_out = []
    idx_list = []
    str_match = 'msi_' + str(msi_run_id) + '_'  # eligible folders must have this in their name
    for f in folders_all:
        if str_match in f:
            idx_grid1 = int(f.replace(str_match, ''))
            if idx_grid1 >= idx_min:
                idx_list.append(idx_grid1)
                folders_out.append(f)
    if len(idx_list) == 0:
        return idx_min
    for idx_grid2 in range(idx_min, max(idx_list)+2):
        if idx_grid2 not in idx_list: break
    # idx_dir = os.path.join(dir_results_msi, str_match + str(idx_grid2))
    # if not os.path.isdir(idx_dir):
    #     os.mkdir(idx_dir)
    return idx_grid2

def grid_n_levels(df_grid):
    n_clip = len(df_grid['clip'].unique())
    n_smooth = len(df_grid['smooth'].unique())
    n_bin = len(df_grid['bin'].unique())
    n_segment = len(df_grid['segment'].unique())
    return n_clip, n_smooth, n_bin, n_segment

def clean_df_grid(df_grid):
    # [(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]
    # for proc_step,  in ['clip', 'smooth', 'bin', 'segment', 'feature'] and i in range(10):
    #     print(proc_step, i)

    scenarios = [(idx, row_n, proc_step) for idx, row_n in df_grid.iterrows()
                 for proc_step in ['clip', 'smooth', 'bin', 'segment']]
    for idx, row_n, proc_step in scenarios:
        try:
            df_grid.loc[idx][proc_step] = literal_eval(row_n[proc_step])
        except ValueError:
            pass
    return df_grid
        # try:
        #     df_grid.loc[idx]['smooth'] = literal_eval(row_n['smooth'])
        # except ValueError:
        #     pass
        # try:
        #     df_grid.loc[idx]['bin'] = literal_eval(row_n['smooth'])
        # except ValueError:
        #     pass
        # try:
        #     df_grid.loc[idx]['segment'] = literal_eval(row_n['segment'])
        # except ValueError:
        #     pass

def recurs_dir(base_dir, search_ext='.bip', level=None):
    '''
    Searches all folders and subfolders recursively within <base_dir>
    for filetypes of <search_exp>.
    Returns sorted <outFiles>, a list of full path strings of each result.

    Parameters:
        base_dir: directory path that should include files to be returned
        search_ext: file format/extension to search for in all directories
            and subdirectories
        level: how many levels to search; if None, searches all levels

    Returns:
        out_files: include the full pathname, filename, and ext of all
            files that have ``search_exp`` in their name.
    '''
    if level is None:
        level = 1
    else:
        level -= 1
    d_str = os.listdir(base_dir)
    out_files = []
    for item in d_str:
        full_path = os.path.join(base_dir, item)
        if not os.path.isdir(full_path) and item.endswith(search_ext):
            out_files.append(full_path)
        elif os.path.isdir(full_path) and level >= 0:
            new_dir = full_path  # If dir, then search in that
            out_files_temp = recurs_dir(new_dir, search_ext)
            if out_files_temp:  # if list is not empty
                out_files.extend(out_files_temp)  # add items
    return sorted(out_files)

def retrieve_files(data_dir, panel_type, crop_type):
    '''
    Gets all the necessary files to be used in this scenario.

    Returns:
        df_crop (``pandas.DataFrame``): a dataframe containing all the cropping
            instructions for all the input datacubes
    '''
    fname_crop_info = os.path.join(data_dir, crop_type + '.csv')
    df_crop = pd.read_csv(fname_crop_info)
    df_crop['date'] = pd.to_datetime(df_crop['date'])
    df_crop['directory'] = df_crop.apply(lambda row : os.path.join(
        row['directory'], panel_type), axis = 1)

    gdf_wells_fname = os.path.join(data_dir, 'plot_bounds_wells.geojson')
    gdf_aerf_fname = os.path.join(data_dir, 'aerf_whole_field_sample_locs.geojson')
    gdf_wells = gpd.read_file(gdf_wells_fname)
    gdf_aerf = gpd.read_file(gdf_aerf_fname)
    return df_crop, gdf_wells, gdf_aerf

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def unique_datacubes(df_crop):
    '''
    Returns a list of unique datacubes from df_crop
    '''
    fname_list = []
    for idx, row in df_crop.iterrows():
        fname = os.path.join(row['directory'], row['name_short'] + row['name_long'] + row['ext'])
        if fname not in fname_list:
            fname_list.append(fname)

    df_files = pd.DataFrame(columns=['filename', 'size'])
    for fname in fname_list:
        if os.path.isfile(fname):
            data = [fname, os.path.getsize(fname)]
            df_temp = pd.DataFrame(data=[data], columns=df_files.columns)
            df_files = df_files.append(df_temp)
    print('Total number of datacubes to crop: {0}'.format(len(df_files)))
    print('Total file size: {0}\n'.format(convert_size(df_files['size'].sum())))

def check_processing(dir_out, ext='.bip', n_files=833):
    '''
    Checks directory; if it doesn't exist, it is created; if processing is
    finished, True is returned to indicate this step can be skipped.
    '''
    if not os.path.isdir(dir_out):
        pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
        # try:
        #     os.mkdir(dir_out)
        # except FileNotFoundError:
        #     os.mkdir(os.path.split(dir_out)[0])
        #     os.mkdir(dir_out)
        skip = False
    elif len(fnmatch.filter(os.listdir(dir_out), '*' + ext)) >= n_files:
        skip = True  # directory exists and contains many files; clipping was already done
    else:
        skip = False  # directory exists, but clipping was not completed
    return skip

def get_msi_segment_dir(row, level='segment'):
    '''
    Gets the msi directory equivalent of the directory where the .spec and
    other files are located following segmentation. These will be transfered
    to 2nd tier storage after testing/plotting.

    Parameters:
        row (``pd.Series``):
    '''
    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, _ = get_clip_type(row)
    smooth_type, _, _ = get_smooth_type(row)
    bin_type, _, _, _ = get_bin_type(row)
    segment_type, _, _, _, _, _, _ = get_segment_type(row)
    if level == 'segment':
        msi_seg_dir = '/'.join((panel_type, crop_type, clip_type, smooth_type, bin_type, segment_type))
    elif level == 'bin':
        msi_seg_dir = '/'.join((panel_type, crop_type, clip_type, smooth_type, bin_type))
    elif level == 'smooth':
        msi_seg_dir = '/'.join((panel_type, crop_type, clip_type, smooth_type))
    elif level == 'clip':
        msi_seg_dir = '/'.join((panel_type, crop_type, clip_type))
    elif level == 'crop':
        msi_seg_dir = '/'.join((panel_type, crop_type))
    return msi_seg_dir

def tier2_data_transfer(dir_base, row):
    '''
    Actually transfers the data to MSI 2nd tier storage

    I think this may take quite some time, so should always be used in a
    parallelized way

    Parameters:
        row (``pd.Series``):
    '''
    # msi_dir = 'results/ref_closest_panel/crop_plot/clip_none/smooth_none/seg_mcari2_50_upper'
    msi_seg_dir = get_msi_segment_dir(row)
    tier2_dir = os.path.join('S3://', msi_seg_dir)
    subprocess.call(['s3cmd', 'put', '-r', dir_base + mis_seg_dir2, tier2_dir])
    subprocess.call(['rm', '-r', dir_base + mis_seg_dir2])

def tier2_results_transfer(msi_result_dir, globus_client_id='684eb60a-9c5e-48af-929d-0880fd829173'):
    '''
    Transfers results from msi_0_000 folder from high performance storage
    to 2nd tier storage
    '''
    tier2_dir = os.path.join('S3://results', msi_result_dir)
    subprocess.call(['s3cmd', 'put', '-r', dir_base + msi_result_dir, tier2_dir])
    subprocess.call(['rm', '-r', dir_base + msi_result_dir])


def get_globus_data_dir(dir_base, msi_run_id, row,
                        msi_base='/home/yangc1/public',
                        level='segment'):
    '''
    Gets the data directory to transfer all files

    Parameters:
        level (``str``): The data directory level to transfer; must
            be one of ['segment', 'bin', 'smooth', 'clip', 'crop'].
    '''
    msi_seg_dir = get_msi_segment_dir(row, level=level)
    dest_base_dir = os.path.basename(dir_base) + '_msi_run_' + str(msi_run_id)
    dir_source_data = '/'.join(
        (msi_base, os.path.basename(dir_base), 'data', msi_seg_dir + '/'))
    dir_dest_data = '/'.join(
        ('/' + dest_base_dir, 'data', msi_seg_dir + '/'))
    return dir_source_data, dir_dest_data

def get_globus_results_dir(dir_base, msi_run_id, row, msi_base='/home/yangc1/public'):
    label_base = 'msi_' + str(msi_run_id) + '_' + str(row.name).zfill(3)
    dest_base_dir = os.path.basename(dir_base) + '_msi_run_' + str(msi_run_id)
    dir_source_results = '/'.join(
        (msi_base,  os.path.basename(dir_base), 'results',
         'msi_' + str(msi_run_id) + '_results', label_base + '/'))
    dir_dest_results = '/'.join(
        ('/' + dest_base_dir, 'results',
         'msi_' + str(msi_run_id) + '_results', label_base + '/'))
    return dir_source_results, dir_dest_results

def globus_autoactivate(tc, endpoint, if_expires_in=7200):
    r = tc.endpoint_autoactivate(endpoint, if_expires_in=if_expires_in)
    if (r["code"] == "AutoActivationFailed"):
        print("Endpoint requires manual activation, please open "
              "the following URL in a browser to activate the "
              "endpoint:")
        print("https://app.globus.org/file-manager?origin_id=%s"
              % endpoint)

def globus_transfer(dir_source_data, dir_dest_data, TRANSFER_REFRESH_TOKEN, client,
                    TRANSFER_TOKEN, label=None, delete=True,
                    source_endpoint='d865fc6a-2db3-11e6-8070-22000b1701d1',
                    dest_endpoint='fb6f1c6b-86b1-11e8-9571-0a6d4e044368'):
    '''
    Transfers results from msi_0_000 folder from high performance storage
    to 2nd tier storage using the GLOBUS Python SDK API. This required some
    setup (see "globus_token.py")


    dir_source_data = '/home/yangc1/public/hs_process/results/msi_1_results/'
    dir_dest_data = 'hs_process/results/msi_1_results'
    '''
    if source_endpoint is None:
        source_endpoint = tc.endpoint_search(filter_fulltext='umnmsi#home')
    if dest_endpoint is None:
        # dest_endpoint = tc.endpoint_search(filter_fulltext='umnmsi#tier2')
        tier2_id = 'fb6f1c6b-86b1-11e8-9571-0a6d4e044368'
        dest_endpoint = tc.get_endpoint(tier2_id)

    # First, get the transfer client using access tokens
    authorizer = globus_sdk.RefreshTokenAuthorizer(
        TRANSFER_REFRESH_TOKEN, client, access_token=TRANSFER_TOKEN)
    tc = globus_sdk.TransferClient(authorizer=authorizer)
    globus_autoactivate(tc, source_endpoint)
    globus_autoactivate(tc, dest_endpoint)

    submission_id = None
    # see: https://globus-sdk-python.readthedocs.io/en/stable/clients/transfer/#helper-objects
    tdata = globus_sdk.TransferData(  # initialize data transfer
        tc, source_endpoint=source_endpoint, destination_endpoint=dest_endpoint,
        label=label, submission_id=submission_id,
        sync_level=2, verify_checksum=False, preserve_timestamp=False,
        encrypt_data=False, deadline=None, recursive_symlinks='ignore')

    tdata.add_item(dir_source_data, dir_dest_data, recursive=True)  # add directory
    transfer_result = tc.submit_transfer(tdata)
    print("GLOBUS TRANSFER task_id:", transfer_result["task_id"])

    if delete is True:
        print('Waiting for transfer {0} to complete...'
              ''.format(transfer_result['task_id']))
        c = it.count(1)
        while not tc.task_wait(transfer_result['task_id'], timeout=60):
            print('Transfer {0} has not yet finished; transfer submitted {1} '
                  'minute(s) ago'.format(transfer_result['task_id'], next(c)))
        print('DONE.')
        ddata = globus_sdk.DeleteData(tc, source_endpoint, recursive=True)
        ddata.add_item(dir_source_data)
        delete_result = tc.submit_delete(ddata)
        print("GLOBUS DELETE task_id:", delete_result['task_id'])
        return transfer_result, delete_result
    return transfer_result

def restart_script():
    print("argv was", sys.argv)
    print("sys.executable was", sys.executable)
    print("restart now")
    # os.execv(sys.executable, ['python'] + sys.argv)
    os.execv(sys.executable, ['python', __file__] + sys.argv[1:])

def save_n_obs(dir_results_meta, df_join, msi_run_id, grid_idx, y_label, feat):
    fname = os.path.join(dir_results_meta, 'msi_{0}_n_observations.csv'
                         ''.format(msi_run_id))
    if not os.path.exists(fname):
        with open(fname, 'w+') as f:
            f.write('msi_run_id, grid_idx, y_label, feature, obs_n\n')

    with open(fname, 'a+') as f:
        f.write('{0}, {1}, {2}, {3}, {4}\n'
                ''.format(msi_run_id, grid_idx, y_label, feat, len(df_join)))

# In[Timing functions]
def time_setup_img(dir_out, msi_run_id):
    '''
    Set up times dictionary and save to file for every loop to append a new
    row
    '''
    cols = ['msi_run_id', 'grid_idx', 'n_jobs', 'time_start', 'time_end', 'time_total',
            'crop', 'clip', 'smooth', 'bin', 'segment']

    pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
    fname_times = os.path.join(dir_out, 'msi_' + str(msi_run_id) + '_time_imgproc.csv')
    if not os.path.isfile(fname_times):
        df_times = pd.DataFrame(columns=cols)
        df_times.to_csv(fname_times, index=False)
    time_dict = {i:[None] for i in cols}
    return time_dict

def time_setup_training(dir_out, msi_run_id):
    '''
    Set up times dictionary and save to file for every loop to append a new
    row
    '''
    cols = ['n_jobs', 'msi_run_id', 'grid_idx', 'y_label', 'feats',
            'time_start', 'time_end', 'time_total',
            # 'init1', 'init2', 'init3',
            'load_ground', 'load_spec', 'join_data', 'feat_sel', 'tune',
            'test', 'plot']
    pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
    fname_times = os.path.join(dir_out, 'msi_' + str(msi_run_id) + '_time_train.csv')
    if not os.path.isfile(fname_times):
        df_times = pd.DataFrame(columns=cols)
        df_times.to_csv(fname_times, index=False)
    time_dict = {i:[None] for i in cols}
    return time_dict

# def time_setup_training(dir_out, y_label_list, extra_feats_names, msi_run_id):
#     '''
#     Set up times dictionary and save to file for every loop to append a new
#     row
#     '''
#     cols = ['n_jobs', 'msi_run_id', 'grid_idx',
#             'time_start', 'time_end', 'time_total', 'sttp-init']
#     for y_label in y_label_list:
#         for feat_name in extra_feats_names:
#             col_str = 'sttp-' + y_label + '-' + feat_name
#             cols.append(col_str)

#     if not os.path.isdir(dir_out):
#         os.mkdir(dir_out)
#     fname_times = os.path.join(dir_out, 'msi_' + str(msi_run_id) + '_train_runtime.csv')
#     if not os.path.isfile(fname_times):
#         df_times = pd.DataFrame(columns=cols)
#         df_times.to_csv(fname_times, index=False)
#     time_dict = {i:[None] for i in cols}
#     return time_dict

def append_times(dir_out, time_dict, msi_run_id):
    '''
    Appends time info to the 'runtime.csv' in ``dir_out``

    Parameters:
        time_dict (``dict``): contains the time delta or datetime string to
            be written to each .csv column
    '''
    time_end = datetime.now()
    time_dict['time_end'] = [str(time_end)]
    time_start = datetime.strptime(
        time_dict['time_start'][0], '%Y-%m-%d %H:%M:%S.%f')
    time_total = time_end - time_start
    time_dict['time_total'] = [str(time_total)]

    if 'segment' in time_dict.keys():
        fname = 'msi_' + str(msi_run_id) + '_time_imgproc.csv'
    else:
        fname = 'msi_' + str(msi_run_id) + '_time_train.csv'
    fname_times = os.path.join(dir_out, fname)
    df_time = pd.DataFrame.from_dict(time_dict)
    df_time.to_csv(fname_times, header=None, mode='a', index=False,
                   index_label=df_time.columns)
    time_dict_null = dict.fromkeys(time_dict.keys(), [None])
    return time_dict_null

def time_loop_init(time_dict, msi_run_id, grid_idx, n_jobs):
    '''
    Initialization function for keeping track of time. Returns ``time_dict``,
        which will hold the time delta or datetime string for each step in the
        script.
    '''
    time_start = datetime.now()
    time_dict['n_jobs'] = [n_jobs]
    time_dict['msi_run_id'] = [msi_run_id]
    time_dict['grid_idx'] = [grid_idx]
    time_dict['time_start'] = [str(time_start)]
    return time_dict, time_start

def time_step(time_dict, key, time_last):
    '''
    Calculates the time since time_last and adds it to ``time_dict`` for the
    appropriate key.
    '''
    time_new = datetime.now()
    time_dif = time_new - time_last
    time_dict[key] = [str(time_dif)]
    return time_dict, time_new

# In[Count processed image functions]

def proc_files_count_setup(dir_out, msi_run_id):
    '''
    Set up processed files dict and save to file for every loop to append a new
    row
    '''
    cols = ['n_jobs', 'msi_run_id', 'grid_idx', 'processed']

    pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
    fname_n_files = os.path.join(dir_out, 'msi_' + str(msi_run_id) + '_imgproc_n_files.csv')
    if not os.path.isfile(fname_n_files):
        df_proc = pd.DataFrame(columns=cols)
        df_proc.to_csv(fname_n_files, index=False)
    proc_dict = {i:[None] for i in cols}
    return proc_dict

def proc_files_count(proc_dict, n_jobs, msi_run_id, key, dir_data, row, ext='.spec'):
    '''
    Calculates the time since time_last and adds it to ``time_dict`` for the
    appropriate key.
    '''
    proc_dict['n_jobs'] = [n_jobs]
    proc_dict['msi_run_id'] = [msi_run_id]
    # dir_data has to be explicit if we're going to do this for every single level..
    # n_files_proc = len(recurs_dir(get_spec_data(dir_data, row, feat='reflectance'), search_ext=ext))
    n_files_proc = len(fnmatch.filter(os.listdir(
        get_spec_data(dir_data, row, feat='reflectance')), '*' + ext))
    proc_dict[key] = [n_files_proc]
    return proc_dict, n_files_proc

def proc_files_append(dir_out, proc_dict, msi_run_id):
    '''
    Appends time info to the '_imgproc_n_files.csv' in ``dir_out``

    Parameters:
        proc_dict (``dict``): contains the n processed files to be written to
        each .csv column
    '''
    fname = 'msi_' + str(msi_run_id) + '_imgproc_n_files.csv'
    fname_n_files = os.path.join(dir_out, fname)
    df_proc = pd.DataFrame.from_dict(proc_dict)
    df_proc.to_csv(fname_n_files, header=None, mode='a', index=False,
                   index_label=df_proc.columns)
    proc_dict_null = dict.fromkeys(proc_dict.keys(), [None])
    return proc_dict_null

# In[Image processing functions]
def print_details(row):
    print('\nProcessing scenario ID: {0}'.format(row.name))
    print('Panels type: {0}'.format(row['dir_panels']))
    print('Crop type: {0}'.format(row['crop']))
    print('Clip type: {0}'.format(row['clip']))
    print('Smooth type: {0}'.format(row['smooth']))
    print('Bin type: {0}'.format(row['bin']))
    print('Segment type: {0}'.format(row['segment']))

def get_clip_type(row):
    '''
    Determines the clip type being used in this scenario (and updates
    wl_bands accordingly)

    Parameters:
        row (``pd.Series``):
    '''
    if pd.isnull(row['clip']):
        clip_type = 'clip_none'
        wl_bands = row['clip']
    elif len(row['clip']['wl_bands']) == 2:
        clip_type = 'clip_ends'
        wl_bands = row['clip']['wl_bands']
    elif len(row['clip']['wl_bands']) == 4:
        clip_type = 'clip_all'
        wl_bands = row['clip']['wl_bands']
    return clip_type, wl_bands

def get_smooth_type(row):
    '''
    Determines the smooth type being used in this scenario and returns window
    size and order for use in file names, etc.
    '''
    if pd.isnull(row['smooth']):
        smooth_type = 'smooth_none'
        window_size = row['smooth']
        order = row['smooth']
    else:
        window_size = row['smooth']['window_size']
        order = row['smooth']['order']
        smooth_type = 'smooth_window_{0}'.format(window_size)
    return smooth_type, window_size, order

def get_bin_type(row):
    '''
    Determines the bin type being used in this scenario (and updates
    accordingly)

    Parameters:
        row (``pd.Series``):
    '''
    if pd.isnull(row['bin']):
        method_bin = None
    else:
        method_bin = row['bin']['method']
    if method_bin == 'spectral_mimic':
        sensor = row['bin']['sensor']
        bandwidth = None
        bin_type = 'bin_mimic_{0}'.format(sensor.replace('-', '_'))
    elif method_bin == 'spectral_resample':
        sensor = None
        bandwidth = row['bin']['bandwidth']
        bin_type = 'bin_resample_{0}nm'.format(bandwidth)
    else:
        sensor = None
        bandwidth = None
        bin_type = 'bin_none'

    return bin_type, method_bin, sensor, bandwidth

def get_segment_type(row):
    '''
    Determines the segment type being used in this scenario and returns other relevant
    information for use in file names, etc.
    '''
    if pd.isnull(row['segment']):
        method = None
        wl1 = None
        wl2 = None
        wl3 = None
        mask_percentile = None
        mask_side = None
        segment_type = 'seg_none'
    elif row['segment']['method'] == 'mcari2':
        method = row['segment']['method']
        wl1 = row['segment']['wl1']
        wl2 = row['segment']['wl2']
        wl3 = row['segment']['wl3']
        mask_percentile = row['segment']['mask_percentile']
        mask_side = row['segment']['mask_side']
        if isinstance(mask_percentile, list):
            mask_pctl_print = '_'.join([str(x) for x in mask_percentile])
            segment_type = 'seg_{0}_{1}_{2}'.format(method, mask_pctl_print, get_side_inverse(mask_side))
        else:
            segment_type = 'seg_{0}_{1}_{2}'.format(method, mask_percentile, get_side_inverse(mask_side))
    elif row['segment']['method'] == 'ndi':
        method = row['segment']['method']
        wl1 = row['segment']['wl1']
        wl2 = row['segment']['wl2']
        wl3 = None
        mask_percentile = row['segment']['mask_percentile']
        mask_side = row['segment']['mask_side']
        if isinstance(mask_percentile, list):
            mask_pctl_print = '_'.join([str(x) for x in mask_percentile])
            segment_type = 'seg_{0}_{1}_{2}'.format(method, mask_pctl_print, get_side_inverse(mask_side))
        else:
            segment_type = 'seg_{0}_{1}_{2}'.format(method, mask_percentile, get_side_inverse(mask_side))
    elif row['segment']['method'] == ['mcari2', 'mcari2']:
        method = row['segment']['method']
        wl1 = row['segment']['wl1']
        wl2 = row['segment']['wl2']
        wl3 = row['segment']['wl3']
        mask_percentile = row['segment']['mask_percentile']
        mask_side = row['segment']['mask_side']
        segment_type = 'seg_{0}_between_{1}_{2}_pctl'.format(method[0], mask_percentile[0], mask_percentile[1])
    elif isinstance(row['segment']['method'], list) and row['segment']['method'][1] != 'mcari2':
        method = row['segment']['method']
        wl1 = row['segment']['wl1']
        wl2 = row['segment']['wl2']
        wl3 = row['segment']['wl3']
        mask_percentile = row['segment']['mask_percentile']
        mask_side = row['segment']['mask_side']
        if method == ['mcari2', [545, 565]]:
            segment_type = 'seg_mcari2_{0}_{1}_green_{2}_{3}'.format(mask_percentile[0], get_side_inverse(mask_side[0]), mask_percentile[1], get_side_inverse(mask_side[1]))
        elif method == ['mcari2', [800, 820]]:
            segment_type = 'seg_mcari2_{0}_{1}_nir_{2}_{3}'.format(mask_percentile[0], get_side_inverse(mask_side[0]), mask_percentile[1], get_side_inverse(mask_side[1]))
    return segment_type, method, wl1, wl2, wl3, mask_percentile, mask_side

def smooth_get_base_dir(dir_data, panel_type, crop_type, clip_type):
    '''
    Gets the base directory for the smoothed images
    '''
    if clip_type == 'clip_none':
        base_dir = os.path.join(dir_data, panel_type, crop_type)
    else:
        base_dir = os.path.join(dir_data, panel_type, crop_type, clip_type)
    return base_dir

def bin_get_base_dir(dir_data, panel_type, crop_type, clip_type, smooth_type):
    '''
    Gets the base directory for the binned images
    '''
    if smooth_type == 'smooth_none' and clip_type == 'clip_none':
        base_dir = os.path.join(dir_data, panel_type, crop_type)
    elif smooth_type == 'smooth_none' and clip_type != 'clip_none':
        base_dir = os.path.join(dir_data, panel_type, crop_type, clip_type)
    else:  # smooth_type != 'smooth_none'
        base_dir = os.path.join(dir_data, panel_type, crop_type, clip_type, smooth_type)
    return base_dir

def seg_get_base_dir(dir_data, panel_type, crop_type, clip_type, smooth_type,
                     bin_type):
    '''
    Gets the base directory for the segmented images
    '''
    if bin_type == 'bin_none' and smooth_type == 'smooth_none' and clip_type == 'clip_none':
        base_dir = os.path.join(dir_data, panel_type, crop_type)
    elif bin_type == 'bin_none' and smooth_type == 'smooth_none' and clip_type != 'clip_none':
        base_dir = os.path.join(dir_data, panel_type, crop_type, clip_type)
    elif bin_type == 'bin_none' and smooth_type != 'smooth_none':
        base_dir = os.path.join(dir_data, panel_type, crop_type, clip_type, smooth_type)
    else:  # bin_type != 'bin_none'
        base_dir = os.path.join(dir_data, panel_type, crop_type, clip_type, smooth_type, bin_type)
    return base_dir

def crop(df_crop, panel_type, dir_out_crop, out_force=True, n_files=854,
         gdf_aerf=None, gdf_wells=None):
    '''
    Gets the cropping info for each site and crops all the datacubes

    Parameters:
        df_crop (``pandas.DataFrame``): a dataframe containing all the cropping
            instructions for all the input datacubes
    '''
    if check_processing(dir_out_crop, ext='.bip', n_files=n_files):
        return

    folder_name = None
    name_append = os.path.split(dir_out_crop)[-1].replace('_', '-')

    if panel_type == 'ref_closest_panel':  # data doesn't exist for 7/23 aerf small plot
        df_crop_aerf_small = df_crop[(df_crop['study'] == 'aerfsmall') &
                                     (df_crop['date'] != datetime(2019, 7, 23))]
    else:
        df_crop_aerf_small = df_crop[df_crop['study'] == 'aerfsmall']
    df_crop_aerf_whole = df_crop[df_crop['study'] == 'aerffield']
    df_crop_wells_18 = df_crop[(df_crop['study'] == 'wells') &
                               (df_crop['date'].dt.year == 2018)]
    df_crop_wells_19 = df_crop[(df_crop['study'] == 'wells') &
                               (df_crop['date'].dt.year == 2019)]

    hsbatch = batch()
    hsbatch.io.set_io_defaults(force=out_force)
    hsbatch.spatial_crop(fname_sheet=df_crop_wells_18, method='many_gdf',
                         gdf=gdf_wells, base_dir_out=dir_out_crop,
                         folder_name=folder_name, name_append=name_append)
    hsbatch.spatial_crop(fname_sheet=df_crop_wells_19, method='many_gdf',
                         gdf=gdf_wells, base_dir_out=dir_out_crop,
                         folder_name=folder_name, name_append=name_append)
    hsbatch.spatial_crop(fname_sheet=df_crop_aerf_small, method='single',
                         base_dir_out=dir_out_crop, folder_name=folder_name,
                         name_append=name_append)
    hsbatch.spatial_crop(fname_sheet=df_crop_aerf_whole, method='many_gdf',
                         gdf=gdf_aerf, base_dir_out=dir_out_crop,
                         folder_name=folder_name, name_append=name_append)

def chunk_by_n(array, n):
    np.random.shuffle(array)  # studies have different size images, so shuffling makes each chunk more similar
    arrays = np.array_split(array, n)
    list_out = []
    for l in arrays:
        list_out.append(l.tolist())
    return list_out

def check_missed_files(fname_list, base_dir_out, ext_out, f_pp, row, base_dir_f,
                       out_force, lock):
    '''
    Check if any files were not processed that were supposed to be processed.

    Parameters:
        f_pp (function): The parallel processing function to run to
            complete the processing of the missing files.
        **kwargs (dict): keyword arguments to pass to f_pp.
    '''
    # Goal: take out filepaths and end text
    to_process = [os.path.splitext(os.path.basename(i))[0].rsplit('-')[0] for i in fname_list]
    name_ex, ext = os.path.splitext(fname_list[0])
    end_str = '-' + '-'.join(name_ex.split('-')[1:]) + ext
    # Find processed files without filepaths and end text
    # base_dir_spec = os.path.join(dir_out_mask, 'reflectance')
    fname_list_complete = fnmatch.filter(os.listdir(base_dir_out), '*' + ext_out)  # no filepath
    processed = [os.path.splitext(i)[0].split('-')[0] for i in fname_list_complete]

    missed = [f for f in to_process if f not in processed]
    base_dir = os.path.dirname(fname_list[0])
    fname_list_missed = [os.path.join(base_dir, f + end_str) for f in missed]
    if len(missed) > 0:
        print('There were {0} images that slipped through the cracks. '
              'Processing them manually now...\n'.format(len(missed)))
        print('Directory: {0}'.format(base_dir))
        print('Here are the missed images:\n{0}\n'.format(missed))
        f_pp(fname_list_missed, row, base_dir_f, out_force, lock)

def clip(dir_data, row, out_force=True, n_files=854):
    '''
    Clips each of the datacubes according to instructions in df_grid
    '''
    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, wl_bands = get_clip_type(row)
    base_dir = os.path.join(dir_data, panel_type, crop_type)
    dir_out_clip = os.path.join(dir_data, panel_type, crop_type, clip_type)
    if check_processing(dir_out_clip, ext='.bip', n_files=n_files):
        return

    if wl_bands is not None:
        folder_name = None
        name_append = os.path.split(dir_out_clip)[-1].replace('_', '-')
        hsbatch = batch()
        hsbatch.io.set_io_defaults(force=out_force)
        hsbatch.spectral_clip(base_dir=base_dir, folder_name=folder_name,
                              name_append=name_append,
                              base_dir_out=dir_out_clip,
                              wl_bands=wl_bands)
    else:
        print('Clip: ``clip_type`` is None, so there is nothing to process.')

def clip_f_pp(fname_list_clip, wl_bands, dir_out_clip, out_force, lock):
    '''
    Parallel processing: clips each of the datacubes according to instructions
    in df_grid organized for multi-core processing.

    These are the lines of code that have to be passed as a function to
    ProcessPoolExecutor()
    '''
    assert wl_bands is not None, ('``wl_bands`` must not be ``None``')
    hsbatch = batch(lock=lock)
    hsbatch.io.set_io_defaults(force=out_force)
        # clip2(fname_list_clip, wl_bands, dir_out_clip, hsbatch)
    # if wl_bands is not None:
    folder_name = None
    name_append = os.path.split(dir_out_clip)[-1].replace('_', '-')
    hsbatch.spectral_clip(fname_list=fname_list_clip, folder_name=folder_name,
                          name_append=name_append,
                          base_dir_out=dir_out_clip,
                          wl_bands=wl_bands)

def clip_pp(dir_data, row, n_jobs, out_force=True, n_files=854):
    '''
    Actual execution of band clipping via multi-core processing
    '''
    m = Manager()
    lock = m.Lock()

    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, wl_bands = get_clip_type(row)
    base_dir = os.path.join(dir_data, panel_type, crop_type)
    dir_out_clip = os.path.join(dir_data, panel_type, crop_type, clip_type)
    already_processed = check_processing(dir_out_clip, ext='.bip', n_files=n_files)
    if out_force is False and already_processed is True:
        fname_list = []
    elif clip_type == 'clip_none':
        fname_list = []
    else:
        fname_list = fnmatch.filter(os.listdir(base_dir), '*.bip')  # no filepath
        fname_list = [os.path.join(base_dir, f) for f in fname_list]
        # fname_list = recurs_dir(base_dir, search_ext='.bip', level=0)

    chunk_size = int(len(fname_list) / (n_jobs*2))
    if len(fname_list) == 0:
        print('Clip: ``clip_type`` is either None and there is nothing to '
              'process, or all the images are already processed.')
    else:
        chunks = chunk_by_n(fname_list, n_jobs*2)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            executor.map(
                clip_f_pp, chunks, it.repeat(wl_bands),
                it.repeat(dir_out_clip), it.repeat(out_force), it.repeat(lock))

        ext_out = '.bip'
        check_missed_files(fname_list, dir_out_clip, ext_out, clip_f_pp, row,
                           dir_out_clip, out_force, lock)

def smooth(dir_data, row, out_force=True, n_files=854):
    '''
    Smoothes each of the datacubes according to instructions in df_grid
    '''
    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, wl_bands = get_clip_type(row)
    smooth_type, window_size, order = get_smooth_type(row)
    base_dir = smooth_get_base_dir(dir_data, panel_type, crop_type, clip_type)
    dir_out_smooth = os.path.join(dir_data, panel_type, crop_type, clip_type,
                                  smooth_type)
    if check_processing(dir_out_smooth, ext='.bip', n_files=n_files):
        return

    if window_size is not None and order is not None:
        folder_name = None
        name_append = os.path.split(dir_out_smooth)[-1].replace('_', '-')
        hsbatch = batch()
        hsbatch.io.set_io_defaults(force=out_force)
        hsbatch.spectral_smooth(base_dir=base_dir, folder_name=folder_name,
                                name_append=name_append,
                                base_dir_out=dir_out_smooth,
                                window_size=window_size, order=order)
    else:
        print('Smooth: ``smooth_type`` is None, so there is nothing to process.')

def smooth_f_pp(fname_list_smooth, window_size, order, dir_out_smooth, out_force, lock):
    '''
    Parallel processing: smoothes each of the datacubes according to
    instructions in df_grid organized for multi-core processing.
    '''
    msg = ('``window_size`` must not be ``None``')
    assert window_size is not None and order is not None, msg
    hsbatch = batch(lock=lock)
    hsbatch.io.set_io_defaults(force=out_force)
        # clip2(fname_list_clip, wl_bands, dir_out_clip, hsbatch)
    # if wl_bands is not None:
    folder_name = None
    name_append = os.path.split(dir_out_smooth)[-1].replace('_', '-')
    hsbatch.spectral_smooth(
        fname_list=fname_list_smooth, folder_name=folder_name,
        name_append=name_append, base_dir_out=dir_out_smooth,
        window_size=window_size, order=order)

def smooth_pp(dir_data, row, n_jobs, out_force=True, n_files=854):
    '''
    Actual execution of band smoothing via multi-core processing
    '''
    m = Manager()
    lock = m.Lock()

    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, wl_bands = get_clip_type(row)
    smooth_type, window_size, order = get_smooth_type(row)
    base_dir = smooth_get_base_dir(dir_data, panel_type, crop_type, clip_type)
    dir_out_smooth = os.path.join(dir_data, panel_type, crop_type, clip_type,
                                  smooth_type)
    already_processed = check_processing(dir_out_smooth, ext='.bip', n_files=n_files)
    if out_force is False and already_processed is True:
        fname_list = []
    elif smooth_type == 'smooth_none':
        fname_list = []
    else:
        fname_list = fnmatch.filter(os.listdir(base_dir), '*.bip')  # no filepath
        fname_list = [os.path.join(base_dir, f) for f in fname_list]
        # fname_list = recurs_dir(base_dir, search_ext='.bip', level=0)

    if len(fname_list) == 0:
        print('Smooth: ``smooth_type`` is either None and there is nothing to '
              'process, or all the images are already processed.')
    else:
        np.random.shuffle(fname_list)  # studies have different size images, so shuffling makes each chunk more similar
        chunks = chunk_by_n(fname_list, n_jobs*2)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            executor.map(
                smooth_f_pp, chunks, it.repeat(window_size),
                it.repeat(order), it.repeat(dir_out_smooth),
                it.repeat(out_force), it.repeat(lock))

        ext = '.bip'
        check_missed_files(fname_list, dir_out_smooth, ext, smooth_f_pp, row,
                           dir_out_smooth, out_force, lock)

def bin_f_pp(fname_list_bin, row, dir_out_bin, out_force, lock):
    '''
    Parallel processing: spectral mimic/resampling for each of the datacubes
    according to instructions in df_grid organized for multi-core processing.
    '''
    bin_type, method_bin, sensor, bandwidth = get_bin_type(row)
    msg = ('``bin_type`` must not be ``None``')
    assert bin_type is not None, msg

    hsbatch = batch(lock=lock)
    hsbatch.io.set_io_defaults(force=out_force)
    folder_name = None
    name_append = os.path.split(dir_out_bin)[-1].replace('_', '-')
    if method_bin == 'spectral_resample':
        hsbatch.spectral_resample(
            fname_list=fname_list_bin, folder_name=folder_name,
            name_append=name_append, base_dir_out=dir_out_bin,
            bandwidth=bandwidth)
    elif method_bin == 'spectral_mimic':
        hsbatch.spectral_mimic(
            fname_list=fname_list_bin, folder_name=folder_name,
            name_append=name_append, base_dir_out=dir_out_bin,
            sensor=sensor)
    else:
        print('Bin: method "{0}" is not supported.'.format(method_bin))

def bin_pp(dir_data, row, n_jobs, out_force=True, n_files=854):
    '''
    Actual execution of spectral resampling/mimicking via multi-core processing
    '''
    m = Manager()
    lock = m.Lock()
    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, wl_bands = get_clip_type(row)
    smooth_type, window_size, order = get_smooth_type(row)
    bin_type, method_bin, sensor, bandwidth = get_bin_type(row)

    base_dir = bin_get_base_dir(dir_data, panel_type, crop_type, clip_type,
                                smooth_type)
    dir_out_bin = os.path.join(dir_data, panel_type, crop_type, clip_type,
                               smooth_type, bin_type)
    already_processed = check_processing(dir_out_bin, ext='.bip', n_files=n_files)
    if out_force is False and already_processed is True:
        fname_list = []
    elif bin_type == 'bin_none':
        fname_list = []
    else:
        fname_list = fnmatch.filter(os.listdir(base_dir), '*.bip')  # no filepath
        fname_list = [os.path.join(base_dir, f) for f in fname_list]
        # fname_list = recurs_dir(base_dir, search_ext='.bip', level=0)

    if len(fname_list) == 0:
        print('Bin: ``bin_type`` is either None and there is nothing to '
              'process, or all the images are already processed.')
    else:
        chunks = chunk_by_n(fname_list, n_jobs*2)
        chunk_avg = sum([len(i) for i in chunks]) / len(chunks)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            executor.map(
                bin_f_pp, chunks, it.repeat(row),
                it.repeat(dir_out_bin), it.repeat(out_force), it.repeat(lock))

        ext = '.bip'
        check_missed_files(fname_list, dir_out_bin, ext, bin_f_pp, row,
                           dir_out_bin, out_force, lock)

        # # Goal: take out filepaths and end text
        # to_process = [os.path.splitext(os.path.basename(i))[0].rsplit('-')[0] for i in fname_list]
        # name_ex, ext = os.path.splitext(fname_list[0])
        # end_str = '-' + '-'.join(name_ex.split('-')[1:]) + ext
        # # Find processed files without filepaths and end text
        # base_dir_seg = os.path.join(base_dir_bm, segment_type)
        # fname_list_complete = fnmatch.filter(os.listdir(base_dir_seg), '*' + '.spec')  # no filepath
        # processed = [os.path.splitext(i)[0].rsplit('-')[0] for i in fname_list_complete]

        # missed = [f for f in to_process if f not in processed]
        # # base_dir = os.path.dirname(fname_list[0])
        # fname_list_missed = [os.path.join(base_dir, f + end_str) for f in missed]
        # if len(missed) > 0:
        #     print('There were {0} images that slipped through the cracks. '
        #           'Processing them manually now...\n'.format(len(missed)))
        #     print('Here are the missed images:\n{0}\n'.format(missed))
        #     bin_f_pp(fname_list_missed, row, dir_out_bin, out_force, lock)

def parse_wls(wl, idx=0):
    '''
    Require that wl is a list; if a range of wls is desired, the min and max
    should be in the list.
    If there are two items in a list, and both are also lists, then it's a
    two-step segementation process.

    If two-step process (wl is a list of lists), then idx should be passed to
    indicate which list to use.
    '''
    if isinstance(wl, list) and len(wl) == 1:
        wl_n = int(wl[0])
        wl_str = str(int(wl[0]))
    elif isinstance(wl, list) and len(wl) == 2:
        if isinstance(wl[0], list):  # two-step process
            wl_n = int(np.mean(wl[idx]))
            wl_str = str(int(wl_n))
        else:  # take the average
            wl_n = int(np.mean(wl))
            wl_str = str(int(wl_n))
    elif isinstance(wl, int):
        wl_n = wl
        wl_str = str(wl)
    elif isinstance(wl, float):
        wl_n = int(wl)
        wl_str = str(int(wl))
    return wl_n, wl_str

def get_side_inverse(mask_side):
    '''
    Gets the inverse of mask side
    '''
    if mask_side == 'lower':
        mask_side_inv = 'upper'
    elif mask_side == 'upper':
        mask_side_inv = 'lower'
    elif mask_side == 'outside':
        mask_side_inv = 'between'
    elif mask_side == 'between':
        mask_side_inv = 'outside'
    return mask_side_inv

def seg_spec_and_derivative(fname_list_seg, dir_out_mask, name_append,
                            hsbatch):
    '''
    Calculates spectra and derivative spectra for all files and saves to
    "reflectance" and "derivative_x" folders.
    '''
    fname_list_names = [os.path.splitext(os.path.basename(f))[0].split('-')[0] for f in fname_list_seg]

    # If seg is None, then be sure to grab dirname(fname_list_seg[0])
    if os.path.split(dir_out_mask)[-1] == 'seg_none':
        dir_unmask = os.path.dirname(fname_list_seg[0])
        fname_sample = fnmatch.filter(os.listdir(dir_unmask), '*.bip')[0]
        name_append_dirname = os.path.splitext(fname_sample)[0].split('-', maxsplit=1)[-1]
        fname_list_mask = [os.path.join(dir_unmask, f) + '-' + name_append_dirname + '.bip' for f in fname_list_names]
    else:
        fname_list_mask = [os.path.join(dir_out_mask, f) + '-' + name_append + '.bip' for f in fname_list_names]

    hsbatch.cube_to_spectra(
        fname_list=fname_list_mask, base_dir_out=dir_out_mask,
        folder_name='reflectance', name_append=name_append,
        write_geotiff=False)

    dir_out_spec = os.path.join(dir_out_mask, 'reflectance')
    fname_list_der = [os.path.join(dir_out_spec, f) + '-' + name_append + '-mean.spec' for f in fname_list_names]
    name_append_der = name_append + '-derivative'
    hsbatch.spectra_derivative(
        fname_list=fname_list_der, name_append=name_append_der, order=1,
        base_dir_out=dir_out_mask, folder_name='derivative_1')
    hsbatch.spectra_derivative(
        fname_list=fname_list_der, name_append=name_append_der, order=2,
        base_dir_out=dir_out_mask, folder_name='derivative_2')
    hsbatch.spectra_derivative(
        fname_list=fname_list_der, name_append=name_append_der, order=3,
        base_dir_out=dir_out_mask, folder_name='derivative_3')

def seg_zero_step(fname_list_seg, base_dir_bm, hsbatch, row):
    '''
    During the analysis, we may want to include auxiliary features from the
    band math in addition to the spectral features. Therfore, we must be sure
    to perform the band math even if it is not used for segmentation so it is
    available during model training.

    base_dir can also be a list of filenames (implemented for multi-core
    processing)
    '''
    segment_type, _, _, _, _, _, _ = get_segment_type(row)

    dir_out_mask = os.path.join(base_dir_bm, segment_type)
    name_append = segment_type.replace('_', '-')
    seg_spec_and_derivative(fname_list_seg, dir_out_mask, name_append, hsbatch)

    folder_name = 'bm_mcari2'
    name_append = folder_name.replace('_', '-')
    hsbatch.segment_band_math(
        fname_list=fname_list_seg, folder_name=folder_name,
        name_append=name_append, base_dir_out=base_dir_bm, write_geotiff=False,
        method='mcari2', wl1=[800], wl2=[670], wl3=[550], plot_out=False,
        out_force=False)
    folder_name = 'bm_ndi'
    name_append = folder_name.replace('_', '-')
    hsbatch.segment_band_math(
        fname_list=fname_list_seg, folder_name=folder_name,
        name_append=name_append, base_dir_out=base_dir_bm, write_geotiff=False,
        method='ndi', wl1=[770, 800], wl2=[650, 680], plot_out=False,
        out_force=False)

def seg_one_step(fname_list_seg, base_dir_bm, hsbatch, row):
    '''
    Perform band math then performs one-step segmentation. "one-step" refers to
    having only a single masking step rather than two masking steps (e.g.,
    mask below 75th pctl then above 95th pctl)
    fname_list_seg should be a list of filenames (implemented for multi-core
    processing)
    '''
    segment_type, method, wl1, wl2, wl3, mask_percentile, mask_side = get_segment_type(row)

    if isinstance(method, str):
        folder_name = 'bm_{0}'.format(method)
        name_append = folder_name.replace('_', '-')
        base_dir_bm = os.path.join(base_dir_bm, folder_name)
        hsbatch.segment_band_math(
            fname_list=fname_list_seg, folder_name=None,
            name_append=name_append, base_dir_out=base_dir_bm, write_geotiff=False,
            method=method, wl1=wl1, wl2=wl2, wl3=wl3, plot_out=False,
            out_force=False)
    elif method == [545, 565]:
        folder_name = 'bm_green'
        name_append = folder_name.replace('_', '-')
        base_dir_bm = os.path.join(base_dir_bm, folder_name)
        hsbatch.segment_composite_band(
            fname_list=fname_list_seg, folder_name=None,
            name_append=name_append, base_dir_out=base_dir_bm, write_geotiff=False,
            wl1=method, list_range=True, plot_out=False, out_force=False)
    elif method == [800, 820]:
        folder_name = 'bm_nir'
        name_append = folder_name.replace('_', '-')
        base_dir_bm = os.path.join(base_dir_bm, folder_name)
        hsbatch.segment_composite_band(
            fname_list=fname_list_seg, folder_name=None,
            name_append=name_append, base_dir_out=base_dir_bm, write_geotiff=False,
            wl1=method, list_range=True, plot_out=False, out_force=False)

    dir_out_mask = os.path.join(os.path.split(base_dir_bm)[0], segment_type)
    name_append = segment_type.replace('_', '-')

    hsbatch.segment_create_mask(
        fname_list=fname_list_seg, mask_dir=base_dir_bm, folder_name=None,
        name_append=name_append, base_dir_out=dir_out_mask,
        write_datacube=True, write_spec=False, write_geotiff=False,
        mask_percentile=mask_percentile, mask_side=mask_side, out_force=True)

    seg_spec_and_derivative(fname_list_seg, dir_out_mask, name_append, hsbatch)

def seg_two_step(fname_list_seg, base_dir_bm, hsbatch, row):
    '''
    Performs band math then performs two-step segmentation. "two-step" refers to
    having more than a single masking step rather than a simple single masking
    step (e.g., mask only below 90th pctl)
    '''
    segment_type, methods, wl1, wl2, wl3, mask_percentiles, mask_sides = get_segment_type(row)
    mask_dirs = []
    for i, method in enumerate(methods):
        if isinstance(method, str):
        # if method == 'mcari2':
            folder_name = 'bm_{0}'.format(method)
            name_append = folder_name.replace('_', '-')
            mask_dirs.append(os.path.join(base_dir_bm, folder_name))
            hsbatch.segment_band_math(
                fname_list=fname_list_seg, folder_name=None,  name_append=name_append,
                base_dir_out=mask_dirs[i], write_geotiff=False, method=method,
                wl1=wl1[i], wl2=wl2[i], wl3=wl3[i], plot_out=False)
        elif method == [545, 565]:
            folder_name = 'bm_green'
            name_append = folder_name.replace('_', '-')
            mask_dirs.append(os.path.join(base_dir_bm, folder_name))
            hsbatch.segment_composite_band(
                fname_list=fname_list_seg, folder_name=None, name_append=name_append,
                base_dir_out=mask_dirs[i], write_geotiff=False, wl1=method,
                list_range=True, plot_out=False)
        elif method == [800, 820]:
            folder_name = 'bm_nir'
            name_append = folder_name.replace('_', '-')
            mask_dirs.append(os.path.join(base_dir_bm, folder_name))
            hsbatch.segment_composite_band(
                fname_list=fname_list_seg, folder_name=None, name_append=name_append,
                base_dir_out=mask_dirs[i], write_geotiff=False, wl1=method,
                list_range=True, plot_out=False)

    dir_out_mask = os.path.join(os.path.split(mask_dirs[0])[0], segment_type)
    name_append = segment_type.replace('_', '-')

    hsbatch.segment_create_mask(  # it would be much faster to write_spec, but then we have to reorganize reflectance, derivative folders
        fname_list=fname_list_seg, mask_dir=mask_dirs, folder_name=None,
        name_append=name_append, base_dir_out=dir_out_mask,
        write_datacube=True, write_spec=False, write_geotiff=False,
        mask_percentile=mask_percentiles, mask_side=mask_sides, out_force=False)

    seg_spec_and_derivative(fname_list_seg, dir_out_mask, name_append, hsbatch)

def seg(dir_data, row, out_force=True, n_files=854):
    '''
    Segments each of the datacubes according to instructions in df_grid. This is the
    high level function that accesses seg_zero_step, seg_one_step, or seg_two_step
    '''
    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, wl_bands = get_clip_type(row)
    smooth_type, window_size, order = get_smooth_type(row)
    bin_type, _, _, _ = get_bin_type(row)
    segment_type, method, _, _, _, _, _ = get_segment_type(row)

    base_dir = seg_get_base_dir(dir_data, panel_type, crop_type, clip_type, smooth_type, bin_type)
    base_dir_bm = os.path.join(dir_data, panel_type, crop_type, clip_type,
                               smooth_type)
    dir_out_mask = os.path.join(dir_data, panel_type, crop_type, clip_type,
                                smooth_type, segment_type)
    if check_processing(dir_out_mask, ext='.bip', n_files=n_files):
        return

    hsbatch = batch()
    hsbatch.io.set_io_defaults(force=out_force)
    if isinstance(method, list) and len(method) == 2: # two step
        seg_two_step(base_dir, base_dir_bm, hsbatch, row)
    elif method is not None:  # one step
        seg_one_step(base_dir, base_dir_bm, hsbatch, row)
    else:  # just create the spectra
        print('Segment: ``segment_type`` is None, so there will not be any '
              'segmentation/masking performed. However, MCARI2 band math will '
              'still be performed so data are available when tuning/training '
              'the models with auxiliary features.')
        seg_zero_step(base_dir, base_dir_bm, hsbatch, row)
        name_append = segment_type.replace('_', '-')
        hsbatch.cube_to_spectra(base_dir=base_dir, folder_name=segment_type,
                                name_append=name_append,
                                base_dir_out=base_dir_bm, write_geotiff=False)

def seg_f_pp(fname_list_seg, row, base_dir_bm, out_force, lock):
    '''
    Parallel processing: segments each of the datacubes according to
    instructions in df_grid organized for multi-core processing.
    '''
    _, method_seg, _, _, _, _, _ = get_segment_type(row)

    hsbatch = batch(lock=lock)
    hsbatch.io.set_io_defaults(force=out_force)
    if isinstance(method_seg, list) and len(method_seg) == 2: # two step
        seg_two_step(fname_list_seg, base_dir_bm, hsbatch, row)
    elif method_seg is not None:  # one step
        seg_one_step(fname_list_seg, base_dir_bm, hsbatch, row)
    else:
        seg_zero_step(fname_list_seg, base_dir_bm, hsbatch, row)

def seg_pp(dir_data, row, n_jobs, out_force=True, n_files=854):
    '''
    Actual execution of band smoothing via multi-core processing
    '''
    m = Manager()
    lock = m.Lock()

    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, _ = get_clip_type(row)
    smooth_type, _, _ = get_smooth_type(row)
    bin_type, _, _, _ = get_bin_type(row)
    segment_type, method_seg, _, _, _, _, _ = get_segment_type(row)

    base_dir = seg_get_base_dir(dir_data, panel_type, crop_type, clip_type,
                                smooth_type, bin_type)
    base_dir_bm = os.path.join(dir_data, panel_type, crop_type, clip_type,
                               smooth_type, bin_type)
    dir_out_mask = os.path.join(dir_data, panel_type, crop_type, clip_type,
                                smooth_type, bin_type, segment_type)
    dir_out_spec = os.path.join(dir_out_mask, 'reflectance')
    # dir_out_der1 = os.path.join(dir_out_mask, 'derivative_1')
    pathlib.Path(dir_out_spec).mkdir(parents=True, exist_ok=True)

    proc_mask = check_processing(dir_out_mask, ext='.bip', n_files=n_files)
    proc_spec = check_processing(dir_out_spec, ext='.spec', n_files=n_files)
    # proc_der = check_processing(dir_out_der1, ext='.spec', n_files=n_files)
    if out_force is False and proc_mask is True and proc_spec is True:
        fname_list = []
    # elif segment_type == 'seg_none':  # can't do this because we still have to run seg_zero_step
    #     fname_list = []
    else:
        fname_list = fnmatch.filter(os.listdir(base_dir), '*.bip')  # no filepath
        fname_list = [os.path.join(base_dir, f) for f in fname_list]
        # fname_list = recurs_dir(base_dir, search_ext='.bip', level=0)

    if len(fname_list) == 0:
        print('Segment: all images are already processed.\n')
    else:
        if method_seg is None:
            print('Segment: ``segment_type`` is None, so there will not be any '
                  'segmentation/masking performed. Mean spectra and derivative '
                  'spectra are being extracted...')

        chunks = chunk_by_n(fname_list, n_jobs*2)
        # print('Length of fname_list: {0}'.format(len(fname_list)))
        # print('Number of chunks: {0}'.format(len(chunks)))
        chunk_avg = sum([len(i) for i in chunks]) / len(chunks)
        # print('Average length of each chunk: {0:.1f}'.format(chunk_avg))
        # print('Number of cores: {0}\n'.format(n_jobs))
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            executor.map(
                seg_f_pp, chunks, it.repeat(row),
                it.repeat(base_dir_bm), it.repeat(out_force), it.repeat(lock))

        ext_out = '.spec'
        check_missed_files(fname_list, dir_out_spec, ext_out, seg_f_pp, row,
                           base_dir_bm, out_force, lock)

def feats_f_pp(fname_list_derivative, dir_out_derivative, name_append,
               out_force, lock):
    '''
    Parallel processing: calculates the derivative spectra for each of the
    datacubes according to instructions in df_grid organized for multi-core
    processing.
    '''
    hsbatch = batch(lock=lock)
    hsbatch.io.set_io_defaults(force=out_force)
    folder_name = None
    # name_append = os.path.split(dir_out_derivative)[-1].replace('_', '-')
    hsbatch.spectra_derivative(
        fname_list=fname_list_derivative, folder_name=None,
        name_append=name_append, base_dir_out=dir_out_derivative)

def feats_pp(dir_data, row, n_jobs, out_force=True, n_files=854):
    '''
    Actual execution of spectral derivative calculation via multi-core processing
    '''
    m = Manager()
    lock = m.Lock()

    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, wl_bands = get_clip_type(row)
    smooth_type, window_size, order = get_smooth_type(row)
    bin_type, _, _, _ = get_bin_type(row)
    segment_type, method_seg, _, _, _, _, _ = get_segment_type(row)
    feature_type = row['features']

    if feature_type == 'reflectance' or feature_type is None:  # we don't have to do anything since this was already done in segmentatino step
        print('Features: ``feature_type`` is either None or "reflectance" and '
              'there is nothing to process.')
        return

    # base_dir_feat = seg_get_base_dir(dir_data, panel_type, crop_type,
    #                                  clip_type, smooth_type, bin_type)
    base_dir = os.path.join(dir_data, panel_type, crop_type, clip_type,
                            smooth_type, bin_type, segment_type)
    dir_out_derivative = os.path.join(base_dir, feature_type)
    name_append = segment_type.replace('_', '-')  #  want to keep this to keep file names unique
    already_processed = check_processing(dir_out_derivative, ext='.spec', n_files=n_files)
    if out_force is False and already_processed is True:
        fname_list = []
    else:
        fname_list = recurs_dir(os.path.join(base_dir, 'reflectance'), search_ext='.spec', level=0)

    chunk_size = int(len(fname_list) / (n_jobs*2))
    np.random.shuffle(fname_list)  # studies have different size images, so shuffling makes each chunk more similar
    chunks = chunk_by_n(fname_list, n_jobs*2)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        executor.map(
            feats_f_pp, chunks, it.repeat(dir_out_derivative),
            it.repeat(name_append), it.repeat(out_force), it.repeat(lock))

# In[Training initialization functions]
def load_ground_data(dir_data, y_label='biomass_kgha'):
    '''
    Loads the ground data for supervised regression. This should be saved to the
    MSI data directory
    '''
    fname_wells_18 = os.path.join(dir_data, 'wells_ground_data_2018.csv')
    fname_wells_19 = os.path.join(dir_data, 'wells_ground_data_2019.csv')
    fname_aerf_19 = os.path.join(dir_data, 'aerf_ground_data.csv')
    df1 = pd.read_csv(fname_wells_18)
    df2 = pd.read_csv(fname_wells_19)
    df3 = pd.read_csv(fname_aerf_19)
    col = ['study', 'date_image', 'plot_id', 'trt', 'rate_n_pp_kgha',
           'rate_n_sd_plan_kgha', 'rate_n_total_kgha', 'growth_stage',
           y_label]
           # 'tissue_n_pct', 'biomass_kgha', 'nup_kgha']
    df_wells1 = df1[pd.notnull(df1[y_label])][col].reset_index(drop=True)
    df_wells2 = df2[pd.notnull(df2[y_label])][col].reset_index(drop=True)
    df_aerf = df3[pd.notnull(df3[y_label])][col].reset_index(drop=True)
    df_ground = df_wells1.append(df_wells2).reset_index(drop=True)
    df_ground = df_ground.append(df_aerf).reset_index(drop=True)
    df_ground.insert(1, 'date', None)
    df_ground['date'] = df_ground['date_image'].apply(lambda value:value.replace('-',''))
    del df_ground['date_image']
    return df_ground

def get_spec_data(dir_data, row, feat='reflectance'):
    '''
    Searches for hyperspectral .spec files; from here, can be loaed in for
    supervised regression or simply counted to be sure all expected files were
    processed

    Parameters:
        feat (``str``): must be either "reflectance" or "derivative",
        indicating which directory path to return.
    '''
    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, _ = get_clip_type(row)
    smooth_type, _, _ = get_smooth_type(row)
    bin_type, _, _, _ = get_bin_type(row)
    segment_type, _, _, _, _, _, _ = get_segment_type(row)
    base_dir_spec = os.path.join(dir_data, panel_type, crop_type, clip_type,
                                 smooth_type, bin_type, segment_type, feat)
    return base_dir_spec

def load_spec_data(dir_data, row, feat='reflectance'):
    '''
    Loads the hyperspectral image data for supervised regression

    Must have the following meta columns: study, date, plot_id

    feat must be one of "reflectance" or "derivative".
    '''
    base_dir_spec = get_spec_data(dir_data, row, feat)
    hsbatch = batch()
    df_spec = hsbatch.spectra_to_df(
        base_dir=base_dir_spec, search_ext='spec', dir_level=0)
        # multithread=multithread)
    df_spec = insert_date_study(df_spec)
    meta_bands = hsbatch.io.tools.meta_bands  # dict to map band to wavelength
    bands = list(meta_bands.keys())
    return df_spec, meta_bands, bands

def load_preseg_stats(dir_data, row, bm_folder_name='bm_mcari2'):
    '''
    For every "group" of processed images (all segmentation options), gather
    pre-segmentation statistics (e.g., MCARI2 10th percentile values).

    These pre-segmentation statistics can be included as additional features to
    the spectral dataset and model training/validation will be performed both
    with and without these features
    '''
    panel_type = row['dir_panels']
    crop_type = row['crop']
    clip_type, _ = get_clip_type(row)
    smooth_type, _, _ = get_smooth_type(row)
    bin_type, _, _, _ = get_bin_type(row)
    segment_type, _, _, _, _, _, _ = get_segment_type(row)
    base_dir_bm = os.path.join(dir_data, panel_type, crop_type, clip_type,
                               smooth_type, bin_type, bm_folder_name)
    stats_csv = bm_folder_name.replace('_', '-') + '-stats.csv'
    df_bm_stats = pd.read_csv(os.path.join(base_dir_bm, stats_csv))
    df_bm_stats = insert_date_study(df_bm_stats)
    return df_bm_stats

def insert_date_study(df):
    '''
    Takes <df> and parses the 'fname' column to extract "study" and "date"
    info, then adds this as its own column before returning <df>.

    This is useful because we load the spectral data (via .spec files), which
    are essentially numpy arrays. Thus, we have to get study, date, and plot
    information via the filename to be able to insert into the dataframe to
    be able to join to the ground truth data.
    '''
    df.insert(1, 'date', None)
    df.insert(1, 'study', None)
    df_temp = df.copy()
    for idx, spec in df_temp.iterrows():
        fname = spec['fname']
        study_str = 'study_'
        date_str = 'date_'
        plot_str = '_plot_'
        study = fname[fname.find(study_str)+len(study_str):fname.find(date_str)-1]
        date = fname[fname.find(date_str)+len(date_str):fname.find(plot_str)]
        df.loc[idx, 'study'] = study
        df.loc[idx, 'date'] = date
    return df

def join_ground_bm_spec(df_ground, df_bm_stats, df_spec, on=['study', 'date', 'plot_id']):
    '''
    Joins data so it is available in a single dataframe. Before joining, each
    unique dataset will be given an "dataset_id" to help with stratified
    sampling later on
    '''
    df_ground_copy = df_ground.copy()
    df_ground_copy.insert(0, 'dataset_id', None)
    df_ground_copy['dataset_id'] = df_ground_copy.groupby(['study','date']).ngroup()
    df_join = df_ground_copy.merge(df_bm_stats, on=on)
    df_join = df_join.merge(df_spec, on=on)
    return df_join

def join_ground_spec(df_ground, df_spec, on=['study', 'date', 'plot_id']):
    '''
    Joins data so it is available in a single dataframe. Before joining, each
    unique dataset will be given an "dataset_id" to help with stratified
    sampling later on
    '''
    df_ground_copy = df_ground.copy()
    df_ground_copy.insert(0, 'dataset_id', None)
    df_ground_copy['dataset_id'] = df_ground_copy.groupby(['study','date']).ngroup()
    df_join = df_ground_copy.merge(df_spec, on=on)
    return df_join

def save_joined_df(dir_results, df_join, msi_run_id, grid_idx, y_label):
    '''
    Saves the joined dataframe to a new folder in ``dir_results`` with another
    README.txt file that provides some basic details about the processing that
    was performed in case the grid_idx gets messed up..

    grid_idx is the index of df_grid (this will change if df_grid changes,
        so that is why ``msi_run_id`` is also included in the folder name)
    '''
    folder_name = 'msi_' + str(msi_run_id) + '_' + str(grid_idx).zfill(3)
    dir_out = os.path.join(dir_results, folder_name, y_label)
    pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
    str1 = folder_name + '_'
    str2 = y_label
    fname_out = os.path.join(dir_out, str1 + str2 + '_data.csv')
    df_join.to_csv(fname_out, index=False)

def create_readme(dir_results, msi_run_id, row):
    '''
    Creates a README file that includes information about this run scenario (to
    be saved alongside the results files)
    '''
    folder_name = 'msi_' + str(msi_run_id) + '_' + str(row.name).zfill(3)
    dir_out = os.path.join(dir_results, folder_name)
    pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(dir_out, folder_name + '_README.txt'), 'w+') as f:
        f.write('MSI run ID: {0}\n'.format(msi_run_id))
        f.write('Processing scenario ID: {0}\n\n'.format(row.name))
        f.write('Panels type: {0}\n'.format(row['dir_panels']))
        f.write('Crop type: {0}\n'.format(row['crop']))
        f.write('Clip type: {0}\n'.format(row['clip']))
        f.write('Smooth type: {0}\n'.format(row['smooth']))
        f.write('Bin type: {0}\n'.format(row['bin']))
        f.write('Segment type: {0}\n\n'.format(row['segment']))

def write_to_readme(msg, dir_results, msi_run_id, row):
    '''
    Writes ``msg`` to the README.txt file
    '''
    folder_name = 'msi_' + str(msi_run_id) + '_' + str(row.name).zfill(3)
    dir_out = os.path.join(dir_results, folder_name)
    with open(os.path.join(dir_out, folder_name + '_README.txt'), 'a') as f:
        f.write(str(msg) + '\n')

def get_random_seed(dir_results, msi_run_id, row, seed=None):
    '''
    Assign the random seed
    '''
    if seed is None:
        seed = np.random.randint(0, 1e6)
    else:
        seed = int(seed)
    write_to_readme('Random seed: {0}'.format(seed), dir_results, msi_run_id, row)
    return seed

# In[Training utility functions]
def split_train_test(df, test_size=0.4, random_seed=None, stratify=None):
    '''
    Splits ``df`` into train and test sets based on proportion indicated by
    ``test_size``
    '''
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=stratify)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_train, df_test

def get_repeated_stratified_kfold(df, n_splits=3, n_repeats=2,
                                  random_state=None):
    '''
    Stratifies ``df`` by "dataset_id", and creates a repeated, stratified
    k-fold cross-validation object that can be used for any sk-learn model
    '''
    X_null = np.zeros(len(df))  # not necessary for StratifiedKFold
    y_train_strat = df['dataset_id'].values
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=random_state)
    cv_rep_strat = rskf.split(X_null, y_train_strat)
    return cv_rep_strat

def check_stratified_proportions(df, cv_rep_strat):
    '''
    Checks the proportions of the stratifications in the dataset and prints
    the number of observations in each stratified group
    '''
    cols_meta = ['dataset_id', 'study', 'date', 'plot_id', 'trt', 'growth_stage']
    # X_meta_train = df_train[cols_meta].values
    # X_meta_test = df_test[cols_meta].values
    X_meta = df[cols_meta].values
    print('Number of observations in each cross-validation dataset (key=ID; value=n):')
    train_list = []
    val_list = []
    for train_index, val_index in cv_rep_strat:
        X_meta_train_fold = X_meta[train_index]
        X_meta_val_fold = X_meta[val_index]
        X_train_dataset_id = X_meta_train_fold[:,0]
        train = {}
        val = {}
        for uid in np.unique(X_train_dataset_id):
            n1 = len(np.where(X_meta_train_fold[:,0] == uid)[0])
            n2 = len(np.where(X_meta_val_fold[:,0] == uid)[0])
            train[uid] = n1
            val[uid] = n2
        train_list.append(train)
        val_list.append(val)
    print('Train set:')
    for item in train_list:
        print(item)
    print('Test set:')
    for item in val_list:
        print(item)

def impute_missing_data(X, random_seed, method='iterative'):
    '''
    Imputes missing data in X - sk-learn models will not work with null data

    method should be one of "iterative" (takes more time) or "knn"
    '''
    if np.isnan(X).any() is False:
        return X

    if method == 'iterative':
        imp = IterativeImputer(max_iter=10, random_state=random_seed)
    elif method == 'knn':
        imp = KNNImputer(n_neighbors=2, weights='uniform')
    X_out = imp.fit_transform(X)
    return X_out

def numeric_df_cols(df):
    '''
    Changes all numeric dataframe column headings to integer if they are
    strings. Caution, float-like strings will also be changed to integers.

    Useful becasue we want to access df columns by band number to make for
    convenient construction of the sk-learn X feature matrix
    '''
    df_out = df.copy()
    for c in df.columns:
        if isinstance(c, str) and c.isnumeric():
            df_out.rename(columns = {c: int(c)}, inplace=True)
    return df_out

def get_X_and_y(df, bands, y_label, random_seed, key_or_val='keys', extra=None):
    '''
    Gets the X and y from df; y is determined by the ``y_label`` column

    ``key_or_val`` should be either "keys" (band info in keys of ``bands``) or
    "values" (band info in values of ``bands``)

    ``extra`` can be a string or a list of strings, but they should be column
    names in ``df`` (e.g., "pctl_10th")
    '''
    if isinstance(bands, dict):
        if key_or_val == 'keys':
            bands = sorted(list(bands.keys()))
        elif key_or_val == 'values':
            bands = sorted(list(bands.values()))
    if extra is None:
        extra = [None]
    if extra != [None]:
        if not isinstance(extra, list):
            extra = [extra]
        for col in extra:
            bands.append(col)
    X = df[bands].values
    X.shape
    y = df[y_label].values
    X = impute_missing_data(X, random_seed, method='iterative')
    return X, y

# In[Hyperparameter tuning functions]
def param_grid_add_key(param_grid_dict, key='regressor__'):
    '''
    Define tuning parameter grids for pipeline or transformed regressor

    key should either be "transformedtargetregressor__regressor__" for pipe key
    or "regressor__" for transformer key
    '''
    param_grid_mod = param_grid_dict.copy()
    for model in param_grid_dict:
        param_grid_mod[model] = {f'{key}{k}': v for k, v in param_grid_mod[model].items()}
    return param_grid_mod

@ignore_warnings(category=ConvergenceWarning)
def feat_selection_lasso(X, y, alpha, max_iter, random_seed):
    '''
    Feature selection via lasso algorithm
    '''
    model_las = Lasso(
        alpha=alpha, max_iter=max_iter, selection='cyclic',
        random_state=random_seed).fit(X, y)
    model_bs = SelectFromModel(model_las, prefit=True)
    feats = model_bs.get_support(indices=True)
    coefs = model_las.coef_[feats]  # get ranking and save that too
    feat_ranking = rankdata(-coefs, method='min')
    X_select = model_bs.transform(X)
    return X_select, feats, feat_ranking

def f_lasso_feat_n(x, X1, y1, max_iter, random_seed):
    '''
    Uses the Lasso model to determine the number of features selected with a
    given x value (x represents the alpha hyperparameter of the Lasso model)
    '''
    if not isinstance(x, list):
        x = [x]
    feat_n_sel = []
    for alpha in x:
        model_las = Lasso(
            alpha=alpha, max_iter=max_iter, selection='cyclic',
            random_state=random_seed).fit(X1, y1)
        model_bs = SelectFromModel(model_las, prefit=True)
        feats = model_bs.get_support(indices=True)
        feat_n_sel.append(len(feats))  # subtracting 1 to account for 0 feats (which will be -1 and isn't the optimal) - we need at leaast 1 feature
    if len(feat_n_sel) == 1:
        return feat_n_sel[0]
    else:
        return feat_n_sel

def f_lasso_feat_max(x, n_feats, X1, y1, max_iter, random_seed):
    feat_n_sel = f_lasso_feat_n(x, X1, y1, max_iter, random_seed)
    print(feat_n_sel)
    der = n_feats - feat_n_sel  # difference between n_feats and selected (we want to be 0 for connvergence)
    print(der)
    return der

def setup_warning_catcher():
    """ Wrap warnings.showwarning with code that records warnings. """
    caught_warnings = []
    original_showwarning = warnings.showwarning
    def custom_showwarning(*args,  **kwargs):
        caught_warnings.append(args[0])
        return original_showwarning(*args, **kwargs)
    warnings.showwarning = custom_showwarning
    return caught_warnings

def find_alpha_max(X1, y1, max_iter, random_seed):
    '''
    Finds the max alpha value for Lasso feature selection, which can be passed
    to Lasso to achieve a single (one) feature.

    This contrasts the minimum alpha that will achieve many features (up to
    240 for Pika II hyperspectral data)
    '''
    feat_n_sel = f_lasso_feat_n(1, X1, y1, max_iter, random_seed)
    x = 0
    if feat_n_sel <= 1:
        feat_n_sel = 2
        while feat_n_sel > 1:
            x += 0.01
            feat_n_sel = f_lasso_feat_n(x, X1, y1, max_iter, random_seed)
    else:
        while feat_n_sel > 1:
            x += 1
            feat_n_sel = f_lasso_feat_n(x, X1, y1, max_iter, random_seed)
            # print('Iteration: {0}\nFeature n: {1}\n'.format(x, feat_n_sel))
    return x

def gradient_descent_step_pct_alpha_min(
        feat_n_sel, feat_n_last, n_feats, step_pct):
    '''
    Adjusts step_pct dynamically based on progress of reaching n_feats

    Ideally, step_pct should be large if we're a long way from n_feats, and
    much smaller if we're close to n_feats
    '''
    # find relative distance in a single step
    n_feats_closer = feat_n_sel - feat_n_last
    pct_closer = n_feats_closer/n_feats
    pct_left = (n_feats-feat_n_sel)/n_feats
    # print(pct_closer)
    # print(pct_left)
    if pct_closer < 0.08 and pct_left > 0.5 and step_pct * 10 < 1:  # if we've gotten less than 8% the way there
        print('Old "step_pct": {0}'.format(step_pct))
        step_pct *= 10
        print('New "step_pct": {0}'.format(step_pct))
    elif pct_closer < 0.15 and pct_left > 0.4 and step_pct * 5 < 1:  # if we've gotten less than 8% the way there
        print('Old "step_pct": {0}'.format(step_pct))
        step_pct *= 5
        print('New "step_pct": {0}'.format(step_pct))
    elif pct_closer < 0.3 and pct_left > 0.3 and step_pct * 2 < 1:  # if we've gotten less than 8% the way there
        print('Old "step_pct": {0}'.format(step_pct))
        step_pct *= 2
        print('New "step_pct": {0}'.format(step_pct))

    elif pct_closer > 0.1 and pct_left < pct_closer*1.3:  # if % gain is 77% of what is left, slow down a bit
        print('Old "step_pct": {0}'.format(step_pct))
        step_pct /= 5
        print('New "step_pct": {0}'.format(step_pct))
    elif pct_closer > 0.05 and pct_left < pct_closer*1.3:  # if % gain is 77% of what is left, slow down a bit
        print('Old "step_pct": {0}'.format(step_pct))
        step_pct /= 2
        print('New "step_pct": {0}'.format(step_pct))
    else:  # keep step_pct the same
        pass
    return step_pct

# @ignore_warnings(category=ConvergenceWarning)
def find_alpha_min(X1, y1, max_iter, random_seed, n_feats, alpha_init=1e-3,
                   step_pct=0.01, method='full',
                   exit_on_stagnant_n=5):
    '''
    Finds the min alpha value for Lasso feature selection, which can be passed
    to Lasso to achieve many features.

    method:
        options:
            "convergence_warning": proceeds normally until a ConvergenceWarning
                is reached, then just stops there (using this method is making
                the decision that we will stop short of testing the full
                feature set, and will only try up to the number of features
                that first produces a ConvergenceWarning). This option makes
                features selection much faster, but does not look at all
                features.
            "full": proceeds until all features are represented (feature
                selection can be slow, but will look at all features).
    step_pct (``float``): indicates the percentage to adjust alpha by on each
        iteration.
    exit_on_stagnant_n (``int``): Will stop searching for minimum alpha value
        if number of selected features do not change after this many
        iterations.
    '''
    msg = ('Leaving while loop before finding the alpha value that achieves '
           'selection of {0} feature(s) ({1} alpha value to use).\n')
    feat_n_sel = n_feats+1  # initialize to enter the while loop
    while feat_n_sel > n_feats:
        feat_n_sel = f_lasso_feat_n(alpha_init, X1, y1, max_iter, random_seed)
        alpha_init *= 10  # adjust alpha_init until we get a reasonable number of features
    x = alpha_init

    if method == 'convergence_warning':
        same_n = 0
        caught_warnings_list = setup_warning_catcher()
        # will loop until warning or until all features are selected
        while len(caught_warnings_list) < 1:
            feat_n_last = feat_n_sel
            feat_n_sel = f_lasso_feat_n(x, X1, y1, max_iter, random_seed)
            same_n += 1 if feat_n_last == feat_n_sel else -same_n
            if same_n > exit_on_stagnant_n:
                print(msg.format(n_feats, 'minimum'))
                break
            if feat_n_sel < n_feats:
                step_pct = gradient_descent_step_pct_alpha_min(
                    feat_n_sel, feat_n_last, n_feats, step_pct)
                x *= (1-step_pct)
            else:
                x *= 1+(step_pct/2)

            if feat_n_sel >= n_feats:  # as soon as selected features meets or exceeds possible features, we're done
                break
            print('alpha: {0}'.format(x))
            print('Features selected: {0}\n'.format(feat_n_sel))
    elif method == 'full':
        same_n = 0
        while feat_n_sel != n_feats:
            feat_n_last = feat_n_sel
            # print(x)
            feat_n_sel = f_lasso_feat_n(x, X1, y1, max_iter, random_seed)
            same_n += 1 if feat_n_last == feat_n_sel else -same_n
            if same_n > exit_on_stagnant_n:
                print(msg.format(n_feats, 'minimum'))
                break
            if feat_n_sel < n_feats:
                step_pct = gradient_descent_step_pct_alpha_min(
                    feat_n_sel, feat_n_last, n_feats, step_pct)
                x *= (1-step_pct)
            else:  # we went over; go back to prvious step, make much smaller, and adjust alpha down a bit
                x /= (1-step_pct)
                step_pct /= 10
                x *= (1-step_pct)
            print('alpha: {0}'.format(x))
            print('Features selected: {0}'.format(feat_n_sel))
            print('Iterations without progress: {0}\n'.format(same_n))
    print('Using up to {0} selected features\n'.format(feat_n_sel))
    return x, step_pct

def alpha_min_parallel_1(global_dict, lock, X1, y1, max_iter, random_seed, n_feats, exit_on_stagnant_n):
    with lock:  # makes sure other workers don't change
        if global_dict['same_n'] > exit_on_stagnant_n or global_dict['feat_n_sel'] == n_feats:
            # print(msg.format(n_feats, 'minimum'))
            return

        global_dict['feat_n_last'] = global_dict['feat_n_sel']

        # perhaps we have to have a global_dict_master and global_dict for each worker
        # as is, I don't think f_lasso_feat_n() will be executed in parallel..?
        global_dict['feat_n_sel'] = f_lasso_feat_n(global_dict['alpha'], X1, y1, max_iter, random_seed)
        global_dict['same_n'] += 1 if global_dict['feat_n_last'] == global_dict['feat_n_sel'] else -global_dict['same_n']

        if global_dict['feat_n_sel'] < n_feats:
            global_dict['step_pct'] = gradient_descent_step_pct_alpha_min(
                global_dict['feat_n_sel'], global_dict['feat_n_last'],
                n_feats, global_dict['step_pct'])
            global_dict['alpha'] *= (1-global_dict['step_pct'])
        else:  # we went over; go back to prvious step, make much smaller, and adjust alpha down a bit
            global_dict['alpha'] /= (1-global_dict['step_pct'])
            global_dict['step_pct'] /= 10
            global_dict['alpha'] *= (1-global_dict['step_pct'])
        print('alpha: {0}'.format(global_dict['alpha']))
        print('Features selected: {0}'.format(global_dict['feat_n_sel']))
        print('Iterations without progress: {0}\n'.format(global_dict['same_n']))

def alpha_min_parallel(global_dict, w_dict, lock, X1, y1, max_iter,
                       random_seed, n_feats, exit_on_stagnant_n):
    # with lock:  # makes sure other workers don't change
    if global_dict['same_n'] > exit_on_stagnant_n or global_dict['feat_n_sel'] == n_feats:
        # print(msg.format(n_feats, 'minimum'))
        return w_dict

    w_dict['feat_n_last'] = w_dict['feat_n_sel']

    # perhaps we have to have a w_dict_master and w_dict for each worker
    # as is, I don't think f_lasso_feat_n() will be executed in parallel..?
    w_dict['feat_n_sel'] = f_lasso_feat_n(w_dict['alpha'], X1, y1, max_iter, random_seed)
    if global_dict['feat_n_sel'] == n_feats:  # another worker found alpha
        return w_dict
    if w_dict['feat_n_sel'] == n_feats:
        with lock:
            global_dict['step_pct'] = w_dict['step_pct']
            global_dict['alpha'] = w_dict['alpha']
            global_dict['feat_n_sel'] = w_dict['feat_n_sel']
        return w_dict

    w_dict['same_n'] += 1 if w_dict['feat_n_last'] == w_dict['feat_n_sel'] else -w_dict['same_n']

    if w_dict['feat_n_sel'] < n_feats:
        w_dict['step_pct'] = gradient_descent_step_pct_alpha_min(
            w_dict['feat_n_sel'], w_dict['feat_n_last'],
            n_feats, w_dict['step_pct'])
        w_dict['alpha'] *= (1-w_dict['step_pct'])
    else:  # we went over; go back to prvious step, make much smaller, and adjust alpha down a bit
        w_dict['alpha'] /= (1-w_dict['step_pct'])
        w_dict['step_pct'] /= 10
        w_dict['alpha'] *= (1-w_dict['step_pct'])
    print('alpha: {0}'.format(w_dict['alpha']))
    print('Features selected: {0}'.format(w_dict['feat_n_sel']))
    print('Iterations without progress: {0}\n'.format(w_dict['same_n']))

    if global_dict['feat_n_sel'] == n_feats:  # another worker found alpha
        return w_dict
    else:
        return w_dict
    # with lock:  # just to update global_dict
    #     global_dict['last_to_report'] = w_dict['worker_n']
    #     global_dict['same_n'] += 1 if w_dict['same_n'] > 0 and global_dict['feat_n_last'] == w_dict['feat_n_sel'] else -global_dict['same_n']
    #     if w_dict['feat_n_sel'] < n_feats:
    #         global_dict['step_pct'] = w_dict['step_pct']
    #         global_dict['alpha'] = w_dict['alpha']


def find_alpha_min_pp(X1, y1, max_iter, random_seed, n_feats, n_jobs, alpha_init=1e-3,
                      step_pct=0.01, exit_on_stagnant_n=5):
    msg = ('Leaving while loop before finding the alpha value that achieves '
           'selection of {0} feature(s) ({1} alpha value to use).\n')
    feat_n_sel = n_feats+1  # initialize to enter the while loop
    while feat_n_sel > n_feats:
        feat_n_sel = f_lasso_feat_n(alpha_init, X1, y1, max_iter, random_seed)
        alpha_init *= 10  # adjust alpha_init until we get a reasonable number of features
    N_WORKERS = os.cpu_count()
    with Manager() as manager:
        lock = manager.Lock()
        global_dict = manager.dict(alpha=alpha_init, feat_n_last=feat_n_sel,
                                   feat_n_sel=feat_n_sel, step_pct=step_pct,
                                   same_n=0, fresh=True, last_to_report=0)
        while global_dict['feat_n_sel'] != n_feats:
            # Sends each worker/CPU the function to execute
            # worker_dicts = []
            # for i in range(N_WORKERS):
            #     w_dict = {  # make a copy
            #         'worker_n'=i+1,
            #         'alpha'=global_dict['alpha'],
            #         'feat_n_last'=global_dict['feat_n_last'],
            #         'feat_n_sel'=global_dict['feat_n_sel'],
            #         'step_pct'=global_dict['step_pct'],
            #         'same_n'=global_dict['same_n']}
            #     w_dict['step_pct'] = gradient_descent_step_pct_alpha_min(
            #         w_dict['feat_n_sel'], w_dict['feat_n_last'],
            #         n_feats, w_dict['step_pct'])
            #     w_dict['alpha'] *= (1-w_dict['step_pct'])
            #     global_dict['step_pct'] = w_dict['step_pct']
            #     global_dict['alpha'] = w_dict['alpha']
            #     worker_dicts = worker_dicts.append(worker_d)
            # pool = [Process(target=alpha_min_parallel, args=(global_dict, lock, X1, y1, max_iter, random_seed, n_feats, exit_on_stagnant_n))
            #         for _ in range(N_WORKERS)]
            pool = [Process(target=alpha_min_parallel, args=(global_dict, w_dict, lock, X1, y1, max_iter, random_seed, n_feats, exit_on_stagnant_n))
                    for w_dict in worker_dicts]
            for p in pool:
                p.start()
            for p in pool:
                p.join()
                # p.close()
            if global_dict['same_n'] > exit_on_stagnant_n:
                print(msg.format(n_feats, 'minimum'))
                break
        for p in pool:
            alpha = global_dict['alpha']
            step_pct = global_dict['step_pct']
            feat_n_sel = global_dict['feat_n_sel']
            p.terminate()
            p.close()
            del p
        print('Alpha: {0}\nNumber of features:{1}'.format(global_dict['alpha'], global_dict['feat_n_sel']))
    print('Using up to {0} selected features\n'.format(feat_n_sel))
    return alpha, step_pct

def build_feat_selection_df(
        X1, y1, max_iter, random_seed, n_feats=None, n_linspace=200,
        method_alpha_min='full', alpha_init=1e-3, step_pct=0.01):
    '''
    Builds a dynamic list of all alpha values to try for features selection,
    with the goal of covering the full range of features from 1 to all
    features (max number of features is variable depending on dataset)

    The ``start`` variable in this function probably need to be adjusted, and
    really we have to come up with a better way to dynamically choose the start
    number..

    method_alpha_min:
        options:
            "convergence_warning": proceeds normally until a ConvergenceWarning
                is reached, then just stops there (using this method is making
                the decision that we will stop short of testing the full
                feature set, and will only try up to the number of features
                that first produces a ConvergenceWarning). This option makes
                features selection much faster, but does not look at all
                features.
            "full": proceeds until all features are represented (feature
                selection can be slow, but will look at all features).
    '''
    msg = ('``method_alpha_min`` must be either "full" or'
           '"convergence_warning"')
    assert method_alpha_min in ["full", "convergence_warning"], msg
    if n_feats is None:
        n_feats = X1.shape[1]
    else:
        n_feats = int(n_feats)
    if n_feats > X1.shape[1]:
        n_feats = X1.shape[1]
    alpha_min, step_pct = find_alpha_min(
        X1, y1, max_iter, random_seed, n_feats, alpha_init=alpha_init,
        step_pct=step_pct, method=method_alpha_min, exit_on_stagnant_n=5)
    # alpha_min, step_pct = find_alpha_min_pp(
    #     X1, y1, max_iter, random_seed, n_feats, alpha_init=alpha_init,
    #     step_pct=step_pct, exit_on_stagnant_n=5)

    start = np.log(alpha_min)
    alpha_max = find_alpha_max(X1, y1, max_iter, random_seed)  # pretty easy to find
    stop = np.log(alpha_max)
    logspace_list = list(np.logspace(start, stop, num=n_linspace, base=np.e))
    return logspace_list, alpha_min, step_pct

def filter_logspace_list(logspace_list, X1, y1, max_iter, random_seed):
    df = None
    for alpha in logspace_list:
        _, feats, feat_ranking = feat_selection_lasso(
            X1, y1, alpha, max_iter, random_seed)
        df_temp = pd.DataFrame(data=[[alpha, len(feats), tuple(feats), tuple(feat_ranking)]],
                               columns=['alpha', 'n_feats', 'feats', 'feat_ranking'])
        if df is None:
            df = df_temp.copy()
        else:
            df = df.append(df_temp)
    return df
    # df = df.drop_duplicates(subset=['feats'], ignore_index=True)
    # logspace_list_filtered = list(df['alpha'])
    # return logspace_list_filtered

def filter_logspace_list_pp(logspace_list, X1, y1, max_iter, random_seed, n_jobs):
    # m = Manager()
    # lock = m.Lock()
    chunks = chunk_by_n(logspace_list, n_jobs*2)
    if len(logspace_list) < n_jobs * 2:
        chunks = chunk_by_n(logspace_list, n_jobs)
    # print('Length of logspace_list: {0}'.format(len(logspace_list)))
    # print('Number of chunks: {0}'.format(len(chunks)))
    chunk_avg = sum([len(i) for i in chunks]) / len(chunks)
    # print('Average length of each chunk: {0:.1f}'.format(chunk_avg))
    # print('Number of cores: {0}\n'.format(n_jobs))
    df_all = None
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for df_feats in executor.map(
                filter_logspace_list, chunks, it.repeat(X1), it.repeat(y1),
                it.repeat(max_iter), it.repeat(random_seed)):
            # with lock:
            if df_all is None:
                df_all = df_feats.copy()
            else:
                df_all = df_all.append(df_feats)
    df_all = df_all.drop_duplicates(subset=['feats'], ignore_index=True)
    logspace_list_filtered = list(reversed(sorted(df_all['alpha'])))
    return logspace_list_filtered

@ignore_warnings(category=ConvergenceWarning)
def model_tuning(model, param_grid, standardize, scoring, refit,
                 X_select, y, cv_rep_strat, n_jobs=1):
    '''
    X_select represents the X matrix of only the selected features from lasso
    feature selection process

    On MSI, n_jobs should be a positive integer, otherwise CPUs will probably
    be incorrectly allocated. Because we're parrallelizing higher in the loop,
    we can just put this at 1 core (after all, we have to run GridSearchCV many
    times).
    '''
#    model_svr = SVR(tol=1e-2)
    transformer = TransformedTargetRegressor(
        regressor=model, transformer=PowerTransformer(
            'yeo-johnson', standardize=standardize))
    clf = GridSearchCV(transformer, param_grid, n_jobs=n_jobs, cv=cv_rep_strat,
                       return_train_score=True, scoring=scoring, refit=refit)
    # with ignore_warnings(category=ConvergenceWarning):
    clf.fit(X_select, y)
    df_tune = pd.DataFrame(clf.cv_results_)
    return df_tune, transformer

def get_tuning_scores_multiple(feats, feat_ranking, alpha, df_tune, cols,
                               scoring1='neg_mean_absolute_error',
                               scoring2='neg_mean_squared_error',
                               scoring3=None):
    '''
    Retrieves training and validation scores to be inserted into the results
    dataframe.
    '''
    rank_score = 'rank_test_' + scoring1
    scoring_train = 'mean_train_' + scoring1
    scoring_test = 'mean_test_' + scoring1
    std_train = 'std_train_' + scoring1
    std_test = 'std_test_' + scoring1
    scoring2_train = 'mean_train_' + scoring2
    scoring2_test = 'mean_test_' + scoring2
    std2_train = 'std_train_' + scoring2
    std2_test = 'std_test_' + scoring2
    if scoring3 is not None:
        scoring3_train = 'mean_train_' + scoring3
        scoring3_test = 'mean_test_' + scoring3
        std3_train = 'std_train_' + scoring3
        std3_test = 'std_test_' + scoring3
    score_train = df_tune[df_tune[rank_score] == 1][scoring_train].values[0]
    std_train = df_tune[df_tune[rank_score] == 1][std_train].values[0]
    score_val = df_tune[df_tune[rank_score] == 1][scoring_test].values[0]
    std_val = df_tune[df_tune[rank_score] == 1][std_test].values[0]
    score2_train = df_tune[df_tune[rank_score] == 1][scoring2_train].values[0]
    std2_train = df_tune[df_tune[rank_score] == 1][std2_train].values[0]
    score2_val = df_tune[df_tune[rank_score] == 1][scoring2_test].values[0]
    std2_val = df_tune[df_tune[rank_score] == 1][std2_test].values[0]
    if scoring3 is not None:
        score3_train = df_tune[df_tune[rank_score] == 1][scoring3_train].values[0]
        std3_train = df_tune[df_tune[rank_score] == 1][std3_train].values[0]
        score3_val = df_tune[df_tune[rank_score] == 1][scoring3_test].values[0]
        std3_val = df_tune[df_tune[rank_score] == 1][std3_test].values[0]

    params = df_tune[df_tune[rank_score] == 1]['params'].values[0]
    if scoring3 is None:
        data = [len(feats), alpha, score_train,
                std_train, score_val, std_val, score2_train, std2_train,
                score2_val, std2_val, params, feats, feat_ranking]
    else:
        data = [len(feats), alpha, score_train,
                std_train, score_val, std_val, score2_train, std2_train,
                score2_val, std2_val, score3_train, std3_train, score3_val,
                std3_val, params, feats, feat_ranking]
    df_temp = pd.DataFrame(data=[data], columns=cols)
    # for key in df_temp['tune_params'].values[0].keys():
    #     print('{0}: {1}'.format(key, df_temp['tune_params'].values[0][key]))
    return df_temp

def print_model(model):
    '''Simply prints the model type of ``model``'''
    if isinstance(model, Lasso):
        print('Lasso:')
    if isinstance(model, SVR):
        print('Support vector regression:')
    if isinstance(model, RandomForestRegressor):
        print('Random forest:')
    if isinstance(model, PLSRegression):
        print('Partial least squares regression:')

def get_lasso_feats(df_temp, key, transformer_lasso, X_select, y):
    '''
    Because Lasso has it's own feature selection built right into the
    aglorithm, it may be desireable to see the actual features used by the
    prediction model. This function gets the features chosen from feature
    selection and adds them as two additional columns to the dataframe.
    '''
    las2_alpha = df_temp['tune_params'].values[0][f'{key}alpha']  # to figure out which bands were actually used
    transformer_lasso.set_params(**{f'{key}alpha': las2_alpha})
    key_c = f'{key}'[:-2]
    transformer_lasso.get_params()[key_c].fit(X_select, y)
    model_bs2 = SelectFromModel(transformer_lasso.get_params()[key_c], prefit=True)
    feats_used = model_bs2.get_support(indices=True)
    df_temp['feat_n_used'] = len(feats_used)
    df_temp['features_used'] = [feats_used]
    return df_temp

def tune_model(X_select, y, model, param_grid,
               standardize, scoring, scoring_refit, key, cv_rep_strat,
               feats, feat_ranking, alpha, cols):
    '''
    Tunes a single model and appends results as a new row to df_full
    '''
    # cols_e = cols.extend(['feat_n_used', 'features_used'])
    data = [len(feats), alpha]
    data[2:] = [np.nan] * 15
    if (len(feats) == 0) or (isinstance(model, PLSRegression) and len(feats) <= 1):
        df_temp = pd.DataFrame(data=[data], columns=cols)
    else:
        df_tune, transformer = model_tuning(
            model, param_grid, standardize, scoring,
            scoring_refit, X_select, y, cv_rep_strat)
        df_temp = get_tuning_scores_multiple(
            feats, feat_ranking, alpha, df_tune, cols, scoring1=scoring[0],
            scoring2=scoring[1], scoring3=scoring[2])
    if isinstance(model, Lasso) and len(feats) > 0:
        df_temp = get_lasso_feats(df_temp, key, transformer, X_select, y)
    elif isinstance(model, Lasso) and len(feats) == 0:
        df_temp['feat_n_used'] = len(feats)
        df_temp['features_used'] = [feats]
    return df_temp

# @ignore_warnings(category=ConvergenceWarning)
def execute_tuning(alpha_list, X, y, model_list, param_grid_dict, standardize, scoring, scoring_refit,
                   max_iter, random_seed, key, df_train, n_splits, n_repeats,
                   print_results=False):
    '''
    Execute model tuning, saving gridsearch hyperparameters for each number
    of features.
    '''
    cols = [
        'feat_n', 'alpha', 'score_train_mae', 'std_train_mae', 'score_val_mae',
        'std_val_mae',
        'score_train_mse', 'std_train_mse', 'score_val_mse', 'std_val_mse',
        'score_train_r2', 'std_train_r2', 'score_val_r2', 'std_val_r2',
        'tune_params', 'features', 'feat_ranking']

    param_grid_dict_key = param_grid_add_key(param_grid_dict, key)
    df_tune_feat_list = (None,) * len(model_list)
    # alpha_list = chunks[7]
    for alpha in alpha_list:
        X_select, feats, feat_ranking = feat_selection_lasso(
            X, y, alpha, max_iter, random_seed)

        print('Number of features: {0}'.format(len(feats)))
        temp_list = []
        for idx1, (model, param_grid) in enumerate(zip(model_list, param_grid_dict_key.values())):
            cv_rep_strat = get_repeated_stratified_kfold(
                df_train, n_splits, n_repeats, random_seed)

            param_grid_dc = deepcopy(param_grid)
            # will show a verbose warning if n_components exceeds n_feats
            if f'{key}n_components' in param_grid_dc:
                n_comp = param_grid_dc[f'{key}n_components']
                if len(feats) < max(n_comp):
                    print('Trimming excess components in <param_grid>...')
                    n_comp_trim = [i for i in n_comp if i <= len(feats)]
                    param_grid_dc[f'{key}n_components'] = n_comp_trim
                    # print('n_components: {0}'.format(param_grid_dc[f'{key}n_components']))
            df_tune_temp = tune_model(
                X_select, y, model, param_grid_dc,
                standardize, scoring, scoring_refit, key, cv_rep_strat,
                feats, feat_ranking, alpha, cols)
            if print_results is True:
                print_model(model)
                print('R2: {0:.3f}\n'.format(df_tune_temp['score_val_r2'].values[0]))
            temp_list.append(df_tune_temp)
        df_tune_feat_list = append_tuning_results(df_tune_feat_list, temp_list)
    return df_tune_feat_list

# len(df_tune_feat_list[0])
# len(df_tune_feat_list[1])
# @ignore_warnings(category=ConvergenceWarning)
def execute_tuning_pp(
        logspace_list, X1, y1, model_list, param_grid_dict, standardize,
        scoring, scoring_refit, max_iter, random_seed, key, df_train,
        n_splits, n_repeats, df_tune_all_list, n_jobs):
    '''
    Actual execution of hyperparameter tuning via multi-core processing
    '''
    # m = Manager()
    # lock = m.Lock()
    # chunks = chunk_by_n(reversed(logspace_list))
    # chunk_size = int(len(logspace_list) / (n_jobs*2)) + 1
    chunks = chunk_by_n(logspace_list, n_jobs*2)  # remember this shuffles logspace_list
    if len(logspace_list) < n_jobs * 2:
        chunks = chunk_by_n(logspace_list, n_jobs)
    # print('Length of logspace_list: {0}'.format(len(logspace_list)))
    # print('Number of chunks: {0}'.format(len(chunks)))
    chunk_avg = sum([len(i) for i in chunks]) / len(chunks)
    # print('Average length of each chunk: {0:.1f}'.format(chunk_avg))
    # print('Number of cores: {0}\n'.format(n_jobs))

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # for alpha, df_tune_feat_list in zip(reversed(logspace_list), executor.map(execute_tuning, it.repeat(X1), it.repeat(y1), it.repeat(model_list), it.repeat(param_grid_dict), reversed(logspace_list),
        #                                                                           it.repeat(standardize), it.repeat(scoring), it.repeat(scoring_refit), it.repeat(max_iter), it.repeat(random_seed),
        #                                                                           it.repeat(key), it.repeat(df_train), it.repeat(n_splits), it.repeat(n_repeats))):
        for df_tune_feat_list in executor.map(execute_tuning, chunks, it.repeat(X1), it.repeat(y1), it.repeat(model_list), it.repeat(param_grid_dict),
                                              it.repeat(standardize), it.repeat(scoring), it.repeat(scoring_refit), it.repeat(max_iter), it.repeat(random_seed),
                                              it.repeat(key), it.repeat(df_train), it.repeat(n_splits), it.repeat(n_repeats)):
                # chunksize=chunk_size))

            # print('type: {0}'.format(type(df_tune_feat_list[0])))
            # print('len: {0}'.format(len(df_tune_feat_list)))
            # print('Len: {0}'.format(len(df_tune_all_list)))
            # with lock:
            df_tune_all_list = append_tuning_results(df_tune_all_list, df_tune_feat_list)
    return df_tune_all_list


    # chunk_size = int(len(logspace_list) / (os.cpu_count()*2))
    # chunks = [logspace_list[x:x+chunk_size] for x in range(0, len(logspace_list), chunk_size)]
    # with ProcessPoolExecutor() as executor:
    #     array = [(X1, y1, model_list, param_grid_dict, alpha, standardize,
    #               scoring, scoring_refit, max_iter, random_seed, key, df_train,
    #               n_splits, n_repeats)
    #               for alpha in reversed(logspace_list)]
    #     # executor.map(execute_tuning, *zip(*array))
    #     for alpha, df_tune_feat_list in zip(
    #             reversed(logspace_list),
    #             executor.map(execute_tuning, *zip(*array))):
    #         # print('Alpha: {0}\nResult: {1}\n'.format(alpha, df_tune_feat_list))
    #         df_tune_all_list = append_tuning_results(
    #             df_tune_all_list, df_tune_feat_list)
    # return df_tune_all_list

def filter_tuning_results(df_tune_all_list, score):
    '''
    Remove dupilate number of features (keep only lowest error)
    '''
    df_tune_list = ()
    for df_tune in df_tune_all_list:
        df_tune = df_tune.reset_index(drop=True)
        array_idx = df_tune.groupby(['feat_n'])[score].transform(max) == df_tune[score]
        # if first non-zero feat_n row is NaN, include that so other dfs have same number of rows (PLS)
        if np.isnan(df_tune.loc[df_tune['feat_n'].idxmin(),score]):
            array_idx.loc[df_tune['feat_n'].idxmin()] = True
        df_tune['feat_n'] = df_tune['feat_n'].apply(pd.to_numeric)
        df_filtered = df_tune[array_idx].drop_duplicates(['feat_n']).sort_values('feat_n').reset_index(drop=True)
        df_tune_list += (df_filtered,)
    return df_tune_list

def summarize_tuning_results(df_tune_list, model_list, param_grid_dict,
                             key=''):
    '''
    Summarizes the hyperparameters from the tuning process into a single
    dataframe
    '''
    cols_params = ['feat_n']
    for k1, v1 in param_grid_dict.items():
        for k2, v2 in param_grid_dict[k1].items():
            k2_short = k2.replace(key, '')
            cols_params.append(k1 + '_' + k2_short)
            
    df_params = pd.DataFrame(columns=cols_params)
    # for each feature (if missing just put nan)
    feat_max = max([df['feat_n'].max() for df in df_tune_list])
    for feat_n in range(1,feat_max+1):
        data_dict = {i: np.nan for i in cols_params}
        data_dict['feat_n'] = [feat_n]
        for idx_df, df_original in enumerate(df_tune_list):
            df = df_original.copy()
            df.set_index('feat_n', inplace=True)
            if isinstance(model_list[idx_df], Lasso) and feat_n in df.index:
                las_params = df.loc[feat_n]['tune_params']
                try:
                    las_alpha = las_params[f'{key}alpha']
                except TypeError:  # when cell is nan instead of dict
                    continue  # go to next index where there is actually data
                data_dict['las_alpha'] = [las_alpha]
            elif isinstance(model_list[idx_df], SVR) and feat_n in df.index:
                svr_params = df.loc[feat_n]['tune_params']
                svr_kernel = svr_params[f'{key}kernel']
                try:
                    svr_gamma = svr_params[f'{key}gamma']
                except KeyError:
                    svr_gamma = np.nan
                svr_C = svr_params[f'{key}C']
                svr_epsilon = svr_params[f'{key}epsilon']
                data_dict['svr_kernel'] = [svr_kernel]
                data_dict['svr_gamma'] = [svr_gamma]
                data_dict['svr_C'] = [svr_C]
                data_dict['svr_epsilon'] = [svr_epsilon]
            elif isinstance(model_list[idx_df], RandomForestRegressor) and feat_n in df.index:
                rf_params = df.loc[feat_n]['tune_params']
                rf_min_samples_split = rf_params[f'{key}min_samples_split']
                rf_max_feats = rf_params[f'{key}max_features']
                data_dict['rf_min_samples_split'] = [rf_min_samples_split]
                data_dict['rf_max_feats'] = [rf_max_feats]
            elif isinstance(model_list[idx_df], PLSRegression) and feat_n in df.index:
                # The model shouldn't even be in model_list if there is no information for it.
                # This was changed when the PLS component list was modified to cut out
                # components that are greater than number of features (e.g., we can't
                # tune on 8 components with 4 features).
    
                # Thus, assume the "garbage" was filtered out already and we will only
                # arrive here if 'tune_params' exists.
                pls_params = df.loc[feat_n]['tune_params']
                if pd.notnull(pls_params):
                    pls_n_components = pls_params[f'{key}n_components']
                    pls_scale = pls_params[f'{key}scale']
                else:
                    pls_n_components = np.nan
                    pls_scale = np.nan
                data_dict['pls_n_components'] = [pls_n_components]
                data_dict['pls_scale'] = [pls_scale]
        df_summary_row = pd.DataFrame.from_dict(data=data_dict)
        df_params = df_params.append(df_summary_row)
    return df_params.reset_index(drop=True)

def append_tuning_results(df_tune_all_list, df_tune_feat_list):
    '''
    Appends tune_feat to tune_all as a list/tuple
    '''
    msg = ('<df_tune_all_list> and <df_tune_feat_list> must be the same '
           'length.')
    assert len(df_tune_all_list) == len(df_tune_feat_list), msg
    df_out_list = ()
    for df_all, df_single in zip(df_tune_all_list, df_tune_feat_list):

        if df_all is None and df_single is not None:
            df_all = df_single.copy()
        elif df_all is not None and df_single is not None:
            df_all = df_all.append(df_single)
        else:
            pass
        df_out_list += (df_all,)
    return df_out_list

def tuning_mode_count(df_params, svr_only=False):
    '''
    Calculates the mode and count of the most popular

    ``svr_only`` needs work
    '''
    cols_params = list(df_params.columns)
    if 'feat_n' in cols_params:
        cols_params.remove('feat_n')
    if svr_only is True:
        kernel_mode = df_params['svr_kernel'].mode().values[0]
        df_params_mode = df_params[df_params['svr_kernel'] == kernel_mode][cols_params].mode()
        n_obs = len(df_params[df_params['svr_kernel'] == kernel_mode][cols_params])
    else:
        df_params_mode = df_params[cols_params].mode()
        n_obs = len(df_params[cols_params])
    data = []
    for col_name in df_params_mode.columns:
        val_count = df_params_mode.loc[0][col_name]
        try:
            count = df_params[col_name].value_counts()[val_count]
        except TypeError:
            count = df_params[col_name].value_counts()[int(val_count)]
        except KeyError:
            count = np.nan
        data.append(count/n_obs)
    df_params_mode_count = pd.DataFrame(data=[data], columns=df_params_mode.columns)
    return df_params_mode, df_params_mode_count

def feats_readme(fname_feats_readme, fname_data, meta_bands, extra_feats=None):
    '''
    Describes the features being used from df_join to tune and train the
    models.

    Parameters:
        dir_feats (``str``): The directory to create the README.txt file
        bands (``dict``): Dictionary containing band number (keys) and
            center wavelength (values).
    '''
    with open(os.path.join(fname_feats_readme), 'w+') as f:
        f.write('Features available for tuning:\n\n')
        f.write('Feature number: Wavelength (for spectral and derivative '
                'features only)\n'
                'Any "extra" features are described by the column name from '
                'their input data source\n')
        f.write('Training data is saved at:\n')
        f.write('{0}\n'.format(fname_data))
        for k, v in sorted(meta_bands.items()):
            # print("{0}: {1}\n".format(k, v))
            f.write('{0}: {1}\n'.format(k, v))
        n = max(meta_bands)
        if isinstance(extra_feats, str):
            n += 1
            f.write('{0}: {1}\n'.format(n, extra_feats))
        elif isinstance(extra_feats, list):
            for ef in extra_feats:
                n += 1
                f.write('{0}: {1}\n'.format(n, ef))

def save_tuning_results(
        dir_out_tune, df_tune_list, model_list, df_params, df_params_mode,
        df_params_mode_count, meta_bands,
        fname_base='msi_00_000_measurement_units'):
    '''
    Saves all tuning results to ``dir_out_tune``

    Parameters:
        fname_base (``str``): the beginning of the final file names to be
            saved. Other information will be appended to this string to
            desribe predictions/scoring, as well as the model being used.
    '''
    tune_str = '-tuning-'
    for idx, model in enumerate(model_list):
        df = df_tune_list[idx]
        if isinstance(model, Lasso):
            model_str = 'lasso'
        elif isinstance(model, SVR):
            model_str = 'svr'
        elif isinstance(model, RandomForestRegressor):
            model_str = 'rf'
        elif isinstance(model, PLSRegression):
            model_str = 'pls'
        fname_model = os.path.join(
            dir_out_tune, fname_base + tune_str + model_str + '.csv')
        df.to_csv(fname_model, index=False)
    fname_tuning1 = os.path.join(dir_out_tune, fname_base + tune_str + 'summary.csv')
    df_params.to_csv(fname_tuning1, index=False)
    fname_tuning2 = os.path.join(dir_out_tune, fname_base + tune_str + 'mode.csv')
    df_params_mode.to_csv(fname_tuning2, index=False)
    fname_tuning3 = os.path.join(dir_out_tune, fname_base + tune_str + 'mode-count.csv')
    df_params_mode_count.to_csv(fname_tuning3, index=False)

def load_tuning_results(dir_out_tune, model_list,
                        fname_base='msi_00_000_measurement_units'):
    tune_str = '-tuning-'
    df_tune_list = ()
    for model in model_list:
        if isinstance(model, Lasso):
            model_str = 'lasso'
        elif isinstance(model, SVR):
            model_str = 'svr'
        elif isinstance(model, RandomForestRegressor):
            model_str = 'rf'
        elif isinstance(model, PLSRegression):
            model_str = 'pls'
        fname_model = os.path.join(
            dir_out_tune, fname_base + tune_str + model_str + '.csv')
        # df_tune_list.append(pd.read_csv(fname_model))
        df_tune_list += (pd.read_csv(fname_model),)

    fname_params = os.path.join(
                dir_out_tune, fname_base + tune_str + 'summary.csv')
    df_params = pd.read_csv(fname_params)
    fname_mode = os.path.join(
                dir_out_tune, fname_base + tune_str + 'mode.csv')
    df_params_mode = pd.read_csv(fname_mode)
    fname_count = os.path.join(
                dir_out_tune, fname_base + tune_str + 'mode-count.csv')
    df_params_mode_count = pd.read_csv(fname_count)
    return df_tune_list, df_params, df_params_mode, df_params_mode_count

# In[Model testing functions]
def get_errors(model, X, y):
    '''
    Returns the MAE, RMSE, MSLE, and R2 for a fit model
    '''
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return y_pred, mae, rmse, r2

def prep_pred_dfs(df_test, feat_n_list, y_label='nup_kgha'):
    cols_scores = ['feat_n', 'feats', 'score_train_mae', 'score_test_mae',
                   'score_train_rmse', 'score_test_rmse',
                   'score_train_r2', 'score_test_r2']
    cols_meta = ['study', 'date', 'plot_id', 'trt', 'rate_n_pp_kgha',
                 'rate_n_sd_plan_kgha', 'rate_n_total_kgha', 'growth_stage',
                 y_label]
    cols_preds = cols_meta + feat_n_list
    df_pred = pd.DataFrame(columns=cols_preds)
    df_pred[cols_meta] = df_test[cols_meta]
    df_score = pd.DataFrame(columns=cols_scores)
    return df_pred, df_score

def get_params(params, idx=None, col='tune_params'):
    '''
    Retrieves dictionary of paramters to use for model training; can be either
    pd.DataFrame or dictionary; if dictionary, just returns itself
    '''
    if isinstance(params, pd.DataFrame):
        assert idx is not None, ('"idx" must be defined')
        try:
            params_dict = literal_eval(params.iloc[idx][col])
        except ValueError:
            params_dict = params.iloc[idx][col]
        # except IndexError:

    elif isinstance(params, dict):  # else must be dict
        params_dict = params.copy()
    else:
        params_dict = None
    return params_dict

def get_params_row(row, col='tune_params'):
    '''
    Retrieves dictionary of paramters to use for model training; can be either
    a pd.Series (single row of pd.DataFrame) or dictionary; if dictionary, just
    returns itself
    '''
    if isinstance(row, pd.Series):
        try:
            params_dict = literal_eval(row[col])
        except ValueError:
            params_dict = row[col]
        # except IndexError:

    elif isinstance(row, dict):  # else must be dict
        params_dict = row.copy()
    else:
        params_dict = None
    return params_dict

def get_params_all(params_las, params_svr, params_rf, params_pls, idx=None,
                   col='tune_params'):
    '''
    Retrieves parameters for all models
    '''
    params_las_dict = get_params(params_las, idx=idx, col=col)
    params_svr_dict = get_params(params_svr, idx=idx, col=col)
    params_rf_dict = get_params(params_rf, idx=idx, col=col)
    params_pls_dict = get_params(params_pls, idx=idx, col=col)
    return params_las_dict, params_svr_dict, params_rf_dict, params_pls_dict

def get_feats_int(row, y_col='features', idx=False):
    '''
    Returns a list of integers that represent band numbers
    '''
    try:
        feats = row[y_col].strip('[]').split(' ')
        feats_int = []
        for item in feats:
            if item == '':
                pass
            elif '\n' in item:
                item = item.replace('\n','')
                feats_int.append(int(item))
            else:
                feats_int.append(int(item))
    except AttributeError:
        feats_int = list(row['features'])
    return feats_int

def check_for_multiple(model, df):
    '''
    Checks if model and df are a list; if so, we have to test multiple models

    Returns ``model`` and ``df`` as lists. If ``model`` was input as a single
    model, it is returned as a list with lenght equal to one.
    '''
    msg1 = ('If passing multiple models, ``model`` and ``df`` must both be '
            'lists and must have equal lengths.')
    if isinstance(model, list):
        assert isinstance(df, list), msg1
        assert len(model) == len(df), msg1
    else:
        model_list = [model]
        df_list = [df]
    return model_list, df_list

def predict_me(model, params_dict, standardize, X1_select, y1,
               X1_test_select, y1_test, feats_int, df_pred, df_score):
    '''
    Uses params_dict to set up the model, train, test, and report error
    '''
    if isinstance(params_dict, dict):
        # model_pls = PLSRegression(tol=1e-9)
        transformed_model = TransformedTargetRegressor(
            regressor=model, transformer=PowerTransformer(
                'yeo-johnson', standardize=standardize))
        transformed_model.set_params(**params_dict)
        transformed_model.fit(X1_select, y1)
        _, train_mae, train_rmse, train_r2 = get_errors(
            transformed_model, X1_select, y1)
        y_pred, test_mae, test_rmse, test_r2 = get_errors(
            transformed_model, X1_test_select, y1_test)
        data = [len(feats_int), feats_int, train_mae, test_mae,
                train_rmse, test_rmse, train_r2, test_r2]
    elif isinstance(model, PLSRegression) and pd.isnull(params_dict):
        # PLS requires at least two features; just record Nan for this row
        data = [len(feats_int), feats_int]
        data.extend([np.nan] * 6)
        y_pred = [np.nan] * len(df_pred)
    # else:
    #     print('nope')
    #     print(params_dict)
    #     print(type(model))
    df_temp_scores = pd.DataFrame(data=[data], columns=df_score.columns)
    df_score = df_score.append(df_temp_scores)
    df_pred[len(feats_int)] = y_pred
    return df_pred, df_score

def test_predictions(df_test, X1, y1, X1_test, y1_test, model_list,
                     df_tune_list, feat_n_list, y_label='nup_kgha',
                     max_iter=5000, standardize=False, key='',
                     n_feats_linspace=50):
    '''
    Predicts ``y1_test`` for one or many models.

    model (``sklearn`` model): The model to be used to make predictions. Can be
        a list of models to test multiple models, but ``df`` must also be a
        list with an equal number with tuning hyperparameters corresponding to
        each model.
    df (``pandas.DataFrame``): Dataframe containing the tuning hyperparameters
        to use for model testing. Can be a list with equal length of ``model``.
    '''
    df_pred, df_score = prep_pred_dfs(
        df_test, feat_n_list, y_label=y_label)
    df_pred_list = ()
    df_score_list = ()
    for i in range(len(model_list)):
        # df_pred_list.append(df_pred.copy())
        # df_score_list.append(df_score.copy())
        df_pred_list += (df_pred.copy(),)
        df_score_list += (df_score.copy(),)
    # for idx_tune, row1 in df_tune_list[0].iterrows():  # must be sure 'features' exists in all dfs
    #     if idx_tune == 0:
    #         break
        # feats_int = get_feats_int(row, y_col='features')
        # X1_select = X1[:, feats_int]
        # X1_test_select = X1_test[:, feats_int]
        # for idx_model, model in enumerate(model_list):
        #     df_tune = df_tune_list[idx_model]
        #     df_pred = df_pred_list[idx_model]
        #     df_score = df_score_list[idx_model]
        #     feats_int = get_feats_int(row, y_col='features')
        #     X1_select = X1[:, feats_int]
        #     X1_test_select = X1_test[:, feats_int]
            # params_dict = get_params(df_tune, idx=idx_tune)

    for feat_n in df_tune_list[0]['feat_n']:
        # if feat_n == 1:
        #     break
        for idx_model, model in enumerate(model_list):
            df_tune = df_tune_list[idx_model]
            df_pred = df_pred_list[idx_model]
            df_score = df_score_list[idx_model]

            row = df_tune[df_tune['feat_n'] == feat_n]
            # print(row)
            if len(row) == 0:
                continue
            row = row.squeeze()  # changes row from pd.Dataframe to pd.Series
            if pd.isnull(row.score_train_mae):  # not sure why, but sometimes row is len == 0, and sometimes it is len == 1 with all null
                continue
            feats_int = get_feats_int(row, y_col='features')
            X1_select = X1[:, feats_int]
            X1_test_select = X1_test[:, feats_int]

            # params_dict = get_params(df_tune, idx=idx_tune)
            # print(model)
            # print(row)
            params_dict = get_params_row(row)
            if (isinstance(model, PLSRegression) and
                isinstance(params_dict, dict) and
                params_dict[f'{key}n_components'] > len(feats_int)):
                params_dict[f'{key}n_components'] = len(feats_int)
            df_pred, df_score = predict_me(
                model, params_dict, standardize, X1_select, y1,
                X1_test_select, y1_test, feats_int, df_pred, df_score)
            # df_pred_list[idx_model] = df_pred
            # df_score_list[idx_model] = df_score
            df_pred_list = df_pred_list[:idx_model] + (df_pred,) + df_pred_list[idx_model+1:]
            df_score_list = df_score_list[:idx_model] + (df_score,) + df_score_list[idx_model+1:]
                # model, df_tune, params_dict, standardize, X1_select, y1,
                # X1_test_select, y1_test, feats_int, df_pred, df_score)
    for idx_model, model in enumerate(model_list):
        df_score = df_score_list[idx_model]
        new_index = pd.Index(range(1,n_feats_linspace+1), name='feat_n')
        df_score = df_score.set_index('feat_n').reindex(new_index).reset_index()
        # df_score_list[idx_model] = df_score
        df_score_list = df_score_list[:idx_model] + (df_score,) + df_score_list[idx_model+1:]
    return df_pred_list, df_score_list

def set_up_output_dir(dir_results, msi_run_id, grid_idx, y_label, feat_name,
                      test_or_tune='tuning'):
    '''
    Ensures all folder directories are created, then returns each directory
    level for easy access for reading and writing of files for a given
    ``msi_run_id``, ``grid_idx``, ``y_label``, and ``feat_name``.

    test_or_tune (``str``): Should be either 'tuning' or 'testing'
    '''
    folder_msi = 'msi_' + str(msi_run_id) + '_' + str(grid_idx).zfill(3)
    dir_out0 = os.path.join(dir_results, folder_msi)  # folder 1
    # if not os.path.isdir(dir_out0):
    #     pathlib.Path(dir_out0).mkdir(parents=True, exist_ok=True)
    dir_out1 = os.path.join(dir_out0, y_label)  # folder 2
    # if not os.path.isdir(dir_out1):
    #     pathlib.Path(dir_out1).mkdir(parents=True, exist_ok=True)
    dir_out2 = os.path.join(dir_out1, feat_name)  # folder 3
    # if not os.path.isdir(dir_out2):
    #     pathlib.Path(dir_out2).mkdir(parents=True, exist_ok=True)
    dir_out3 = os.path.join(dir_results, folder_msi, y_label, feat_name, test_or_tune)  # folder 4
    # if not os.path.isdir(dir_out3):
    pathlib.Path(dir_out3).mkdir(parents=True, exist_ok=True)
    if test_or_tune == 'testing':
        # if not os.path.isdir(os.path.join(dir_out3, 'figures')):
        pathlib.Path(os.path.join(dir_out3, 'figures')).mkdir(parents=True, exist_ok=True)
    return [dir_out0, dir_out1, dir_out2, dir_out3], [folder_msi, y_label, feat_name, test_or_tune]

def set_up_summary_files(dir_out, y_label, n_feats, msi_run_id):
    '''
    meta_info = [msi_run_id, grid_idx, y_label, extra_feats]
    '''
    cols = ['msi_run_id', 'grid_idx', 'response_label', 'extra_feats', 'model_name']
    feat_list = list(range(1,50+1))
    cols.extend(feat_list)

    msi_str = 'msi_' + str(msi_run_id) + '_'
    fname_sum_mae = os.path.join(dir_out, msi_str + '_'.join((y_label, 'MAE')) + '.csv')
    fname_sum_rmse = os.path.join(dir_out, msi_str + '_'.join((y_label, 'RMSE')) + '.csv')
    fname_sum_r2 = os.path.join(dir_out, msi_str + '_'.join((y_label, 'R2')) + '.csv')
    if not os.path.isfile(fname_sum_mae):
        df_sum_mae = pd.DataFrame(columns=cols)
        df_sum_mae.to_csv(fname_sum_mae, index=False)
    if not os.path.isfile(fname_sum_rmse):
        df_sum_rmse = pd.DataFrame(columns=cols)
        df_sum_rmse.to_csv(fname_sum_rmse, index=False)
    if not os.path.isfile(fname_sum_r2):
        df_sum_r2 = pd.DataFrame(columns=cols)
        df_sum_r2.to_csv(fname_sum_r2, index=False)

def append_test_scores(dir_out, y_label, df_score_list, model_list, metadata):
    '''
    metadata = [msi_run_id, row.name, y_label, extra_feats]

    '''
    cols = ['msi_run_id', 'grid_idx', 'response_label', 'extra_feats', 'model_name']

    msi_str = 'msi_' + str(metadata[0]) + '_'
    fname_sum_mae = os.path.join(dir_out, msi_str + '_'.join((y_label, 'MAE')) + '.csv')
    fname_sum_rmse = os.path.join(dir_out, msi_str + '_'.join((y_label, 'RMSE')) + '.csv')
    fname_sum_r2 = os.path.join(dir_out, msi_str + '_'.join((y_label, 'R2')) + '.csv')

    for idx_model, model in enumerate(model_list):
        meta = metadata + [str(model).split('(')[0]]
        # metadata.append(str(model).split('(')[0])
        s_metadata = pd.Series(meta, index=cols)

        df_score = df_score_list[idx_model]
        df_score.set_index('feat_n')
        s_score_mae = s_metadata.append(df_score['score_test_mae'].transpose())
        s_score_rmse = s_metadata.append(df_score['score_test_rmse'].transpose())
        s_score_r2 = s_metadata.append(df_score['score_test_r2'].transpose())

        df_score_mae = pd.DataFrame(s_score_mae).T
        df_score_rmse = pd.DataFrame(s_score_rmse).T
        df_score_r2 = pd.DataFrame(s_score_r2).T

        df_score_mae.to_csv(fname_sum_mae, header=None, mode='a', index=False,
                            index_label=df_score_mae.columns)
        df_score_rmse.to_csv(fname_sum_rmse, header=None, mode='a',
                             index=False, index_label=df_score_rmse.columns)
        df_score_r2.to_csv(fname_sum_r2, header=None, mode='a', index=False,
                           index_label=df_score_r2.columns)

def save_test_results(dir_out_test, df_pred_list, df_score_list, model_list,
                      fname_base='msi_00_000_measurement_units'):
    '''
    Parameters:
        fname_base (``str``): the beginning of the final file names to be
            saved. Other information will be appended to this string to
            desribe predictions/scoring, as well as the model being used.
    '''
    pred_str = '_test-preds-'
    score_str = '_test-scores-'
    for idx, model in enumerate(model_list):
        df_pred = df_pred_list[idx]
        df_score = df_score_list[idx]
        if isinstance(model, Lasso):
            model_str = 'lasso'
        elif isinstance(model, SVR):
            model_str = 'svr'
        elif isinstance(model, RandomForestRegressor):
            model_str = 'rf'
        elif isinstance(model, PLSRegression):
            model_str = 'pls'
        fname_pred = os.path.join(
            dir_out_test, fname_base + pred_str + model_str + '.csv')
        fname_score = os.path.join(
            dir_out_test, fname_base + score_str + model_str + '.csv')
        df_pred.to_csv(fname_pred, index=False)
        df_score.to_csv(fname_score, index=False)

def load_test_results(dir_out_test, model_list,
                      fname_base='msi_00_000_measurement_units'):
    '''
    Loads all testing results from ``dir_out_test``
    '''
    pred_str = '_test-preds-'
    score_str = '_test-scores-'
    df_pred_list = ()
    df_score_list = ()
    for idx, model in enumerate(model_list):
        if isinstance(model, Lasso):
            model_str = 'lasso'
        elif isinstance(model, SVR):
            model_str = 'svr'
        elif isinstance(model, RandomForestRegressor):
            model_str = 'rf'
        elif isinstance(model, PLSRegression):
            model_str = 'pls'
        fname_pred = os.path.join(
            dir_out_test, fname_base + pred_str + model_str + '.csv')
        fname_score = os.path.join(
            dir_out_test, fname_base + score_str + model_str + '.csv')
        # df_pred_list.append(pd.read_csv(fname_pred))
        # df_score_list.append(pd.read_csv(fname_score))
        df_pred_list += (pd.read_csv(fname_pred),)
        df_score_list += (pd.read_csv(fname_score),)
    return df_pred_list, df_score_list


# In[Plotting functions]
def calc_r2(s_x, s_y, print_out=False):
    s_x = s_x.values
    s_y = s_y.values
    reg1 = LinearRegression().fit(s_x.reshape(-1,1), s_y.reshape(-1,1))
    r_2 = reg1.score(s_x.reshape(-1,1), s_y.reshape(-1,1))
    y_pred = reg1.predict(s_x[:, np.newaxis])
    rmse = np.sqrt(mean_squared_error(s_y, y_pred))
    x_lin = np.linspace(s_x.min()-s_x.min()*0.1, s_x.max()+s_x.max()*0.1, num=20)
    y_lin = reg1.predict(x_lin[:, np.newaxis])
    if print_out is True:
        print(r'R^2 = {0:.3f}'.format(r_2))
        print(r'RMSE = {0:.1f}'.format(rmse))
    return r_2, rmse, x_lin, y_lin

def _annotate_arrow(ax, str_r2, x_arrow, y_arrow, xytext=(0.05, 0.83), ha='left',
                    va='top', fontsize=16, color='#464646'):
    boxstyle_str = 'round, pad=0.5, rounding_size=0.15'
    ax.annotate(
        str_r2,
        xy=(x_arrow, y_arrow),
        xytext=xytext,  # loc to place text
        textcoords='axes fraction',  # placed relative to axes
        ha=ha, va=va, fontsize=fontsize,
        color=color,
        bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                  ec=(0.5, 0.5, 0.5), alpha=0.7),
        arrowprops=dict(arrowstyle='-|>',
                        color=color,
    #                    patchB=el,
                        shrinkA=0,
                        shrinkB=0,
                        connectionstyle='arc3,rad=-0.2',
                        alpha=0.4))

def best_fit_line(df, x_col, y_col, ax, fontsize=16, linecolor='#464646',
                  fontcolor='#464646', xytext=(0.95, 0.05), ha='right', va='bottom'):
#    df = df_preds_las.copy()
    r_2, rmse, x_lin, y_lin = calc_r2(df[x_col], df[y_col])
    lin_r2_las = ax.plot(x_lin, y_lin, color=linecolor, alpha=0.6, linestyle='-')
    str_r2 = r'Best-fit line' + '\n' + 'R$^{2}$'
    str_r2_las = '{0} = {1:.3f}'.format(str_r2, r_2)
    _annotate_arrow(ax, str_r2_las, x_lin[15], y_lin[15], xytext=xytext,
                    ha=ha, va=va, fontsize=fontsize,
                       color=fontcolor)

def prediction_error_label(df_scores, feat_n, ax, fontsize, fontcolor,
                           xy=(0.05, 0.1), ha='left', va='top', hsanalyze=None,
                           units=None):
    score_row = df_scores[df_scores['feat_n'] == int(feat_n)]
    if hsanalyze is not None:
        try:  # works when just finished training the model
            band_nums = hsanalyze.io.tools.get_band_num(score_row['feats'].values[0])
        except TypeError:  # works when loading in data from a previous run
            band_nums = hsanalyze.io.tools.get_band_num(literal_eval(score_row['feats'].values[0]))
        wl_list = []
        for band in band_nums:
            wl = hsanalyze.io.tools.get_wavelength(band)
            wl_list.append(wl)
        print('The following features were used: {0}'.format(wl_list))
    if units is None:
        units = ''
    mae = score_row['score_test_mae'].values[0]
    rmse = score_row['score_test_rmse'].values[0]
    str_err = ('Prediction error\nMAE   = {0:.2f}{2}\nRMSE = {1:.2f}{2}'
               ''.format(mae, rmse, units))
    boxstyle_str = 'round, pad=0.5, rounding_size=0.15'
    ax.annotate(
        str_err,
        xy=xy,
        xycoords='axes fraction',
        ha=ha, va=va, fontsize=fontsize,
        color=fontcolor,
        bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                  ec=(0.5, 0.5, 0.5), alpha=0.7))

def stratified_error(df_pred, y_col, feat_n, ax, levels=[25, 50, 100],
                     fontsize=16, fontcolor='#464646', linecolor='#464646',
                     alpha=0.7):
    font_scale = 16/fontsize
    last_level = 0
    for idx in range(len(levels) + 1):
        try:
            level = levels[idx]
        except IndexError:
            pass
        if idx == 0:
            df_preds = df_pred[df_pred[y_col] < level]
        elif idx == len(levels) - 1:
            df_preds = df_pred[df_pred[y_col] >= last_level]
        else:
            df_preds = df_pred[(df_pred[y_col] >= last_level) & (df_pred[y_col] < level)]
        y_pos = np.mean([level, last_level])
        print(len(df_preds))
        mae = mean_absolute_error(df_preds[y_col], df_preds[feat_n])
        ax.annotate('{0:.1f}'.format(mae), xy=(184.5, y_pos), xytext=(186, y_pos),
                    xycoords='data', annotation_clip=False,
                    fontsize=fontsize*0.65, rotation=270,
                    ha='left', va='center', color=fontcolor,
                    arrowprops=dict(
                         arrowstyle='-[, widthB={0}, lengthB=0.5'.format(1.5*font_scale),  # 1.5, 1.5, 3.15, 4.85
                         lw=0.7, ls=(0, (3, 3)),
                         color=linecolor, alpha=alpha))
        last_level = level

def get_min_max(df, feat_n, y_col):
    try:
        max_pred = df[feat_n].max()
    except KeyError:
        feat_n = str(feat_n)
        max_pred = df[feat_n].max()
    max_plot = math.ceil(np.nanmax([max_pred, df[y_col].max()]))
    # max_plot = math.ceil(np.nanmax([5.130879038769381, 6.102534]))
    min_plot = int(np.nanmin([df[feat_n].min(), df[y_col].min()]))
    return max_plot, min_plot

def plot_meas_pred(feat_n, y_col, df_preds, ax,
                   x_label='Predicted', y_label='Measured', units=None,
                   max_plot=None, min_plot=None, legend='full',
                   fontsize=16, fontcolor='#464646', linecolor='#464646'):
    '''
    Creates a single plot showing measured (y-axis) vs. predicted (x-axis)

    ``feat_n`` should be the column heading of the predicted values to plot from
    ``df_preds``. ``feat_n`` is typically an integer for a given number of
    features, then the column contains the predicted values to plot.
    '''
    if units is None:
        units = ''
    else:
        units = ' ({0})'.format(units)
    # markers = dict(V6='P', V8='X', V10='d', V14='8')
    markers = ('o', 'X', 's', 'P', 'D', '^', 'v', 'p', 'h', 'd', '<', '*', 'H',
               '8', '>')
    hue_order = sorted(df_preds['study'].unique())
    try:
        df_preds['date'] = df_preds['date'].dt.date  # Be sure date does not include time
    except AttributeError:
        pass
    style_order = sorted(df_preds['date'].unique())
    if len(style_order) > 15:
        style = 'month'
        df_preds['month'] = pd.DatetimeIndex(df_preds['date']).month_name()
        style_order = sorted(df_preds['month'].unique())
    else:
        style = 'date'
    colors = sns.color_palette('pastel', len(hue_order))
    if max_plot is None:
        max_plot, _ = get_min_max(df_preds, feat_n, y_col)
    if min_plot is None:
        _, min_plot = get_min_max(df_preds, feat_n, y_col)

    x_lin = np.linspace(min_plot, max_plot, 2)
    ax = sns.lineplot(x=x_lin, y=x_lin, color=linecolor, ax=ax, zorder=0.8, linewidth=2)
    ax.lines[0].set_linestyle('--')
    ax = sns.scatterplot(x=feat_n, y=y_col, data=df_preds,
                         hue='study', style=style,
                         hue_order=hue_order, style_order=style_order,
                         markers=markers[:len(style_order)], ax=ax,
                         legend=legend, palette=colors)
    ax.set_ylim([min_plot, max_plot])
    ax.set_xlim([min_plot, max_plot])
    ax.set_xlabel(x_label + units, fontsize=fontsize, color=fontcolor)
    ax.set_ylabel(y_label + units, fontsize=fontsize, color=fontcolor)
    ax.tick_params(labelsize=int(fontsize*0.95), colors=fontcolor)
    return ax, colors

def plot_get_n_models(dir_results, folder_name):
    '''
    Gets the number of models whose data are populated
    '''
    (df_preds_las, df_preds_svr, df_preds_rf, df_preds_pls,
     df_score_las, df_score_svr, df_score_rf, df_score_pls) =\
        load_test_results(os.path.join(dir_results, folder_name),
        name_append=folder_name.replace('_', '-'))
    preds_list = [df_preds_las, df_preds_svr, df_preds_rf, df_preds_pls]
    preds_list_nonnull = []
    for item in preds_list:
        if item is not None:
            preds_list_nonnull.append(item)
    score_list = [df_score_las, df_score_svr, df_score_rf, df_score_pls]
    score_list_nonnull = []
    for item in score_list:
        if item is not None:
            score_list_nonnull.append(item)
    n_models = len(preds_list_nonnull)
    return n_models, preds_list_nonnull, score_list_nonnull

def plot_titles(ax, title_text, fontsize=16, fontcolor='white',
                facecolor='#585858'):
    '''
    Add titles to plots
    '''
    t1 = ax.set_title(
        title_text, fontsize=fontsize*1.1, fontweight='bold', color=fontcolor,
        bbox=dict(color=facecolor))
    t1.get_bbox_patch().set_boxstyle(
        'ext', pad=0.25, width=ax.get_window_extent().width)
    return t1

def on_resize(event, title_list, axes):
    for t, ax in zip(title_list, axes):
        t.get_bbox_patch().set_boxstyle(
            'ext', pad=0.2, width=ax.get_window_extent().width)

def plot_legend(fig, ax, df_preds, feat_n, colors,
                study_labels=None, date_labels=None,
                fontsize=16, handle_color='#464646', handle_size=80,
                label_color='#464646',
                ncol=4):
    study_n = len(df_preds['study'].unique())
    # date_n = len(df_preds['date'].unique())
    h, l = ax.get_legend_handles_labels()
    h1 = h[1:study_n+2]
    h2 = h[study_n+2:]
    l1 = l[1:study_n+2]
    l2 = l[study_n+2:]
    obs_study_labels = []
    obs_study_list = []
    for study in df_preds['study'].unique():
        obs_study_labels.append(study)
        obs_n = len(df_preds[df_preds['study'] == study])
        obs_study_list.append(obs_n)
    if study_labels is None:
        study_labels = obs_study_labels.copy()
    l1[-1] = 'No. features: {0}'.format(feat_n)
    for handle in h2:
        handle.set_color(handle_color)
        handle._sizes = [handle_size]
    h1_new = []
    for i in range(len(h1)-1):
        handle_new = mlines.Line2D([], [], color=colors[i], marker='s', linestyle='None',
                                   markersize=10, label=l1[i])
        h1_new.append(handle_new)
    h1_new.append(mlines.Line2D([], [], alpha=0.0, label=l1[-1]))
    leg = fig.legend(h2 + h1_new, l2 + l1, loc='upper center',
                     bbox_to_anchor=(0.5, 1.0),
                     fontsize=fontsize*0.85, framealpha=0.85,
                     ncol=ncol, handletextpad=0.1,  # spacing between handle and label
                     columnspacing=0.5,
                     frameon=True, edgecolor=label_color)
    for text in leg.get_texts():
        text.set_color(label_color)
    ax.legend().remove()
    return leg

def legend_resize(fig, leg, twinx=False, twinx_right=0.93):
    '''
    Draws canvas so legend and figure size can be determined, then adjusts
    figure size so it fits well with the figure.
    '''
    fig.canvas.draw()
    fig.tight_layout()
    # height_leg = fig.legends[0].get_window_extent().height  # we recreated the legend (and made a second legend)
    # height_fig = fig.get_window_extent().height
    height_leg = leg.get_window_extent().height  # we recreated the legend (and made a second legend)
    height_fig = fig.get_window_extent().height
    legend_adjust = (1 - (height_leg / height_fig)) * 0.98
    right = twinx_right if twinx is True else 1
    fig.tight_layout(rect=[0, 0, right, legend_adjust])
    return fig

def plot_scores_feats(df, ax, palette, legend, obj='mae', ls_1='-',
                      lw_a=1.5, y_label=None, units=None, fill_std=False,
                      fontsize=16, fontcolor='#464646'):
    if fill_std is True:
        df_wide = df[['feat_n', 'score_train_' + obj, 'score_test_' + obj, 'std_train_' + obj, 'std_test_' + obj]].apply(pd.to_numeric).set_index('feat_n')
    else:
        df_wide = df[['feat_n', 'score_train_' + obj, 'score_test_' + obj]].apply(pd.to_numeric).set_index('feat_n')
    if df_wide['score_train_' + obj].iloc[1] < 0:
        df_wide[['score_train_' + obj, 'score_test_' + obj]] = df_wide[['score_train_' + obj, 'score_test_' + obj]] * -1

    ax = sns.lineplot(
        data=df_wide[['score_train_' + obj, 'score_test_' + obj]], ax=ax, palette=palette, legend=legend)
    ax.lines[0].set_linewidth(lw_a)
    ax.lines[0].set_linestyle(ls_1)
    ax.lines[1].set_linestyle(ls_1)
    if y_label is None:
        y_label = 'Error'
    if units is None:
        units = ''
    else:
        units = ' ({0})'.format(units)
    ax.set_ylabel(y_label + units, fontsize=fontsize, color=fontcolor)
    ax.set_xlabel('Number of features', fontsize=fontsize, color=fontcolor)
    ax.tick_params(labelsize=fontsize*0.85, colors=fontcolor,
                   labelleft=True)
    if fill_std is True:
        x_feats = df_wide.index
        std_l = df_wide['score_test_' + obj].values - df_wide['std_test_' + obj].values
        std_u = df_wide['score_test_' + obj].values + df_wide['std_test_' + obj].values
        ax.fill_between(x_feats, std_l, std_u, facecolor=palette[1], alpha=0.15)
    return ax

def sync_axis_grids(ax, ax2, ax_min=None, ax_max=None, ax2_min=None, ax2_max=None):
    l = ax.get_ylim() if ax_min is None else (ax_min, ax_max)
    l2 = ax2.get_ylim() if ax2_min is None else (ax2_min, ax2_max)
    ax.set_ylim(l)
    ax2.set_ylim(l2)
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(ax.get_yticks())
    ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return ax, ax2

def plot_secondary(df, ax, palette, legend, obj='r2', ls_2='--',
                   lw_b=1, fontsize=16, fontcolor='#464646', fill_std=False):
    if fill_std is True:
        df_wide = df[['feat_n', 'score_train_' + obj, 'score_test_' + obj, 'std_train_' + obj, 'std_test_' + obj]].apply(pd.to_numeric).set_index('feat_n')
    else:
        df_wide = df[['feat_n', 'score_train_' + obj, 'score_test_' + obj]].apply(pd.to_numeric).set_index('feat_n')
    if df_wide['score_train_' + obj].iloc[1] < 0:
        df_wide[['score_train_' + obj, 'score_test_' + obj]] = df_wide[['score_train_' + obj, 'score_test_' + obj]] * -1
    ax2 = ax.twinx()
    ax2.grid()
    ax2 = sns.lineplot(data=df_wide[['score_train_' + obj, 'score_test_' + obj]], ax=ax2, palette=palette, legend=legend)
    ax2.lines[0].set_linewidth(lw_b)
    ax2.lines[1].set_linewidth(lw_b)
    ax2.lines[0].set_linestyle(ls_2)
    ax2.lines[1].set_linestyle(ls_2)
    ax2.set_ylabel(r'R$^{2}$', fontsize=fontsize, color=fontcolor, rotation=0, labelpad=15)
    ax2.tick_params(labelsize=fontsize*0.85, colors=fontcolor, labelright=True)
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1],
                           len(ax.get_yticks())))
    if fill_std is True:
        x_feats = df_wide.index
        std_l = df_wide['score_test_' + obj].values - df_wide['std_test_' + obj].values
        std_u = df_wide['score_test_' + obj].values + df_wide['std_test_' + obj].values
        ax2.fill_between(x_feats, std_l, std_u, facecolor=palette[1], alpha=0.15)
    return ax, ax2

def plot_all_features_single(df, ax, palette, legend, obj1='mae', linewidth=2,
                             linestyle='-'):
    df_wide = df[['feat_n', 'score_test_' + obj1]].apply(pd.to_numeric).set_index('feat_n')
    if df_wide['score_test_' + obj1].iloc[1] < 0:
        df_wide[['score_test_' + obj1]] = df_wide[['score_test_' + obj1]] * -1
    x_feats = df_wide.index
    ax = sns.lineplot(data=df_wide[['score_test_' + obj1]], ax=ax, palette=palette, legend=legend)
#    ax1b = ax1a.twinx()
#    ax1b = sns.lineplot(data=df_wide[['score_test_' + obj2]], ax=ax1b, palette=[palette[1]], legend=legend)
    try:
        ax.lines[-1].set_linewidth(linewidth)
#        ax.lines[-2].set_linewidth(linewidth)
        ax.lines[-1].set_linestyle(linestyle)
#        ax.lines[-2].set_linestyle(linestyle)
    except:
        print('Lines do not exist, may be NaN')
    return ax

def min_max_scores_low(df_scores, obj1='mae', obj2='r2'):
    min_obj1 = np.min([df_scores['score_train_' + obj1].min(),
                       df_scores['score_test_' + obj1].min()])
    max_obj1 = np.max([df_scores['score_train_' + obj1].max(),
                       df_scores['score_test_' + obj1].max()])
    min_obj2 = np.min([df_scores['score_train_' + obj2].min(),
                       df_scores['score_test_' + obj2].min()])
    max_obj2 = np.max([df_scores['score_train_' + obj2].max(),
                       df_scores['score_test_' + obj2].max()])
    return min_obj1, max_obj1, min_obj2, max_obj2

def min_max_scores_high(score_list, obj1, obj2, pct=0.05):
    min_obj1, max_obj1, min_obj2, max_obj2 = [None] * 4
    for df_score in score_list:
        if min_obj1 is None:
            min_obj1, max_obj1, min_obj2, max_obj2 = min_max_scores_low(
                df_score, obj1=obj1, obj2=obj2)
        else:
            min_obj1b, max_obj1b, min_obj2b, max_obj2b = min_max_scores_low(
                df_score, obj1=obj1, obj2=obj2)
            min_obj1 = min_obj1b if min_obj1b < min_obj1 else min_obj1
            max_obj1 = max_obj1b if max_obj1b > max_obj1 else max_obj1
            min_obj2 = min_obj2b if min_obj2b < min_obj2 else min_obj2
            max_obj2 = max_obj2b if max_obj2b > max_obj2 else max_obj2
    coef_obj1 = (max_obj1 - min_obj1) * pct
    coef_obj2 = (max_obj2 - min_obj2) * pct
    min_obj1 -= coef_obj1
    max_obj1 += coef_obj1
    min_obj2 -= coef_obj2
    max_obj2 += coef_obj2
    return min_obj1, max_obj1, min_obj2, max_obj2

def plot_legend_score(fig, ax2, palette, ls_1='-', ls_2='--', lw_a=1.5, lw_b=1,
                      fontsize=16, handle_color='#464646', label_color='#464646'):
    h, l = ax2.get_legend_handles_labels()
    h.insert(0, mpatches.Patch(color=palette[0], label='Training'))
    h.insert(1, mpatches.Patch(color=palette[1], label='Testing'))
    l = [r'Training', r'Testing', 'Error', r'R$^{2}$']
    h[2].set_linestyle(ls_1)
    h[3].set_linestyle(ls_2)
    h[2].set_linewidth(lw_a)
    h[3].set_linewidth(lw_b)
    h[2].set_color(handle_color)
    h[3].set_color(handle_color)
    leg = fig.legend(
        h, l, loc='upper center', bbox_to_anchor=(0.5, 1.0),
        fontsize=fontsize*0.75, framealpha=0.85, ncol=4, handletextpad=0.5,  # spacing between handle and label
        columnspacing=1.5, frameon=True, edgecolor=label_color)
    for text in leg.get_texts():
        text.set_color(label_color)
    ax2.legend().remove()
    return leg

def plot_pred_figure(fname_out, feat_n,
                     df_pred_list, df_score_list, model_list,
                     x_label='Predicted', y_label='Measured', y_col='nup_kgha',
                     units=None, save_plot=True,
                     fontsize=16, fontcolor='#464646', linecolor='#464646',
                     legend_cols=4):
    '''
    Builds an axes for every regression model, then adds them dynamically to
    the matplotlib figure to be saved

    feat_n = 9
    x_label = r'Predicted Vine N (ppm)'  # '(kg ha$^{-1}$)'
    y_label = r'Measured Vine N (ppm)'
    y_col = 'value'

    Parameters:
        units (``str``): The units to display in the plot's annotated error
            boxes (e.g., '%' will add the percent symbol after the error
            value).
    '''
    plt.style.use('seaborn-whitegrid')

    try:
        temp = df_pred_list[0][feat_n]
        temp = None
    except KeyError:
        feat_n = str(feat_n)
    if save_plot is True:
        fig, axes = plt.subplots(1, len(model_list), figsize=(len(model_list)*5, 5.5), sharex=True, sharey=True, dpi=300)
    else:
        fig, axes = plt.subplots(1, len(model_list), figsize=(len(model_list)*5, 5.5), sharex=True, sharey=True)
    max_plot = None
    min_plot = None
    for df_pred in df_pred_list:
        if max_plot is None:
            max_plot, min_plot = get_min_max(df_pred, feat_n, y_col)
        else:
            max_plot2, min_plot2 = get_min_max(df_pred, feat_n, y_col)
            max_plot = max_plot2 if max_plot2 > max_plot else max_plot
            min_plot = min_plot2 if min_plot2 < min_plot else min_plot
    BoxStyle._style_list['ext'] = ExtendedTextBox
    title_list = []
    _ = plot_titles(axes[0], 'Test')
    for idx, (ax, df_preds, df_scores) in enumerate(zip(axes, df_pred_list, df_score_list)):
        if df_preds[feat_n].isnull().values.any():
            continue
        legend = 'full' if idx == 0 else False
        ax, colors = plot_meas_pred(
            feat_n, y_col, df_preds, ax, x_label=x_label, y_label=y_label,
            units=units, max_plot=max_plot, min_plot=min_plot, legend=legend)
        if idx == 0:
            leg = plot_legend(
                fig, ax, df_preds, feat_n, colors, handle_color=fontcolor,
                label_color=fontcolor, ncol=legend_cols)
            fig = legend_resize(fig, leg)
        if model_list is None:
            t = plot_titles(ax, 'Model {0}'.format(idx))
        else:
            t = plot_titles(ax, str(model_list[idx]).split('(')[0])
        title_list.append(t)
        best_fit_line(
            df_preds, feat_n, y_col, ax, fontsize=fontsize*0.75,
            linecolor=linecolor, fontcolor=fontcolor, xytext=(0.95, 0.05),
            ha='right', va='bottom')
        prediction_error_label(
            df_scores, feat_n, ax, fontsize*0.75, fontcolor, xy=(0.04, 0.94),
            ha='left', va='top', units=units)
    cid = plt.gcf().canvas.mpl_connect(
        'resize_event', lambda event: on_resize(event, title_list, axes))
    if save_plot is True:
        fig.savefig(fname_out, dpi=300)
        plt.close(fig)
        fig.clf()
        return None
    else:
        return fig

def plot_score_figure(
        fname_out, df_score_list, model_list, y_label=None, units=None,
        save_plot=True, ls_1='-', ls_2='--', lw_a=1.5, lw_b=1, obj1='mae',
        obj2='r2', fontsize=16, fontcolor='#464646', linecolor='#464646'):
    '''
    Plot the error for all number of features
    '''
    palette = sns.color_palette("mako_r", 2)

    if save_plot is True:
        fig, axes = plt.subplots(1, len(model_list), figsize=(len(model_list)*5, 5.5), sharex=True, sharey=True, dpi=300)
    else:
        fig, axes = plt.subplots(1, len(model_list), figsize=(len(model_list)*5, 5.5), sharex=True, sharey=True)
    min_obj1, max_obj1, min_obj2, max_obj2 = min_max_scores_high(
        df_score_list, obj1, obj2)
    BoxStyle._style_list['ext'] = ExtendedTextBox
    title_list = []
    axes2 = []
    _ = plot_titles(axes[0], ' ')

    # idx=0
    # ax = axes[idx]
    # df_scores = df_score_list[idx]
    for idx, (ax, df_scores) in enumerate(zip(axes, df_score_list)):
        legend = 'full' if idx == 0 else False
        ax = plot_scores_feats(
            df_scores, ax, palette, legend=False, obj=obj1, y_label=y_label,
            units=units)
        ax, ax2 = plot_secondary(
            df_scores, ax, palette, legend=legend, obj=obj2)
        ax, ax2 = sync_axis_grids(ax, ax2, ax_min=min_obj1, ax_max=max_obj1,
                                  ax2_min=min_obj2, ax2_max=max_obj2)
        if idx == 0:
            ax.set_xlim([0, df_scores['feat_n'].max()])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.tick_params(labelright=False)
            ax2.yaxis.label.set_visible(False)
            leg = plot_legend_score(
                fig, ax2, palette, ls_1=ls_1, ls_2=ls_2, lw_a=lw_a, lw_b=lw_b,
                fontsize=fontsize, handle_color=fontcolor, label_color=fontcolor)
            fig = legend_resize(fig, leg, twinx=True)
        elif idx == len(axes)-1:
            ax.tick_params(labelleft=False)
            ax.yaxis.label.set_visible(False)
        if model_list == None:
            t = plot_titles(ax, 'Model {0}'.format(idx))
        else:
            t = plot_titles(ax, str(model_list[idx]).split('(')[0])
        title_list.append(t)
        axes2.append(ax2)
    cid = plt.gcf().canvas.mpl_connect(
        'resize_event', lambda event: on_resize(event, title_list, axes))
    if save_plot is True:
        fig.savefig(fname_out, dpi=300)
        plt.close(fig)
        fig.clf()
        return None
    else:
        return fig


def plot_pred_figure_pp(folder_list_test, dir_out_list_test, feat_n,
                        df_pred_list, df_score_list, model_list,
                        x_label='Predicted', y_label='Measured', y_col='nup_kgha',
                        units=None, save_plot=True,
                        fontsize=16, fontcolor='#464646', linecolor='#464646',
                        legend_cols=4):
    '''
    Builds an axes for every regression model, then adds them dynamically to
    the matplotlib figure to be saved

    feat_n = 9
    x_label = r'Predicted Vine N (ppm)'  # '(kg ha$^{-1}$)'
    y_label = r'Measured Vine N (ppm)'
    y_col = 'value'

    Parameters:
        units (``str``): The units to display in the plot's annotated error
            boxes (e.g., '%' will add the percent symbol after the error
            value).
    '''
    preds_name = '_'.join(
        ('preds', folder_list_test[1], folder_list_test[2],
        str(feat_n).zfill(3) + '-feats.png'))
    fname_out = os.path.join(
        dir_out_list_test[3], 'figures', preds_name)

    plt.style.use('seaborn-whitegrid')

    try:
        temp = df_pred_list[0][feat_n]
        temp = None
    except KeyError:
        feat_n = str(feat_n)
    if save_plot is True:
        fig, axes = plt.subplots(1, len(model_list), figsize=(len(model_list)*5, 5.5), sharex=True, sharey=True, dpi=300)
    else:
        fig, axes = plt.subplots(1, len(model_list), figsize=(len(model_list)*5, 5.5), sharex=True, sharey=True)
    max_plot = None
    min_plot = None
    for df_pred in df_pred_list:
        if max_plot is None:
            max_plot, min_plot = get_min_max(df_pred, feat_n, y_col)
        else:
            max_plot2, min_plot2 = get_min_max(df_pred, feat_n, y_col)
            max_plot = max_plot2 if max_plot2 > max_plot else max_plot
            min_plot = min_plot2 if min_plot2 < min_plot else min_plot
    BoxStyle._style_list['ext'] = ExtendedTextBox
    title_list = []
    _ = plot_titles(axes[0], 'Test')
    for idx, (ax, df_preds, df_scores) in enumerate(zip(axes, df_pred_list, df_score_list)):
        if df_preds[feat_n].isnull().values.any():
            continue
        legend = 'full' if idx == 0 else False
        ax, colors = plot_meas_pred(
            feat_n, y_col, df_preds, ax, x_label=x_label, y_label=y_label,
            units=units, max_plot=max_plot, min_plot=min_plot, legend=legend)
        if idx == 0:
            leg = plot_legend(
                fig, ax, df_preds, feat_n, colors, handle_color=fontcolor,
                label_color=fontcolor, ncol=legend_cols)
            fig = legend_resize(fig, leg)
        if model_list is None:
            t = plot_titles(ax, 'Model {0}'.format(idx))
        else:
            t = plot_titles(ax, str(model_list[idx]).split('(')[0])
        title_list.append(t)
        best_fit_line(
            df_preds, feat_n, y_col, ax, fontsize=fontsize*0.75,
            linecolor=linecolor, fontcolor=fontcolor, xytext=(0.95, 0.05),
            ha='right', va='bottom')
        prediction_error_label(
            df_scores, feat_n, ax, fontsize*0.75, fontcolor, xy=(0.04, 0.94),
            ha='left', va='top', units=units)
    cid = plt.gcf().canvas.mpl_connect(
        'resize_event', lambda event: on_resize(event, title_list, axes))
    if save_plot is True:
        fig.savefig(fname_out, dpi=300)
        plt.close(fig)
        fig.clf()
    else:
        return fig
