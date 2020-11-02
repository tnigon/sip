# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:15:33 2020

@author: nigo0024
"""

# In[Import libraries]
from sip_functions import *

import os

# In[Set defaults]
n_jobs = 4
msi_run_id = None
idx_min = 0
idx_max = 1

# In[Get arguments]
if __name__ == "__main__":  # required on Windows, so just do on all..
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_jobs',
                        help='Number of CPU cores to use.')
    parser.add_argument('-m', '--msi_run_id',
                        help='The MSI run ID; use 0 to run on local machine.')
    parser.add_argument('-i', '--idx_min',
                        help='Minimum idx to consider in df_grid')
    parser.add_argument('-d', '--idx_max',
                        help='Minimum idx to consider in df_grid')
    args = parser.parse_args()

    if args.n_jobs is not None:
        n_jobs = eval(args.n_jobs)
    else:
        n_jobs = 1
    if args.msi_run_id is not None:
        msi_run_id = eval(args.msi_run_id)
    else:
        msi_run_id = 0
    if args.idx_min is not None:
        idx_min = eval(args.idx_min)
    else:
        idx_min = 0
    if args.idx_max is not None:
        idx_max = eval(args.idx_max)
    else:
        idx_max = idx_min + 1

    # In[Prep I/O]
    if msi_run_id > 0:  # be sure to set keyrings
        dir_base = '/panfs/roc/groups/5/yangc1/public/hs_process'
    else:
        msi_run_id = 0
        dir_base = r'G:\BBE\AGROBOT\Shared Work\hs_process_results'

    dir_data = os.path.join(dir_base, 'data')
    dir_results = os.path.join(dir_base, 'results')
    dir_results_msi = os.path.join(dir_results, 'msi_' + str(msi_run_id) + '_results')
    if not os.path.isdir(dir_results_msi):
        os.mkdir(dir_results_msi)

    # In[Grid search settings]
    hs_settings = {
        'dir_panels': ['ref_closest_panel', 'ref_all_panels'],
        'crop': ['crop_plot', 'crop_buf'],
        'clip': [None,
                 {'wl_bands': [[0, 420], [880, 1000]]},
                 {'wl_bands': [[0, 420], [760, 776], [813, 827], [880, 1000]]}],
        'smooth': [None,
                   {'window_size': 5, 'order': 2},
                   {'window_size': 11, 'order': 2}],
        'segment': [None,
                   {'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550],
                    'mask_percentile': 50, 'mask_side': 'lower'},
                   {'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550],
                    'mask_percentile': 90, 'mask_side': 'lower'},
                   {'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550],
                    'mask_percentile': [50, 75], 'mask_side': 'outside'},
                   {'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550],
                    'mask_percentile': [75, 95], 'mask_side': 'outside'},
                   {'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550],
                    'mask_percentile': [95, 98], 'mask_side': 'outside'},
                   {'method': ['mcari2', [545, 565]], 'wl1': [[800], [None]],
                    'wl2': [[670], [None]], 'wl3': [[550], [None]],
                    'mask_percentile': [90, 75], 'mask_side': ['lower', 'lower']},
                   {'method': ['mcari2', [800, 820]], 'wl1': [[800], [None]],
                    'wl2': [[670], [None]], 'wl3': [[550], [None]],
                    'mask_percentile': [90, 75], 'mask_side': ['lower', 'lower']},
                    ]
        # 'bin': [None,
        #         {'bandwidth': 5},
        #         {'bandwidth': 10},
        #         {'bandwidth': 20},
        #         {'bandwidth': 50}]
        }
    df_grid = hs_grid_search(hs_settings, msi_run_id, dir_out=dir_results)
    time_dict = time_setup_img(dir_results, msi_run_id)
    proc_dict = proc_files_count_setup(dir_results, msi_run_id)

    # In[Process images]
    for idx_grid, row in df_grid.iterrows():
        # if idx_grid >= 1:
        #     break
        if idx_grid < idx_min:
            print('Skipping past idx_grix {0}...'.format(idx_grid))
            continue
        if idx_grid >= idx_max:
            sys.exit('All processing scenarios are finished. Exiting program.')
        print_details(row)

        time_dict, time_last = time_loop_init(time_dict, msi_run_id, row.name, n_jobs)
        if row['dir_panels'] == 'ref_closest_panel':
            n_files = 835
        else:
            n_files = 859
        time_dict, time_last = time_step(time_dict, 'crop', time_last)

        print('Clipping..')  # clips out certain spectral bands from the datacube
        clip_pp(dir_data, row, n_jobs, out_force=False, n_files=n_files)
        time_dict, time_last = time_step(time_dict, 'clip', time_last)
        print('Smoothing..')  # smoothes spectra for every pixel
        smooth_pp(dir_data, row, n_jobs, out_force=False, n_files=n_files)
        time_dict, time_last = time_step(time_dict, 'smooth', time_last)
        print('Segmenting..\n')  # each image has thousands of pixels; segmentation removes unwanted pixels before taking the mean spectra
        seg_pp(dir_data, row, n_jobs, out_force=False, n_files=n_files)
        time_dict, time_last = time_step(time_dict, 'segment', time_last)

        time_dict = append_times(dir_results, time_dict, msi_run_id)  # Saves timing

        # Count and save results
        proc_dict['grid_idx'] = [idx_grid]
        proc_dict, n_files_proc = proc_files_count(
            proc_dict, n_jobs, msi_run_id, 'processed', dir_data, row, ext='.spec')
        if n_files_proc != n_files:
            msg_n_files = ('grid_idx: {0}\nNumber of processed .spec files is not as '
                           'expected.\nProcessed: {1}\nExpected: {2}\n'
                           ''.format(idx_grid, n_files_proc, n_files))
            warnings.warn(msg_n_files, RuntimeWarning)
        proc_dict = proc_files_append(dir_results, proc_dict, msi_run_id)  # Saves file count
