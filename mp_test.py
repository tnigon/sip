# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:53:14 2020

@author: nigo0024
"""

# In[Hypespectral subjective post-processing]
from sip_functions import *

import numpy as np
import os
import platform

from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

msi_run_id = 0

# dir_base = 'yangc1/public/hs_process'
dir_base = r'G:\BBE\AGROBOT\Shared Work\hs_process_results'

# Tuning settings
random_seed = 973717
max_iter = 100000
key = 'regressor__'
n_feats_linspace = 40  # maximum number of features to tune, train, and test for; starts at 1, and goes up to n_feats using n_linspace steps
n_steps_linspace = 100  # number of steps across the feature selection space; may want to adjust this dynamically based on max feats?
standardize = False
scoring = ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2')
scoring_refit = scoring[0]
n_splits = 4
n_repeats = 3
score = 'score_val_mae'

# Model training settings
y_label_list = ['biomass_kgha', 'tissue_n_pct', 'nup_kgha']  # all these lists should have same length
label_print_list = ['Above-ground Biomass', r'Tissue Nitrogen', r'Nitrogen Uptake']  # Plot labels for each folder's dataset
units_list = ['kg ha$^{-1}$', '%', 'kg ha$^{-1}$']
legend_cols_list = [6, 6, 6]
start_alpha_list = [1e-4, 1e-4, 1e-4]
start_step_pct_list = [0.05, 0.05, 0.05]
# extra_feats must match with column heading from df_join_base
extra_feats_list = [None, ['pctl_10th']]  # this can be looked at as a "features" list - in the simple case, I am using only spectral, or spectral + auxiliary features
extra_feats_names = ['spectral', 'aux_mcari2_pctl_10th']

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

# Tuning parameter grid settings
param_grid_las = {'alpha': list(np.logspace(-4, 0, 5))}
param_grid_svr = None
# param_grid_rf = {'n_estimators': [300], 'min_samples_split': list(np.linspace(2, 10, 5, dtype=int)), 'max_features': [0.05, 0.1, 0.3, 0.9]}
param_grid_rf = None
param_grid_pls = {'n_components': list(np.linspace(2, 10, 9, dtype=int)), 'scale': [True, False]}
param_grid_dict = {
    'las': param_grid_las,
    'pls':param_grid_pls}

# Models to evaluate
model_las = Lasso(max_iter=max_iter, selection='cyclic')
# model_svr = SVR(tol=1e-2)
# model_rf = RandomForestRegressor(bootstrap=True)
model_pls = PLSRegression(tol=1e-9)

model_list = (model_las, model_pls)

# In[Full loop]
# Prep directories
dir_data = os.path.join(dir_base, 'data')
dir_results = os.path.join(dir_base, 'results')
time_dict = set_up_time_files(dir_results, y_label_list, extra_feats_names, msi_run_id)
df_grid = hs_grid_search(hs_settings, msi_run_id, dir_out=dir_results)
dir_results_msi = os.path.join(dir_results, 'msi_' + str(msi_run_id) + '_results')
if not os.path.isdir(dir_results_msi):
    os.mkdir(dir_results_msi)

idx_grid = get_idx_grid(dir_results_msi, msi_run_id, idx_min=32)
if idx_grid >= 34:
# if idx_grid >= len(df_grid):
    sys.exit('All processing scenarios are finished. Exiting program.')
row = df_grid.loc[idx_grid]

print('Processing scenario number: {0}\n'.format(idx_grid))

# for idx_grid, row in df_grid.iterrows():
#     # if idx_grid < 125:
#     #     continue
#     if idx_grid == 126:
#         break

# Image processing
time_dict, time_last = time_loop_init(time_dict, msi_run_id, row.name)
if row['dir_panels'] == 'ref_closest_panel':
    n_files = 833
else:
    n_files = 859
time_dict, time_last = time_step(time_dict, 'crop', time_last)

if __name__ == "__main__":  # required on Windows, so just do on all..
    print('Clipping..')  # clips out certain spectral bands from the datacube
    clip_pp(dir_data, row, out_force=False, n_files=n_files)
    # clip(dir_data, row, out_force=False, n_files=n_files)
    time_dict, time_last = time_step(time_dict, 'clip', time_last)
    print('Smoothing..')  # smoothes spectra for every pixel
    smooth_pp(dir_data, row, out_force=False, n_files=n_files)
    # smooth(dir_data, row, out_force=False, n_files=n_files)
    time_dict, time_last = time_step(time_dict, 'smooth', time_last)
    print('Segmenting..\n')  # each image has thousands of pixels; segmentation removes unwanted pixels before taking the mean spectra
    seg_pp(dir_data, row, out_force=False, n_files=n_files)
    # seg(dir_data, row, out_force=False, n_files=n_files)
    time_dict, time_last = time_step(time_dict, 'segment', time_last)

# # clip(dir_data, row, out_force=False, n_files=n_files)
# if platform.system() == 'Windows':
#     if __name__ == "__main__":  # required on Windows, so just do on all..
#         print('Clipping..')  # clips out certain spectral bands from the datacube
#         clip_pp(dir_data, row, out_force=False, n_files=n_files)
#         time_dict, time_last = time_step(time_dict, 'clip', time_last)
#         print('Smoothing..')  # smoothes spectra for every pixel
#         time_dict, time_last = time_step(time_dict, 'smooth', time_last)
#         smooth_pp(dir_data, row, out_force=False, n_files=n_files)
# else:
#     print('Clipping..')  # clips out certain spectral bands from the datacube
#     clip_pp(dir_data, row, out_force=False, n_files=n_files)
#     time_dict, time_last = time_step(time_dict, 'clip', time_last)
#     print('Smoothing..')  # smoothes spectra for every pixel
#     time_dict, time_last = time_step(time_dict, 'smooth', time_last)
#     smooth_pp(dir_data, row, out_force=False, n_files=n_files)

