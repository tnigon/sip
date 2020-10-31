# -*- coding: utf-8 -*-
"""
Test 2 msi_1
"""

# In[Hypespectral subjective post-processing]
from sip_functions import *

import numpy as np
import os

from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

msi_run_id = 1

dir_base = '/panfs/roc/groups/5/yangc1/public/hs_process'
# dir_base = r'G:\BBE\AGROBOT\Shared Work\hs_process_results'

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

idx_grid = get_idx_grid(dir_results_msi, msi_run_id, idx_min=24)
if idx_grid >= 26:
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

# parallelize each processing step
print('Clipping..')  # clips out certain spectral bands from the datacube
# clip creates a folder according the to the "clip_type" (determined via get_clip_type() function)
clip(dir_data, row, out_force=False, n_files=n_files)
time_dict, time_last = time_step(time_dict, 'clip', time_last)

print('Smoothing..')  # smoothes spectra for every pixel
# smooth creates a folder according the to the "smooth_type"
smooth(dir_data, row, out_force=False, n_files=n_files)
time_dict, time_last = time_step(time_dict, 'smooth', time_last)
print('Segmenting..\n')  # each image has thousands of pixels; segmentation removes unwanted pixels before taking the mean spectra
# segmentation creates a folder according the to the "segment_type"
# during segmentation, band math images (MCARI2, NIR, etc.) must be created
# these files are created in their own folder alongside the "segment_type" folder in the "smooth_type" directory
seg(dir_data, row, out_force=False, n_files=n_files)
time_dict, time_last = time_step(time_dict, 'segment', time_last)
# if idx_grid > 13:  # for testing
#     break

# Tuning loop
# TODO: Check to see if all tuning, training, etc. is already complete..?
# Load data just processed
df_spec, meta_bands, bands = load_spec_data(dir_data, row)
df_bm_stats = load_preseg_stats(dir_data, row, bm_folder_name='bm_mcari2')
df_ground = load_ground_data(dir_data)
df_join_base = join_ground_bm_spec(df_ground, df_bm_stats, df_spec)
create_readme(dir_results_msi, msi_run_id, row)
random_seed = get_random_seed(dir_results_msi, msi_run_id, row, seed=random_seed)
time_dict, time_last = time_step(time_dict, 'sttp-init', time_last)

for y_idx, y_label in enumerate(y_label_list):
    # if y_label not in ['nup_kgha']:
    #     continue
    print('\nResponse variable: {0}\n'.format(y_label))
    # y_label = 'nup_kgha'
    df_join = df_join_base.dropna(subset=[y_label], how='all')
    save_joined_df(dir_results_msi, df_join, msi_run_id, row.name, y_label)
    df_train, df_test = split_train_test(
        df_join, random_seed=random_seed, stratify=df_join['dataset_id'])
    cv_rep_strat = get_repeated_stratified_kfold(
        df_train, n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed)
    # check_stratified_proportions(df_train, cv_rep_strat)
    for extra_feats, extra_name in zip(extra_feats_list, extra_feats_names):
        # make the rest of this a single function to parallelize the tuning/training/testing!
        #
        print('Feature set: {0}\n'.format(extra_name))
        X1, y1 = get_X_and_y(df_train, meta_bands, y_label, random_seed, extra=extra_feats)
        df_tune_all_list = (None,) * len(model_list)

        logspace_list, start_alpha, start_step_pct = build_feat_selection_df(
            X1, y1, max_iter, random_seed, n_feats=n_feats_linspace,
            n_linspace=n_steps_linspace, method_alpha_min='full',
            alpha_init=start_alpha_list[y_idx],
            step_pct=start_step_pct_list[y_idx])  # method can be "full" or "convergence_warning"
        start_alpha_list[y_idx] = start_alpha * 5  # remember for future loops
        start_step_pct_list[y_idx] = start_step_pct  # keep this the same
        # end_time = datetime.now()
        # print("Time to get logspace_list: ", end_time - start_time)
        # continue
        print('Hyperparameter tuning...\n')
        for idx_alpha, alpha in enumerate(reversed(logspace_list)):
            # print(idx_alpha, alpha)
            # idx_alpha = 137
            # alpha = 0.00033354627296573643
            df_tune_feat_list = execute_tuning(
                X1, y1, model_list, param_grid_dict,
                alpha, idx_alpha, standardize, scoring, scoring_refit,
                max_iter, random_seed, key, df_train, n_splits, n_repeats)
            df_tune_all_list = append_tuning_results(
                df_tune_all_list, df_tune_feat_list)

        df_tune_list = filter_tuning_results(df_tune_all_list, score)
        df_params = summarize_tuning_results(
            df_tune_list, model_list, param_grid_dict, key)
        df_params_mode, df_params_mode_count = tuning_mode_count(df_params)
        dir_out_list_tune, folder_list_tune = set_up_output_dir(
            dir_results_msi, msi_run_id, row.name, y_label, extra_name,
            test_or_tune='tuning')
        fname_base_tune = '_'.join((folder_list_tune[0], folder_list_tune[1]))
        save_tuning_results(
                dir_out_list_tune[3], df_tune_list, model_list, df_params,
                df_params_mode, df_params_mode_count, meta_bands,
                fname_base_tune)
        fname_feats_readme = os.path.join(dir_out_list_tune[2], folder_list_tune[2] + '_README.txt')
        fname_data = os.path.join(dir_out_list_tune[1], fname_base_tune + '_data.csv')
        feats_readme(fname_feats_readme, fname_data, meta_bands, extra_feats)

        # Testing
        print('Testing...\n')
        X1_test, y1_test = get_X_and_y(
            df_test, meta_bands, y_label, random_seed, extra=extra_feats)
        feat_n_list =list(df_tune_list[0]['feat_n'])  # should be same for all models

        df_pred_list, df_score_list = test_predictions(
            df_test, X1, y1, X1_test, y1_test, model_list, df_tune_list,
            feat_n_list, y_label, max_iter, standardize, key,
            n_feats_linspace)

        dir_out_list_test, folder_list_test = set_up_output_dir(
            dir_results_msi, msi_run_id, row.name, y_label, extra_name,
            test_or_tune='testing')
        set_up_summary_files(dir_results, y_label, n_feats_linspace, msi_run_id)  # initialize summary files
        fname_base_test = '_'.join((folder_list_test[0], folder_list_test[1]))
        metadata = [msi_run_id, row.name, y_label, extra_feats]
        append_test_scores(dir_results, y_label, df_score_list, model_list, metadata)
        save_test_results(dir_out_list_test[3], df_pred_list, df_score_list,
                          model_list, fname_base_test)

        label_print = label_print_list[y_idx]
        units = units_list[y_idx]
        legend_cols = legend_cols_list[y_idx]
        x_label = 'Predicted {0}'.format(label_print)
        y_label1 = 'Measured {0}'.format(label_print)
        y_label2 = '{0} MAE'.format(label_print)
        for idx, df_score in enumerate(df_score_list):
            for feat_n in df_score[pd.notnull(df_score['feats'])]['feat_n']:
                preds_name = '_'.join(
                    ('preds', folder_list_test[1], folder_list_test[2],
                    str(feat_n).zfill(3) + '-feats.png'))
                fname_out_fig1 = os.path.join(
                    dir_out_list_test[3], 'figures', preds_name)
                fig1 = plot_pred_figure(
                    fname_out_fig1, feat_n, df_pred_list, df_score_list,
                    model_list, x_label, y_label=y_label1, y_col=y_label,
                    units=units, save_plot=True, legend_cols=legend_cols)
        score_name = '_'.join(
            ('scores', folder_list_test[1], folder_list_test[2] + '.png'))
        fname_out_fig2 = os.path.join(
            dir_out_list_test[3], 'figures', score_name)
        fig2 = plot_score_figure(
            fname_out_fig2, df_score_list, model_list, y_label=y_label2,
            units=units, obj1='mae', obj2='r2', save_plot=True)
        time_label = ('sttp-' + y_label + '-' + extra_name)
        time_dict, time_last = time_step(time_dict, time_label, time_last)

time_dict = append_times(dir_results, time_dict, msi_run_id)
if msi_run_id > 0:  # be sure msi_id is adjusted when actually running on msi
    tier2_data_transfer(base_dir, row)
    tier2_results_transfer(base_dir, folder_list_test[0])  # msi_result_dir

# AFter 10 loops, I am running into a memoryerror - not sure why this is?
# restart_script()  # try restarting script after every main loop?

# del (fig1, fig2, df_pred_list, df_score_list, df_tune_feat_list,
#      df_tune_all_list, df_tune_list, folder_list_test, cv_rep_strat,
#      X1, X1_test, y1, y1_test, df_spec,
#      meta_bands, bands, df_bm_stats, df_ground, df_join_base, df_join)

# Load tuning results
# df_tune_list, df_params, df_params_mode, df_params_mode_count =\
#     load_tuning_results(dir_out_list_tune[3], model_list, fname_base)

# Load testing results
# df_pred_list, df_score_list = load_test_results(
#     dir_out_list_test[3], model_list, fname_base)

# # In[Remove from loop]
# '''
# Removing cropping from loop because all cropped images are uploaded
# directly to MSI. There are only four groups of cropped images, so there is no
# need to calculate them on MSI
# '''
#     dir_out_crop = os.path.join(dir_data, row['dir_panels'], row['crop'])
#     df_crop, gdf_wells, gdf_aerf = retrieve_files(
#         dir_data, panel_type=row['dir_panels'], crop_type=row['crop'])
#     print('Cropping..')
#     crop(df_crop, panel_type=row['dir_panels'], dir_out_crop=dir_out_crop,
#           out_force=False, n_files=n_files, gdf_aerf=gdf_aerf, gdf_wells=gdf_wells)
# # In[Sample read image]
# from hs_process import hsio

# fname = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\ref_closest_panel\crop_plot\plot_1_2_pika_gige_1_study_aerffield_date_20190723_plot_1_112-crop-plot.bip'
# fname = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\ref_closest_panel\crop_buf\study_aerffield_date_20190629_plot_1112-crop-buf.bip'
# io = hsio(fname)
# io.name_plot
# my_segment = segment(io.spyfile)

# # In[Calculate raw image size]
# df_grid = hs_grid_search(hs_settings, msi_run_id, dir_out_base)
# row_closest = df_grid.iloc[0]
# dir_results = os.path.join(dir_out_base, 'run_' + str(row_closest.name))
# dir_out_crop = os.path.join(dir_out_base, row_closest['dir_panels'], row_closest['crop'])
# df_crop_closest, _, _ = retrieve_files(data_dir, panel_type=row_closest['dir_panels'], crop_type=row_closest['crop'])
# row_all = df_grid.iloc[126]
# dir_results = os.path.join(dir_out_base, 'run_' + str(row_all.name))
# dir_out_crop = os.path.join(dir_out_base, row_all['dir_panels'], row_all['crop'])
# df_crop_all, _, _ = retrieve_files(data_dir, panel_type=row_all['dir_panels'], crop_type=row_all['crop'])

# unique_datacubes(df_crop_closest)
# unique_datacubes(df_crop_all)
