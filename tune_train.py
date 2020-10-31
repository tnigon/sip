# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:19:13 2020

@author: nigo0024
"""

# In[Import libraries]
from sip_functions import *

import numpy as np
import os

import tracemalloc

from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

import globus_sdk

# In[Set defaults]
n_jobs = 4
msi_run_id = None
y_label = 'biomass_kgha'
idx_min = 0
idx_max = 1

CLIENT_ID = ''
TRANSFER_TOKEN = ''
TRANSFER_REFRESH_TOKEN = ''

# In[Tuning and training settings]
random_seed = 973717
max_iter = 100000
key = 'regressor__'
n_feats_linspace = 30  # maximum number of features to tune, train, and test for; starts at 1, and goes up to n_feats using n_linspace steps
n_steps_linspace = 300  # number of steps across the feature selection space; may want to adjust this dynamically based on max feats?
standardize = False
scoring = ('neg_mean_absolute_error', 'neg_mean_squared_error', 'r2')
scoring_refit = scoring[0]
n_splits = 4
n_repeats = 3
score = 'score_val_mae'

y_label_list = ['biomass_kgha', 'tissue_n_pct', 'nup_kgha']  # all these lists should have same length
label_print_list = ['Above-ground Biomass', r'Tissue Nitrogen', r'Nitrogen Uptake']  # Plot labels for each folder's dataset
units_list = ['kg ha$^{-1}$', '%', 'kg ha$^{-1}$']
legend_cols_list = [6, 6, 6]
start_alpha_list = [1e-4, 1e-4, 1e-4]
start_step_pct_list = [0.05, 0.05, 0.05]
extra_feats_list = [None, ['pctl_10th']]  # this can be looked at as a "features" list - in the simple case, I am using only spectral, or spectral + auxiliary features
extra_feats_names = ['spectral', 'aux_mcari2_pctl_10th']

# In[Get arguments]
if __name__ == "__main__":  # required on Windows, so just do on all..
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_jobs',
                        help='Number of CPU cores to use.')
    parser.add_argument('-m', '--msi_run_id',
                        help='The MSI run ID; use 0 to run on local machine.')
    parser.add_argument('-y', '--y_label',
                        help='Response variable to tune, train, etc.')
    parser.add_argument('-i', '--idx_min',
                        help='Minimum idx to consider in df_grid')
    parser.add_argument('-d', '--idx_max',
                        help='Minimum idx to consider in df_grid')
    parser.add_argument('-c', '--CLIENT_ID',
                        help='Globus client ID; can be found at https://auth.globus.org/v2/web/developers on the "MSI SIP" app')
    parser.add_argument('-t', '--TRANSFER_TOKEN',
                        help='Transfer access token.')
    parser.add_argument('-r', '--TRANSFER_REFRESH_TOKEN',
                        help='Transfer refresh token.')
    args = parser.parse_args()

    if args.msi_run_id is not None:
        msi_run_id = eval(args.msi_run_id)
    else:
        msi_run_id = 0
    if args.n_jobs is not None:
        n_jobs = eval(args.n_jobs)
    else:
        n_jobs = 1
    if args.y_label is not None:
        y_label = str(args.y_label)
    else:
        y_label = 'biomass_kgha'

    if args.idx_min is not None:
        idx_min = eval(args.idx_min)
    else:
        idx_min = 0
    if args.idx_max is not None:
        idx_max = eval(args.idx_max)
    else:
        idx_max = idx_min + 1

    if args.CLIENT_ID is not None:
        CLIENT_ID = args.CLIENT_ID
    else:
        CLIENT_ID = ''
    if args.TRANSFER_TOKEN is not None:
        TRANSFER_TOKEN = args.TRANSFER_TOKEN
    else:
        TRANSFER_TOKEN = ''
    if args.TRANSFER_REFRESH_TOKEN is not None:
        TRANSFER_REFRESH_TOKEN = args.TRANSFER_REFRESH_TOKEN
    else:
        TRANSFER_REFRESH_TOKEN = ''

    assert y_label in y_label_list, ('{0} must be one of {1}'
                                     ''.format(y_label, y_label_list))
    y_idx = y_label_list.index(y_label)


    # In[Prep I/O]
    if msi_run_id > 0:  # be sure to set keyrings
        dir_base = '/panfs/roc/groups/5/yangc1/public/hs_process'
    else:
        msi_run_id = 0
        dir_base = r'G:\BBE\AGROBOT\Shared Work\hs_process_results'

    dir_data = os.path.join(dir_base, 'data')
    dir_results = os.path.join(dir_base, 'results')
    dir_results_msi = os.path.join(dir_results, 'msi_' + str(msi_run_id) + '_results')

    df_grid = pd.read_csv(os.path.join(dir_results, 'msi_' + str(msi_run_id) + '_hs_settings.csv'), index_col=0)
    n_clip, n_smooth, n_segment = grid_n_levels(df_grid)
    df_grid = clean_df_grid(df_grid)


    # In[Tuning parameter grid settings]
    param_grid_las = {'alpha': list(np.logspace(-4, 3, 8))}
    param_grid_svr = None
    # param_grid_rf = {'n_estimators': [300], 'min_samples_split': list(np.linspace(2, 10, 5, dtype=int)), 'max_features': [0.05, 0.1, 0.3, 0.9]}
    param_grid_rf = None
    param_grid_pls = {'n_components': list(np.linspace(2, 10, 9, dtype=int)), 'scale': [True, False]}
    param_grid_dict = {
        'las': param_grid_las,
        'pls':param_grid_pls}


    model_las = Lasso(max_iter=max_iter, selection='cyclic')
    # model_svr = SVR(tol=1e-2)
    # model_rf = RandomForestRegressor(bootstrap=True)
    model_pls = PLSRegression(tol=1e-9)
    model_list = (model_las, model_pls)  # Models to evaluate

    time_dict = time_setup_training(dir_results, msi_run_id)

    # In[Main loop]
    for idx_grid, row in df_grid.iterrows():
        # if idx_grid >= 0:
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

        # idx_grid = get_idx_grid(dir_results_msi, msi_run_id, idx_min=33)


        # Tuning loop
        # TODO: Check to see if all tuning, training, etc. is already complete..?
        # Load data just processed
        df_spec, meta_bands, bands = load_spec_data(dir_data, row)
        time_dict, time_last = time_step(time_dict, 'init1', time_last)
        df_bm_stats = load_preseg_stats(dir_data, row, bm_folder_name='bm_mcari2')
        time_dict, time_last = time_step(time_dict, 'init2', time_last)
        df_ground = load_ground_data(dir_data)
        df_join_base = join_ground_bm_spec(df_ground, df_bm_stats, df_spec)
        create_readme(dir_results_msi, msi_run_id, row)
        random_seed = get_random_seed(dir_results_msi, msi_run_id, row, seed=random_seed)
        time_dict, time_last = time_step(time_dict, 'init3', time_last)

        tracemalloc.start()

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
            if time_dict['time_start'] == [None]:
                time_dict, time_last = time_loop_init(time_dict, msi_run_id, row.name, n_jobs)
                time_dict, time_last = time_step(time_dict, 'init1', time_last)
                time_dict, time_last = time_step(time_dict, 'init2', time_last)
                time_dict, time_last = time_step(time_dict, 'init3', time_last)
            # make the rest of this a single function to parallelize the tuning/training/testing!
            print('Feature set: {0}\n'.format(extra_name))
            time_dict['y_label'] = y_label
            time_dict['feats'] = extra_name
            X1, y1 = get_X_and_y(df_train, meta_bands, y_label, random_seed, extra=extra_feats)
            df_tune_all_list = (None,) * len(model_list)

            logspace_list_full, start_alpha, start_step_pct = build_feat_selection_df(
                X1, y1, max_iter, random_seed, n_feats=n_feats_linspace,
                n_linspace=n_steps_linspace, method_alpha_min='full',
                alpha_init=start_alpha_list[y_idx],
                step_pct=start_step_pct_list[y_idx])  # method can be "full" or "convergence_warning"
            start_alpha_list[y_idx] = start_alpha * 5  # remember for future loops
            start_step_pct_list[y_idx] = start_step_pct  # keep this the same
            logspace_list = filter_logspace_list_pp(
                logspace_list_full, X1, y1, max_iter, random_seed, n_jobs)
            time_dict, time_last = time_step(time_dict, 'feat_sel', time_last)

            print('Hyperparameter tuning...\n')
            n_jobs_tuning = n_jobs
            df_tune_all_list = execute_tuning_pp(
                logspace_list, X1, y1, model_list, param_grid_dict,
                standardize, scoring, scoring_refit, max_iter, random_seed,
                key, df_train, n_splits, n_repeats, df_tune_all_list, n_jobs_tuning)
            time_dict, time_last = time_step(time_dict, 'tune1', time_last)

            df_tune_list = filter_tuning_results(df_tune_all_list, score)
            time_dict, time_last = time_step(time_dict, 'tune2', time_last)
            df_params = summarize_tuning_results(
                df_tune_list, model_list, param_grid_dict, key)
            time_dict, time_last = time_step(time_dict, 'tune3', time_last)
            df_params_mode, df_params_mode_count = tuning_mode_count(df_params)
            dir_out_list_tune, folder_list_tune = set_up_output_dir(
                dir_results_msi, msi_run_id, row.name, y_label, extra_name,
                test_or_tune='tuning')
            fname_base_tune = '_'.join((folder_list_tune[0], folder_list_tune[1]))
            save_tuning_results(
                    dir_out_list_tune[3], df_tune_list, model_list, df_params,
                    df_params_mode, df_params_mode_count, meta_bands,
                    fname_base_tune)
            time_dict, time_last = time_step(time_dict, 'tune4', time_last)
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
            time_dict, time_last = time_step(time_dict, 'test', time_last)

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
            # time_label = ('sttp-' + y_label + '-' + extra_name)
            time_dict, time_last = time_step(time_dict, 'plot', time_last)

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            print("[ Top 3 ]")
            for stat in top_stats[:3]:
                print(stat)

            time_dict = append_times(dir_results, time_dict, msi_run_id)

        label_base = 'idx_grid_' + str(idx_grid).zfill(3)

        client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
        # dir_source_data, dir_dest_data = get_globus_data_dir(
        #     dir_base, msi_run_id, row)
        # transfer_result, delete_result = globus_transfer(
        #     dir_source_data, dir_dest_data, TRANSFER_REFRESH_TOKEN, client, TRANSFER_TOKEN,
        #     label=label_base + '-data', delete=True)

        dir_source_results, dir_dest_results = get_globus_results_dir(
            dir_base, msi_run_id, row)
        transfer_result, delete_result = globus_transfer(
            dir_source_results, dir_dest_results, TRANSFER_REFRESH_TOKEN, client,
            TRANSFER_TOKEN, label=label_base + '-results', delete=True)