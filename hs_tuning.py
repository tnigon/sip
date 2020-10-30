# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:29:25 2020

@author: nigo0024
"""
def model_tuning_pp(time_dict, time_last, y_label_list):


def model_tuning_f_pp(
        y_label, df_join_base, dir_results_msi, msi_run_id, row, random_seed,
        n_splits, n_repeats, ):
    print('\nResponse variable: {0}\n'.format(y_label))
    # y_label = 'nup_kgha'
    df_join = df_join_base.dropna(subset=[y_label], how='all')
    save_joined_df(dir_results_msi, df_join, msi_run_id, row.name, y_label)
    df_train, df_test = split_train_test(
        df_join, random_seed=random_seed, stratify=df_join['dataset_id'])
    cv_rep_strat = get_repeated_stratified_kfold(
        df_train, n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed)

for y_idx, y_label in enumerate(y_label_list):
    # # if y_label not in ['nup_kgha']:
    # #     continue
    # print('\nResponse variable: {0}\n'.format(y_label))
    # # y_label = 'nup_kgha'
    # df_join = df_join_base.dropna(subset=[y_label], how='all')
    # save_joined_df(dir_results_msi, df_join, msi_run_id, row.name, y_label)
    # df_train, df_test = split_train_test(
    #     df_join, random_seed=random_seed, stratify=df_join['dataset_id'])
    # cv_rep_strat = get_repeated_stratified_kfold(
    #     df_train, n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed)
    # # check_stratified_proportions(df_train, cv_rep_strat)
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