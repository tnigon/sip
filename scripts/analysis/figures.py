# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 08:23:04 2020

@author: nigo0024
"""
# %% Export Readable hs_settings/df_grid
import os
import pandas as pd
from scripts.analysis import sip_functs_analysis as sip_f

base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'
df_grid = pd.read_csv(os.path.join(base_dir_results, 'msi_2_hs_settings.csv'))
df_grid_readable = sip_f.rename_scenarios(df_grid)
df_grid_readable.to_csv(os.path.join(base_dir_results, 'msi_2_hs_settings_short.csv'), index=False)

# %% Violin plots1
import os
import pandas as pd
from scripts.analysis import sip_functs_analysis as sip_f

base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'
base_dir_out = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_metafigures'
fnames = ['msi_2_biomass_kgha_R2.csv', 'msi_2_nup_kgha_R2.csv', 'msi_2_tissue_n_pct_R2.csv']

df_grid = pd.read_csv(os.path.join(base_dir_results, 'msi_2_hs_settings.csv'))
df_grid = sip_f.rename_scenarios(df_grid)
f_full = [os.path.join(base_dir_results, f) for f in fnames]
for f in f_full:
    df = pd.read_csv(f)
    df = df_grid.merge(df, left_index=True, right_on='grid_idx')
    df_filter = sip_f.sip_results_filter(df, model='Lasso')
    for n_feats in range(1, 20, 3):
        base_name = os.path.splitext(os.path.split(f)[-1])[0]
        fig = sip_f.plot_violin_by_scenario(df_filter, base_name, y=str(n_feats))
        fig.savefig(os.path.join(base_dir_out, '{0}_{1}.png'.format(os.path.splitext(os.path.split(f)[-1])[0], n_feats)), dpi=300)

# %% Find optimum accuracy
import os
import pandas as pd
from scripts.analysis import sip_functs_analysis as sip_f

base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'
base_dir_out = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_metafigures'
fnames = ['msi_2_biomass_kgha_R2.csv', 'msi_2_nup_kgha_R2.csv', 'msi_2_tissue_n_pct_R2.csv']

df_grid = pd.read_csv(os.path.join(base_dir_results, 'msi_2_hs_settings.csv'))
df_grid = sip_f.rename_scenarios(df_grid)
f_full = [os.path.join(base_dir_results, f) for f in fnames]


subset = ['msi_run_id', 'grid_idx', 'response_label', 'feature_set',
          'model_name', 'objective_f', 'n_feats_opt', 'value']
sort_order = ['response_label', 'feature_set', 'model_name', 'objective_f',
              'grid_idx']
df_out = None
for response in ['biomass_kgha', 'nup_kgha', 'tissue_n_pct']:
    for objective_f in ['R2', 'MAE', 'RMSE']:
        f = os.path.join(base_dir_results, 'msi_2_{0}_{1}.csv'.format(response, objective_f))
        df = pd.read_csv(f)
        df.rename(columns={'extra_feats': 'feature_set'}, inplace=True)
        df['objective_f'] = objective_f
        if objective_f in ['MAE', 'RMSE']:
            df['n_feats_opt'] = df[map(str, range(1, 51))].idxmin(axis=1)
            df['value'] = df[map(str, range(1, 51))].min(axis=1)
        else:
            df['n_feats_opt'] = df[map(str, range(1, 51))].idxmax(axis=1)
            df['value'] = df[map(str, range(1, 51))].max(axis=1)
        df = df.astype({'n_feats_opt': 'int'})

        df_out_temp = df_grid.merge(df[subset], left_index=True, on='grid_idx')
        if df_out is None:
            df_out = df_out_temp.copy()
        else:
            df_out = df_out.append(df_out_temp)
df_out.sort_values(by=sort_order, inplace=True)
df_out.to_csv(os.path.join(base_dir_results, 'msi_2_n_feats_opt.csv'), index=False)

# %% Violin plots Opt
import os
import pandas as pd
import matplotlib.pyplot as plt

from scripts.analysis import sip_functs_analysis as sip_f

base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'
base_dir_out = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_metafigures'

df_opt = pd.read_csv(os.path.join(base_dir_results, 'msi_2_n_feats_opt.csv'))
for response in ['biomass_kgha', 'nup_kgha', 'tissue_n_pct']:
    for feature_set in ['reflectance', 'derivative_1', 'derivative_2']:
        # for model in ['Lasso', 'PLSRegression']:
        for objective_f in ['R2', 'MAE', 'RMSE']:
            df_opt_filter = sip_f.sip_n_feats_obj_filter(
                df_opt, response, feature_set, objective_f)
            name = 'msi_2_{0}_{1}_{2}'.format(response, feature_set, objective_f)
            fig1 = sip_f.plot_violin_by_scenario_opt(df_opt_filter, name + '-val', y='value')
            fig2 = sip_f.plot_violin_by_scenario_opt(df_opt_filter, name + '-feats-opt', y='n_feats_opt')
            fig1.savefig(os.path.join(base_dir_out, response, 'values', '{0}-val.png'.format(name)), dpi=300)
            fig2.savefig(os.path.join(base_dir_out, response, 'feats_opt', '{0}-feats-opt.png'.format(name)), dpi=300)
            plt.close(fig1)
            plt.close(fig2)


