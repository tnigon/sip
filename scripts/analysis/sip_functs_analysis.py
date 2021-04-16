# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:42:15 2020

@author: nigo0024
"""
# %% Get number of missing scenarios
# import os
# import pandas as pd

# base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'

# df_time = pd.read_csv(os.path.join(base_dir_results, 'msi_2_time_train.csv'))
# df_count = df_time.groupby(['grid_idx'])['y_label', 'feats'].count()
# df_missing = df_count[(df_count['y_label'] != 9) & (df_count['feats'] != 9)]

# # This tells me that there are 223 scenarios that did not complete all testing
# print(len(df_missing))

# %% Violin plots
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def sip_results_filter(df, model='Lasso'):
    '''
    Filters SIP results by <model>.
    '''
    df_filter = df[(pd.notnull(df['extra_feats'])) &
                   (df['model_name'] == model)]
    df_filter = df_filter.astype({'clip': 'str'})
    df_filter = df_filter.astype({'smooth': 'str'})
    df_filter = df_filter.astype({'bin': 'str'})
    df_filter = df_filter.astype({'segment': 'str'})
    return df_filter

def rename_scenarios(df):
    '''
    Renames scenarios from "hs_settings" (df_grid) names.
    '''
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'grid_idx'}, inplace=True)
    df['dir_panels'].replace({
        'ref_closest_panel': 'closest',
        'ref_all_panels': 'all'}, inplace=True)
    df['crop'].replace({
        'crop_plot': 'plot_bounds',
        'crop_all': 'buffer'}, inplace=True)
    df['clip'].replace({
        np.nan: 'none',
        "{'wl_bands': [[0, 420], [880, 1000]]}": 'ends',
        "{'wl_bands': [[0, 420], [760, 776], [813, 827], [880, 1000]]}": 'all'}, inplace=True)
    df['smooth'].replace({
        np.nan: 'none',
        # "{'window_size': 5, 'order': 2}": 'SG-5',
        "{'window_size': 11, 'order': 2}": 'sg-11'}, inplace=True)
    df['bin'].replace({
        np.nan: 'none',
        "{'method': 'spectral_mimic', 'sensor': 'sentinel-2a'}": 'sentinel-2a_mimic',
        "{'method': 'spectral_resample', 'bandwidth': 20}": 'bin_20nm',
        }, inplace=True)
    df['segment'].replace({
        np.nan: 'none',
        "{'method': 'ndi', 'wl1': [770, 800], 'wl2': [650, 680], 'mask_percentile': 50, 'mask_side': 'lower'}": 'ndi_upper_50',
        "{'method': 'ndi', 'wl1': [770, 800], 'wl2': [650, 680], 'mask_percentile': 50, 'mask_side': 'upper'}": 'ndi_lower_50',
        "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': 50, 'mask_side': 'lower'}": 'mcari2_upper_50',
        "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': 50, 'mask_side': 'upper'}": 'mcari2_lower_50',
        "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': 90, 'mask_side': 'lower'}": 'mcari2_upper_90',
        "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': [50, 75], 'mask_side': 'outside'}": 'mcari2_in_50-75',
        "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': [75, 95], 'mask_side': 'outside'}": 'mcari2_in_75-95',
        # "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': [95, 98], 'mask_side': 'outside'}": 'MCARI2_in_95-98',
        "{'method': ['mcari2', [545, 565]], 'wl1': [[800], [None]], 'wl2': [[670], [None]], 'wl3': [[550], [None]], 'mask_percentile': [90, 75], 'mask_side': ['lower', 'lower']}": 'mcari2_upper_90_green_upper_75'
        # "{'method': ['mcari2', [800, 820]], 'wl1': [[800], [None]], 'wl2': [[670], [None]], 'wl3': [[550], [None]], 'mask_percentile': [90, 75], 'mask_side': ['lower', 'lower']}": 'MCARI2_upper_90_nir_upper_75'},
        }, inplace=True)
    return df

def plot_violin_by_scenario(df_filter, base_name, y='2'):
    '''
    Makes violin plot for each processing scenario
    '''
    fig, axes = plt.subplots(ncols=6, sharey=True, figsize=(14, 8),
                             gridspec_kw={'width_ratios': [2, 2, 3, 2, 3, 9]})

    axes[0] = sns.violinplot(x='dir_panels', y=y, data=df_filter, ax=axes[0])
    axes[1] = sns.violinplot(x='crop', y=y, data=df_filter, ax=axes[1])
    axes[2] = sns.violinplot(x='clip', y=y, data=df_filter, ax=axes[2])
    axes[3] = sns.violinplot(x='smooth', y=y, data=df_filter, ax=axes[3])
    axes[3] = sns.violinplot(x='bin', y=y, data=df_filter, ax=axes[4])
    axes[4] = sns.violinplot(x='segment', y=y, data=df_filter, ax=axes[5])

    [ax.set_xticklabels(ax.get_xticklabels(), rotation=60) for ax in axes]
    fig.suptitle('{0} - {1} feature(s)'.format(base_name, y), fontsize=16)
    fig.tight_layout()
    fig.tight_layout()
    return fig

# %% Violin plots opt
import seaborn as sns
import matplotlib.pyplot as plt

def sip_n_feats_opt_filter(df_opt, response, feature_set, objective_f):
    '''
    Filters SIP results by response, feature_set, model, and objective_f.
    '''
    df_filter = df_opt[(df_opt['response_label'] == response) &
                       (df_opt['feature_set'] == feature_set) &
                       (df_opt['objective_f'] == objective_f)]
    return df_filter


def plot_violin_by_scenario_opt(df_opt, name, y='value'):
    '''
    Makes violin plot for each processing scenario
    '''
    fig, axes = plt.subplots(ncols=6, sharey=True, figsize=(14, 8),
                             gridspec_kw={'width_ratios': [2, 2, 3, 2, 3, 9]})

    axes[0] = sns.violinplot(x='dir_panels', y=y, data=df_opt, ax=axes[0])
    axes[1] = sns.violinplot(x='crop', y=y, data=df_opt, ax=axes[1])
    axes[2] = sns.violinplot(x='clip', y=y, data=df_opt, ax=axes[2])
    axes[3] = sns.violinplot(x='smooth', y=y, data=df_opt, ax=axes[3])
    axes[3] = sns.violinplot(x='bin', y=y, data=df_opt, ax=axes[4])
    axes[4] = sns.violinplot(x='segment', y=y, data=df_opt, ax=axes[5])

    [ax.set_xticklabels(ax.get_xticklabels(), rotation=60) for ax in axes]
    fig.suptitle('{0}'.format(name), fontsize=16)
    fig.tight_layout()
    fig.tight_layout()
    return fig


# # %% SPOTPY example
# import spotpy
# from spotpy import analyser
# from spotpy.examples.spot_setup_rosenbrock import spot_setup

# sampler = spotpy.algorithms.mc(spot_setup(), dbname='RosenMC', dbformat='csv')

# sampler.sample(100000)
# results=sampler.getdata()

# %% Sensitivity analysis using SALib
import os
import pandas as pd
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli

problem = {
    'num_vars': 6,
    'names': ['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'],
    'bounds': [[0, 1],
                [0, 1],
                [0, 2],
                [0, 1],
                [0, 2],
                [0, 8]]}

param_values = saltelli.sample(problem, 1)



base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'
base_dir_out = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_metafigures'

df_opt = pd.read_csv(os.path.join(base_dir_results, 'msi_2_n_feats_opt.csv'))

for response in ['biomass_kgha', 'nup_kgha', 'tissue_n_pct']:
    for feature_set in ['reflectance', 'derivative_1', 'derivative_2']:
        # for model in ['Lasso', 'PLSRegression']:
        for objective_f in ['R2', 'MAE', 'RMSE']:
            df_opt_filter = sip_n_feats_opt_filter(
                df_opt, response, feature_set, objective_f)
            break
        break
    break



df_opt = pd.read_csv(os.path.join(base_dir_results, 'msi_2_n_feats_opt.csv'))
data = [333, 'all', 'plot_bounds', 'none', 'none', 'senitnel-2a_mimic', 'none',
        2, 'nup_kgha', 'reflectance', 'Lasso', 'R2', 8, 0.744304]
df_opt_row = pd.DataFrame(data=[data], columns=df_opt.columns)
df_opt = df_opt.append(df_opt_row)

response = 'nup_kgha'
feature_set = 'reflectance'
objective_f = 'RMSE'
model = 'Lasso'

df_opt_filter = sip_n_feats_opt_filter(df_opt, response, feature_set, objective_f)

df_opt_filter.sort_values(by=['model_name', 'segment', 'bin', 'smooth', 'clip', 'crop', 'dir_panels'], axis=0, inplace=True)
df_opt_filter = df_opt_filter[df_opt_filter['model_name'] == model]


Y = df_opt_filter['value'].to_numpy()

from SALib.sample import saltelli
param_values = saltelli.sample(problem, 46)

Si = sobol.analyze(problem, Y)

Y = np.zeros([param_values.shape[0]])

for i, X in enumerate(param_values):
    Y[i] = evaluate_model(X)

# base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'
# base_dir_out = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta\optimum_features'

# os.path.join(base_dir_results, 'msi_2_nup_kgha_RMSE.csv')

from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from SALib.analyze import sobol
problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-3.14159265359, 3.14159265359],
                [-3.14159265359, 3.14159265359],
                [-3.14159265359, 3.14159265359]]
}
param_values = saltelli.sample(problem, 1000)
Y = Ishigami.evaluate(param_values)

Si = sobol.analyze(problem, Y)

>>> X = saltelli.sample(problem, 1000)
>>> Y = Ishigami.evaluate(X)
>>> Si = sobol.analyze(problem, Y, print_to_console=True)

import SALib
from SALib.test_functions import Ishigami
from SALib.analyze import sobol
problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-3.14159265359, 3.14159265359],
                [-3.14159265359, 3.14159265359],
                [-3.14159265359, 3.14159265359]]
}
problem = {
    'num_vars': 6,
    'names': ['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'],
    'bounds': [[0, 1],
                [0, 1],
                [0, 2],
                [0, 1],
                [0, 2],
                [0, 8]]}

N = 1
seed=998
M=4
X = SALib.sample.saltelli.sample(problem, N, seed=seed)
Y = Ishigami.evaluate(X)
Si = sobol.analyze(problem, Y, print_to_console=True)
