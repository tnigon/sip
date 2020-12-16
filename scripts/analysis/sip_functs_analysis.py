# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:42:15 2020

@author: nigo0024
"""
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

# %% Violin plots
# base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'
# base_dir_out = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta\optimum_features'

# os.path.join(base_dir_results, 'msi_2_nup_kgha_RMSE.csv')
