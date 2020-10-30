# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 08:23:04 2020

@author: nigo0024
"""
import pandas as pd

def read_results_and_filter(fname, df_grid, model='Lasso'):
    '''
    Reads in
    '''
    df = pd.read_csv(fname)

    df_filter = df[(pd.isnull(df['extra_feats'])) &
                   (df['model_name'] == model)]
    df_filter = df_grid.merge(df_filter, left_index=True, right_on='grid_idx')

    df_filter.loc[:,'clip'] = df_filter.loc[:,'clip'].astype('str')
    df_filter.loc[:,'smooth'] = df_filter.loc[:,'smooth'].astype('str')
    df_filter.loc[:,'segment'] = df_filter.loc[:,'segment'].astype('str')

    # df_filter['clip'].unique()
    df_filter['clip'].replace({'nan': 'None', "{'wl_bands': [[0, 420], [880, 1000]]}": 'Ends',
           "{'wl_bands': [[0, 420], [760, 776], [813, 827], [880, 1000]]}": 'All'}, inplace=True)
    df_filter['smooth'].unique()
    df_filter['smooth'].replace({'nan': 'None', "{'window_size': 5, 'order': 2}": 'SG-5',
           "{'window_size': 11, 'order': 2}": 'SG-11'}, inplace=True)
    df_filter['segment'].unique()
    df_filter['segment'].replace({
        'nan': 'None',
        "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': 50, 'mask_side': 'lower'}": 'MCARI2_upper_50',
        "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': 90, 'mask_side': 'lower'}": 'MCARI2_upper_90',
        "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': [50, 75], 'mask_side': 'outside'}": 'MCARI2_in_50-75',
        "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': [75, 95], 'mask_side': 'outside'}": 'MCARI2_in_75-95',
        "{'method': 'mcari2', 'wl1': [800], 'wl2': [670], 'wl3': [550], 'mask_percentile': [95, 98], 'mask_side': 'outside'}": 'MCARI2_in_95-98',
        "{'method': ['mcari2', [545, 565]], 'wl1': [[800], [None]], 'wl2': [[670], [None]], 'wl3': [[550], [None]], 'mask_percentile': [90, 75], 'mask_side': ['lower', 'lower']}": 'MCARI2_upper_90_green_upper_75',
        "{'method': ['mcari2', [800, 820]], 'wl1': [[800], [None]], 'wl2': [[670], [None]], 'wl3': [[550], [None]], 'mask_percentile': [90, 75], 'mask_side': ['lower', 'lower']}": 'MCARI2_upper_90_nir_upper_75'}, inplace=True)
    return df_filter
# In[Boxplot]
import seaborn as sns
import matplotlib.pyplot as plt

def plot_violin_each_processing_step(df_filter, y='2'):
    '''
    Makes violin plot for msi results
    '''
    fig, axes = plt.subplots(ncols=5, sharey=True, figsize=(14, 8),
                             gridspec_kw={'width_ratios': [2, 2, 3, 3, 8]})

    axes[0] = sns.violinplot(x='dir_panels', y=y, data=df_filter, ax=axes[0])
    axes[1] = sns.violinplot(x='crop', y=y, data=df_filter, ax=axes[1])
    axes[2] = sns.violinplot(x='clip', y=y, data=df_filter, ax=axes[2])
    axes[3] = sns.violinplot(x='smooth', y=y, data=df_filter, ax=axes[3])
    axes[4] = sns.violinplot(x='segment', y=y, data=df_filter, ax=axes[4])

    [ax.set_xticklabels(ax.get_xticklabels(), rotation=60) for ax in axes]
    fig.suptitle('{0}'.format(os.path.split(f)[-1]), fontsize=16)
    fig.tight_layout()
    fig.tight_layout()
    return fig

# In[Make plots]
base_dir_results = r'G:\BBE\AGROBOT\Tyler Nigon\subjective_image_proc_msi'
base_dir_out = r'G:\BBE\AGROBOT\Tyler Nigon\subjective_image_proc_msi\figures'
fnames = ['msi_1_biomass_kgha_R2.csv', 'msi_1_nup_kgha_R2.csv', 'msi_1_tissue_n_pct_R2.csv']
df_grid = pd.read_csv(os.path.join(base_dir_results, 'msi_1_hs_settings.csv'))
f_full = [os.path.join(base_dir_results, f) for f in fnames]
for f in f_full:
    df_filter = read_results_and_filter(f, df_grid)
    for n_feats in range(1, 20, 3):
        fig = plot_violin_each_processing_step(df_filter, y=str(n_feats))
        fig.savefig(os.path.join(base_dir_out, '{0}_{1}.png'.format(os.path.splitext(os.path.split(f)[-1])[0], n_feats)), dpi=300)


# In[Get keys]
df_filter['clip'].unique()
df_filter['smooth'].unique()
df_filter['segment'].unique()
