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

# %% Build Spotpy "results" table
import numpy as np
import os
import pandas as pd
from scripts.analysis import sip_functs_analysis as sip_f

base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'
df_opt = pd.read_csv(os.path.join(base_dir_results, 'msi_2_n_feats_opt.csv'))

cols = ['like1', 'like2', 'like3', 'pardir_panels', 'parcrop', 'parclip', 'parsmooth', 'parbin',
        'parsegment', 'simulation_rmse', 'simulation_mae', 'simulation_r2', 'chain']

options = [[response, feature] for response in ['biomass_kgha', 'nup_kgha', 'tissue_n_pct']
           for feature in ['reflectance', 'derivative_1', 'derivative_2']]

for response, feature in options:
    df_spotpy = pd.DataFrame(data=[], columns=cols)
    df_opt_filter1 = df_opt[(df_opt['response_label'] == response) &
                            (df_opt['feature_set'] == feature)]
    df_opt_filter1 = df_opt_filter1.sort_values(['objective_f', 'dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])
    df_spotpy['like1'] = list(df_opt_filter1[df_opt_filter1['objective_f'] == 'RMSE']['value'])
    df_spotpy['like2'] = list(df_opt_filter1[df_opt_filter1['objective_f'] == 'MAE']['value'])
    df_spotpy['like3'] = list(df_opt_filter1[df_opt_filter1['objective_f'] == 'R2']['value'])
    df_spotpy['simulation_rmse'] = list(df_opt_filter1[df_opt_filter1['objective_f'] == 'RMSE']['value'])
    df_spotpy['simulation_mae'] = list(df_opt_filter1[df_opt_filter1['objective_f'] == 'MAE']['value'])
    df_spotpy['simulation_r2'] = list(df_opt_filter1[df_opt_filter1['objective_f'] == 'R2']['value'])
    # df_spotpy['simulation_0'] = list(df_opt_filter1[df_opt_filter1['objective_f'] == 'RMSE'].index)
    # df_spotpy['simulation_1'] = list(df_opt_filter1[df_opt_filter1['objective_f'] == 'MAE'].index)
    # df_spotpy['simulation_2'] = list(df_opt_filter1[df_opt_filter1['objective_f'] == 'R2'].index)

    df_opt_filter2 = df_opt_filter1[(df_opt['objective_f'] == 'RMSE')]
    df_spotpy['pardir_panels'] = list(df_opt_filter2['dir_panels'])
    df_spotpy['parcrop'] = list(df_opt_filter2['crop'])
    df_spotpy['parclip'] = list(df_opt_filter2['clip'])
    df_spotpy['parsmooth'] = list(df_opt_filter2['smooth'])
    df_spotpy['parbin'] = list(df_opt_filter2['bin'])
    df_spotpy['parsegment'] = list(df_opt_filter2['segment'])
    df_opt_filter2.loc[df_opt_filter2['model_name'] == 'Lasso', 'model_idx'] = 0
    df_opt_filter2.loc[df_opt_filter2['model_name'] == 'PLSRegression', 'model_idx'] = 1
    df_spotpy['chain'] = list(df_opt_filter2['model_idx'])
    df_spotpy_int = df_spotpy.copy()
    levels_dict = {}
    for param in ['pardir_panels', 'parcrop', 'parclip', 'parsmooth', 'parbin', 'parsegment']:
        labels, levels = pd.factorize(df_spotpy[param])
        df_spotpy_int.loc[:, param] = labels
        levels_dict[param] = list(levels)
        levels_dict[param + '_idx'] = list(np.unique(labels))
    df_spotpy_int.to_csv(
        os.path.join(base_dir_results, 'spotpy',
                     'results_{0}_{1}.csv'.format(response, feature)),
        index=False)

with open(os.path.join(base_dir_results, 'spotpy', 'README_params.txt'), 'w') as f:
    for k in levels_dict:
        f.write('{0}: {1}\n'.format(str(k), str(levels_dict[k])))

# %% Histogram - get data
import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick

base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'
base_dir_spotpy = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta\spotpy'

df_opt = pd.read_csv(os.path.join(base_dir_results, 'msi_2_n_feats_opt.csv'))

options = [[response, feature, obj] for response in ['biomass_kgha', 'nup_kgha', 'tissue_n_pct']
           for feature in ['reflectance', 'derivative_1', 'derivative_2']
           for obj in ['MAE', 'RMSE', 'R2']]
units_print = {
    'biomass_kgha': r'kg {ha\boldmath$^-1}$',
    'nup_kgha': r'kg ha$^{-1}$',
    'tissue_n_pct': '%'}

for response, feature, obj in options:

    df_opt_filter = df_opt[
        (df_opt['response_label'] == response) &
        (df_opt['feature_set'] == feature) &
        (df_opt['objective_f'] == obj)]
    break

# Choose response, feature set, and cost function manually
response = 'nup_kgha'
feature = 'reflectance'
obj = 'RMSE'
df_opt_filter = df_opt[
    (df_opt['response_label'] == response) &
    (df_opt['feature_set'] == feature) &
    (df_opt['objective_f'] == obj)]


sns.set_style("whitegrid")
grids_all = [(r, c) for r in range(2) for c in range(3)]
grids = [(r, c) for r in range(1) for c in range(3)]
# grids_bottom = [(r, c) for r in range(1,2) for c in range(3)]

scenario_options = {  # These indicate the legend labels
    'dir_panels': {
        'name': 'Reflectance panels',
        'closest': 'Closest panel',
        'all': 'All panels (mean)'},
    'crop': {
        'name': 'Crop',
        'plot_bounds': 'By plot boundary',
        'crop_buf': 'Edges cropped'},
    'clip': {
        'name': 'Clip',
        'none': 'No spectral clipping',
        'ends': 'Ends clipped',
        'all': 'Ends + H2O and O2 absorption'},
    'smooth': {
        'name': 'Smooth',
        'none': 'No spectral smoothing',
        'sg-11': 'Savitzky-Golay smoothing'},
    'bin': {
        'name': 'Bin',
        'none': 'No spectral binning',
        'sentinel-2a_mimic': 'Spectral "mimic" - Sentinel-2A',
        'bin_20nm': 'Spectral "bin" - 20 nm'},
    'segment': {
        'name': 'Segment',
        'none': 'No segmenting',
        'ndi_upper_50': 'NDVI > 50th',
        'ndi_lower_50': 'NDVI < 50th',
        'mcari2_upper_50': 'MCARI2 > 50th',
        'mcari2_lower_50': 'MCARI2 < 50th',
        'mcari2_upper_90': 'MCARI2 > 90th',
        'mcari2_in_50-75': '50th > MCARI2 < 75th',
        'mcari2_in_75-95': '75th > MCARI2 < 95th',
        'mcari2_upper_90_green_upper_75': 'MCARI2 > 90th; green > 75th'},
    }

scenario_options_top = {k: scenario_options[k] for k in ['dir_panels', 'crop', 'clip']}
scenario_options_bottom = {k: scenario_options[k] for k in ['smooth', 'bin', 'segment']}

# %% Curate stats for cumulative density plots
classes_rmse = [14, 15, 16, 17, 18, 19]
classes_n_feats = [0, 5, 10, 15, 20, 25]
array_props = {'models': np.empty([2, len(classes_rmse)])}
cols = ['model_name', 'metric', 'class_min_val', 'proportion']
df_props_model = None
for i_model, model in enumerate(['Lasso', 'PLSRegression']):
    df_model = df_opt_filter[df_opt_filter['model_name'] == model]
    for i_classes, i_min_val in enumerate(range(len(classes_rmse))):
        min_val = classes_rmse[i_min_val]
        prop = (len(df_model[df_model['value'].between(min_val, min_val+1)]) / len(df_model)) * 100
        array_props['models'][i_model, i_classes] = prop
        data = [model, 'rmse', min_val, prop]
        if df_props_model is None:
            df_props_model = pd.DataFrame([data], columns=cols)
        else:
            df_props_model = df_props_model.append(pd.DataFrame([data], columns=cols))
        min_val2 = classes_n_feats[i_min_val]
        prop2 = (len(df_model[df_model['n_feats_opt'].between(min_val2, min_val2+5)]) / len(df_model)) * 100
        array_props['models'][i_model, i_classes] = prop2
        data2 = [model, 'n_feats_opt', min_val2, prop2]
        if df_props_model is None:
            df_props_model = pd.DataFrame([data2], columns=cols)
        else:
            df_props_model = df_props_model.append(pd.DataFrame([data2], columns=cols))

# %% Calculate mean and median of all RMSE values
df_opt_filter['value'].describe()

# %% A1. Plot MSE train and validation score vs number of features
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from ast import literal_eval

def plot_and_fill_std(df, ax, palette, legend, objective='mae'):
    df_wide = df[['feat_n', 'score_train_' + objective, 'score_test_' + objective]].apply(pd.to_numeric).set_index('feat_n')
    if df_wide['score_train_' + objective].iloc[1] < 0:
        df_wide[['score_train_' + objective, 'score_test_' + objective]] = df_wide[['score_train_' + objective, 'score_test_' + objective]] * -1
    x_feats = df_wide.index
    ax = sns.lineplot(data=df_wide[['score_train_' + objective, 'score_test_' + objective]], ax=ax, palette=palette, legend=legend)
    ax.lines[0].set_linewidth(1)
    ax.lines[0].set_linestyle('-')
    ax.lines[1].set_linestyle('-')
    return ax

def plot_secondary(df, ax, palette, legend, objective='r2'):
    df_wide = df[['feat_n', 'score_train_' + objective, 'score_test_' + objective]].apply(pd.to_numeric).set_index('feat_n')
    if df_wide['score_train_' + objective].iloc[1] < 0:
        df_wide[['score_train_' + objective, 'score_test_' + objective]] = df_wide[['score_train_' + objective, 'score_test_' + objective]] * -1
    ax2 = ax.twinx()
    ax2 = sns.lineplot(data=df_wide[['score_train_' + objective, 'score_test_' + objective]], ax=ax2, palette=palette, legend=legend)
    ax2.lines[0].set_linewidth(1)
    ax2.lines[0].set_linestyle('--')
    ax2.lines[1].set_linestyle('--')
    ax2.grid(False)
    return ax, ax2

plt.style.use('seaborn-whitegrid')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

fontsize = 12
fontcolor = '#464646'
colors = ['#f0bf5d', '#68de78']
palette = sns.color_palette("mako_r", 2)

fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
fig.subplots_adjust(
top=0.89,
bottom=0.158,
left=0.16,
right=0.836)

df_las = pd.read_csv(r'G:\BBE\AGROBOT\Shared Work\_migrate\hs_process_results\results\msi_2_results\msi_2_521\nup_kgha\reflectance\testing\msi_2_521_nup_kgha_test-scores-lasso.csv')
d = {}
with open(r'G:\BBE\AGROBOT\Shared Work\_migrate\hs_process_results\results\msi_2_results\msi_2_521\nup_kgha\reflectance\reflectance_README.txt') as f:
    for i, l in enumerate(f):
        if i >= 6:
            band, wl = l.split(': ')
            d[int(band)] = float(wl)

feat_n_opt = df_las[df_las['score_test_rmse'] == df_las['score_test_rmse'].min()]['feat_n'].values[0]
feats_opt = literal_eval(df_las[df_las['score_test_rmse'] == df_las['score_test_rmse'].min()]['feats'].values[0])
wls_opt = [d[b] for b in feats_opt]
len(wls_opt) == feat_n_opt
print([round(wl) for wl in wls_opt])
# [399, 405, 417, 516, 571, 682, 705, 721, 723, 735, 764, 781, 811, 815, 824, 826, 848, 850, 856, 863]

objective = 'rmse'
ax1 = plot_and_fill_std(df_las, ax, palette, legend=False, objective=objective)
ax1, ax1b = plot_secondary(df_las, ax1, palette, legend='full', objective='r2')

if objective == 'rmse':
    ylabel = r'RMSE (kg ha$^{-1}$)'
    ax1.set_ylim([0, 32])
elif objective == 'mae':
    ylabel = r'Error (kg ha$^{-1}$)'
    ax1.set_ylim([0, 25])
ax1b.set_ylim([0, 1])

ax1.tick_params(labelsize=int(fontsize), colors=fontcolor, labelleft=True)
# t1 = ax1b.set_title('Lasso', fontsize=fontsize*1.1, fontweight='bold', color='white', bbox=dict(facecolor=(0.35,0.35,0.35), edgecolor=(0.35,0.35,0.35)))
ax1.set_ylabel(ylabel, fontsize=fontsize, color=fontcolor)
ax1b.set_ylabel(r'R$^{2}$', fontsize=fontsize, color=fontcolor, rotation=0, labelpad=15)
ax1b.tick_params(labelsize=int(fontsize), colors=fontcolor, labelright=True)
ax1.set_xlabel('Number of features', fontsize=fontsize, color=fontcolor)

ax1.set_xlim([-0.1, df_las['feat_n'].max() + 1])

h1, l1 = ax1b.get_legend_handles_labels()
h1.insert(0, mpatches.Patch(color=palette[1], label='Test set'))
h1.insert(0, mpatches.Patch(color=palette[0], label='Train set'))
l1 = [r'Train set', r'Test set', 'RMSE', r'R$^{2}$']
h1[2].set_linestyle('-')
h1[3].set_linestyle('--')
h1[2].set_linewidth(2)
h1[3].set_linewidth(2)
h1[2].set_color((0.35,0.35,0.35))
h1[3].set_color((0.35,0.35,0.35))

leg = ax1b.legend(h1, l1, loc='upper center',
                  handletextpad=0.4, ncol=4, columnspacing=1, fontsize=int(fontsize*0.8),
                  bbox_to_anchor=(0.5, 1.17), frameon=True, framealpha=1,
                  edgecolor=fontcolor)
for handle, text in zip(leg.legendHandles, leg.get_texts()):
    text.set_color(fontcolor)
ax1b.add_artist(leg)

# %% A2a: Lasso vs PLS PDF/boxplot top
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(8, 3.8),
                         gridspec_kw={'height_ratios': [2, 6]})
fig.subplots_adjust(
top=1.0,
bottom=0.268,
left=0.076,
right=0.942,
hspace=0.04,
wspace=0.399)

for i, val in enumerate(['value', 'n_feats_opt']):
    # break
    palette = sns.color_palette('muted', n_colors=len(df_opt_filter['model_name'].unique()))
    ax2 = sns.histplot(ax=axes[1][i], data=df_opt_filter, x=val, alpha=0.3, color='#999999')

    ax2.set_ylabel('Count', weight='bold', fontsize='large')

    ax2.yaxis.set_major_locator(mtick.FixedLocator(ax2.get_yticks()))
    ax2.set_yticklabels(['{:.0f}'.format(t) for t in ax2.get_yticks()], weight='bold', fontsize='large')

    ax3 = ax2.twinx()
    ax3 = sns.kdeplot(ax=ax3, data=df_opt_filter, x=val, hue='model_name',
                      cut=0, bw_adjust=0.7, common_norm=True, common_grid=False,
                      label='temp', palette=palette)

    if val == 'value':
        ax2.yaxis.set_ticks(np.arange(0, 160, 50))
        ax2.set_ylim(bottom=0, top=160)
        ax2.set_xlim(left=14, right=20)
        ax2.set_xlabel('{0} ({1})'.format(obj, units_print[response]),
                       fontweight='bold', fontsize='large')
    else:
        ax2.yaxis.set_ticks(np.arange(0, 330, 100))
        ax2.set_ylim(bottom=0, top=330)
        ax2.set_xlim(left=0, right=31)
        ax2.set_xlabel('Feature $\it{n}$ at optimum ' + '{0}'.format(obj),
                fontweight='bold', fontsize='large')
    # else:
    #     ax1.set_xlim(left=0, right=20)
    # set ticks visible, if using sharex = True. Not needed otherwise
    ax2.tick_params(labelbottom=True)
    ax2.xaxis.set_major_locator(mtick.FixedLocator(ax3.get_xticks()))
    ax2.set_xticklabels(['{:.0f}'.format(t) for t in ax2.get_xticks()],
                        fontweight='bold', fontsize='large')


    step_size = ax2.yaxis.get_ticklocs()[-1] - ax2.yaxis.get_ticklocs()[-2]
    extra = (ax2.get_ylim()[-1] - ax2.yaxis.get_ticklocs()[-1]) / step_size
    space = ax3.get_ylim()[-1] / (3.0 + extra)
    ax3.set_yticks(np.arange(ax3.get_ylim()[0], ax3.get_ylim()[-1]+(space*extra), space))
    ax3.grid(False)
    ax3.set_yticklabels(ax3.get_yticks(), weight='bold', rotation=90,
                        horizontalalignment='left', verticalalignment='center',
                        fontsize='large')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.set_ylabel('Density (%)', weight='bold', fontsize='large')

    h, _ = ax3.get_legend_handles_labels()
    h = list(reversed(h))
    l2 = ['Lasso', 'Partial Least Squares']
    ncol = 2 if len(h) > 4 else 1
    fontsize = 8.75 if len(h) > 3 else 'medium'
    label_color='#464646'
    color_grid = ax3.get_ygridlines()[0].get_color()
    leg = ax3.legend(h, l2,
                     bbox_to_anchor=(0, -0.24, 1, 0), loc='upper left',
                     mode='expand', ncol=ncol,
                     # fontsize=fontsize,
                     framealpha=0.85,
                     handletextpad=0.1,  # spacing between handle and label
                     columnspacing=0.5,
                     frameon=True,
                     edgecolor=color_grid,
                     prop={'weight':'bold',
                           'size': fontsize})

    # Finally, add boxplot now knowing the width
    box_width = 0.7 if len(h) > 4 else 0.65 if len(h) > 2 else 0.45
    ax1 = sns.boxplot(ax=axes[0][i], data=df_opt_filter, x=val, y='model_name',
                      width=box_width, fliersize=2, linewidth=1, palette=palette)
    ax1.set(yticklabels=[], ylabel=None)
    ax1.set(xticklabels=[], xlabel=None)
    if val == 'value':
        ax1.set_xlim(left=14, right=20)
    else:
        ax1.set_xlim(left=0, right=31)

# %% A2b: Lasso vs PLS ECDF/heatmap bottom
fig, axes = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(8, 3.8),
                         gridspec_kw={'height_ratios': [1, 1, 6]})
fig.subplots_adjust(
top=0.994,
bottom=0.274,
left=0.078,
right=0.942,
hspace=0.000,
wspace=0.399)

for i, val in enumerate(['value', 'n_feats_opt']):
    # break

    ax1 = sns.histplot(ax=axes[2][i], data=df_opt_filter, x=val, common_norm=False, cumulative=True, stat='density', alpha=0.3, color='#999999')
    palette=sns.color_palette('muted', n_colors=len(df_opt_filter['model_name'].unique()))
    ax2 = sns.ecdfplot(ax=ax1, data=df_opt_filter, x=val, hue='model_name',
                       label='temp', palette=palette)
    ax1.set_ylim(bottom=0, top=1.05)
    ax1.yaxis.set_ticks(np.arange(0, 1.05, 0.25))


    if val == 'value':
        ax1.set_ylabel('Density', weight='bold', fontsize='large')
        ax1.set_yticklabels(ax1.get_yticks(), weight='bold', fontsize='large')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_xlim(left=14, right=20)
        ax1.set_xlabel('{0} ({1})'.format(obj, units_print[response]),
                       fontweight='bold', fontsize='large')
        df_props_model_pivot = df_props_model[df_props_model['metric'] == 'rmse'].pivot('model_name', 'class_min_val', 'proportion')
    else:
        ax1.set(yticklabels=[], ylabel=None)
        ax1.set_ylabel('')
        ax1.set_xlim(left=0, right=31)
        ax1.set_xlabel('Feature $\it{n}$ at optimum ' + '{0}'.format(obj),
                fontweight='bold', fontsize='large')
        df_props_model_pivot = df_props_model[df_props_model['metric'] == 'n_feats_opt'].pivot('model_name', 'class_min_val', 'proportion')

    ax1.tick_params(labelbottom=True)
    ax1.xaxis.set_major_locator(mtick.FixedLocator(ax1.get_xticks()))
    ax1.set_xticklabels(['{:.0f}'.format(t) for t in ax1.get_xticks()],
                        fontweight='bold', fontsize='large')

    # legend
    h, _ = ax2.get_legend_handles_labels()
    h = list(reversed(h))
    l2 = ['Lasso', 'Partial Least Squares']
    ncol = 2 if len(h) > 4 else 1
    fontsize = 8.75 if len(h) > 3 else 'medium'
    label_color='#464646'
    color_grid = ax2.get_ygridlines()[0].get_color()
    leg = ax2.legend(h, l2,
                     bbox_to_anchor=(0, -0.24, 1, 0), loc='upper left',
                     mode='expand', ncol=ncol,
                     framealpha=0.85,
                     handletextpad=0.1,  # spacing between handle and label
                     columnspacing=0.5,
                     frameon=True,
                     edgecolor=color_grid,
                     prop={'weight':'bold',
                           'size': fontsize})

    for i_model, model in enumerate(['Lasso', 'PLSRegression']):
        ax = sns.heatmap(
            ax=axes[i_model][i], data=df_props_model_pivot.loc[[model]],
            annot=True, fmt='.1f', annot_kws={'weight': 'bold', 'fontsize': 9.5},
            linewidths=4, yticklabels=False,
            cbar=False, cmap=sns.light_palette(palette[i_model], as_cmap=True))
        for t in ax.texts: t.set_text('') if float(t.get_text()) == 0.0 else t.set_text(t.get_text() + '%')
        ax.set_xlabel('')
        ax.set_ylabel('')
        if val == 'n_feats_opt':
            ax.set_xlim(left=0, right=6.2)


# %% A2c: Lasso vs PLS boxplot RMSE
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(4, 3.5),
                         gridspec_kw={'height_ratios': [6, 2.5]})
fig.subplots_adjust(
top=0.987,
bottom=0.15,
left=0.148,
right=0.884,
hspace=0.49,
wspace=0.185)
# fig.suptitle('Model accuracy from subjective image processing ({0} features)'.format(feature), fontsize=16)
# for scenario, (row, col) in zip(scenario_options, grids):
ax1 = sns.histplot(ax=axes[0], data=df_opt_filter, x='value', alpha=0.3, color='#999999')
ax1.yaxis.set_ticks(np.arange(0, 160, 50))

ax1.set_ylabel('Count', weight='bold', fontsize='large')
ax1.set_yticklabels(ax1.get_yticks(), weight='bold', fontsize='large')

palette=sns.color_palette('muted', n_colors=len(df_opt_filter['model_name'].unique()))
ax2 = ax1.twinx()
ax2 = sns.kdeplot(ax=ax2, data=df_opt_filter, x='value', hue='model_name',
                  cut=0, bw_adjust=0.7, common_norm=True, common_grid=False,
                  label='temp', palette=palette)

step_size = ax1.yaxis.get_ticklocs()[-1] - ax1.yaxis.get_ticklocs()[-2]
extra = (ax1.get_ylim()[-1] - ax1.yaxis.get_ticklocs()[-1]) / step_size
space = ax2.get_ylim()[-1] / (3.0 + extra)
ax2.set_yticks(np.arange(ax2.get_ylim()[0], ax2.get_ylim()[-1]+(space*extra), space))
ax2.grid(False)

ax2.set_yticklabels(ax2.get_yticks(), weight='bold', rotation=90,
                    horizontalalignment='left', verticalalignment='center',
                    fontsize='large')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.set_ylabel('Density (%)', weight='bold', fontsize='large')

h, _ = ax2.get_legend_handles_labels()
h = list(reversed(h))
# l1 = list(df_opt_filter['model_name'].unique())
# l2 = [scenario_options['model_name'][uid] for uid in df_opt_filter['model_name'].unique()]
l2 = ['Lasso', 'Partial Least Squares']

width = 0.7 if len(h) > 4 else 0.6 if len(h) > 2 else 0.4
ax3 = sns.boxplot(ax=axes[1], data=df_opt_filter, x='value', y='model_name',
                  width=width, fliersize=2, linewidth=1, palette=palette)

ax3.set_xlabel('{0} ({1})'.format(obj, units_print[response]),
               fontweight='bold', fontsize='large')
# set ticks visible, if using sharex = True. Not needed otherwise
ax3.tick_params(labelbottom=True)
ax3.xaxis.set_major_locator(mtick.FixedLocator(ax3.get_xticks()))
ax3.set_xticklabels(['{:.0f}'.format(t) for t in ax3.get_xticks()],
                    fontweight='bold', fontsize='large')
ax3.set(yticklabels=[], ylabel=None)
ncol = 2 if len(h) > 4 else 1
fontsize = 8.75 if len(h) > 3 else 'medium'
# fontsize = 'small'
label_color='#464646'
color_grid = ax1.get_ygridlines()[0].get_color()
leg = ax3.legend(h, l2,
                 bbox_to_anchor=(0, 1, 1, 0), loc='lower left',
                 mode='expand', ncol=ncol,
                 # fontsize=fontsize,
                 framealpha=0.85,
                 handletextpad=0.1,  # spacing between handle and label
                 columnspacing=0.5,
                 frameon=True,
                 edgecolor=color_grid,
                 prop={'weight':'bold',
                       'size': fontsize})
ax2.get_legend().remove()

# %% A2d: Lasso vs PLS boxplot number of features to optimum
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(4, 3.5),
                         gridspec_kw={'height_ratios': [6, 2.5]})
fig.subplots_adjust(
top=0.987,
bottom=0.15,
left=0.148,
right=0.884,
hspace=0.49,
wspace=0.185)
# fig.suptitle('Model accuracy from subjective image processing ({0} features)'.format(feature), fontsize=16)
# for scenario, (row, col) in zip(scenario_options, grids):
ax1 = sns.histplot(ax=axes[0], data=df_opt_filter, x='n_feats_opt', alpha=0.3, color='#999999')
ax1.set_ylim(bottom=0, top=330)
ax1.yaxis.set_ticks(np.arange(0, 330, 100))

ax1.set_ylabel('Count', weight='bold', fontsize='large')
ax1.set_yticklabels(ax1.get_yticks(), weight='bold', fontsize='large')

palette=sns.color_palette('muted', n_colors=len(df_opt_filter['model_name'].unique()))
ax2 = ax1.twinx()
ax2 = sns.kdeplot(ax=ax2, data=df_opt_filter, x='n_feats_opt', hue='model_name',
                  cut=0, bw_adjust=0.7, common_norm=True, common_grid=False,
                  label='temp', palette=palette)
ax2.set_ylim(bottom=0)
    # break

step_size = ax1.yaxis.get_ticklocs()[-1] - ax1.yaxis.get_ticklocs()[-2]
extra = (ax1.get_ylim()[-1] - ax1.yaxis.get_ticklocs()[-1]) / step_size
space = ax2.get_ylim()[-1] / (3.0 + extra)
ax2.set_yticks(np.arange(ax2.get_ylim()[0], ax2.get_ylim()[-1]+(space*extra), space))
ax2.grid(False)

ax2.set_yticklabels(ax2.get_yticks(), weight='bold', rotation=90,
                    horizontalalignment='left', verticalalignment='center',
                    fontsize='large')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.set_ylabel('Density (%)', weight='bold', fontsize='large')

h, _ = ax2.get_legend_handles_labels()
h = list(reversed(h))
# l1 = list(df_opt_filter['model_name'].unique())
# l2 = [scenario_options['model_name'][uid] for uid in df_opt_filter['model_name'].unique()]
l2 = ['Lasso', 'Partial Least Squares']

width = 0.7 if len(h) > 4 else 0.6 if len(h) > 2 else 0.4
ax3 = sns.boxplot(ax=axes[1], data=df_opt_filter, x='n_feats_opt', y='model_name',
                  width=width, fliersize=2, linewidth=1, palette=palette)

ax3.set_xlabel('Feature $\it{n}$ at optimum ' + '{0}'.format(obj),
                fontweight='bold', fontsize='large')
# set ticks visible, if using sharex = True. Not needed otherwise
ax3.tick_params(labelbottom=True)
ax3.xaxis.set_major_locator(mtick.FixedLocator(ax3.get_xticks()))
ax3.set_xticklabels(['{:.0f}'.format(t) for t in ax3.get_xticks()],
                    fontweight='bold', fontsize='large')
ax3.set(yticklabels=[], ylabel=None)
ncol = 2 if len(h) > 4 else 1
fontsize = 8.75 if len(h) > 3 else 'medium'
# fontsize = 'small'
label_color='#464646'
color_grid = ax1.get_ygridlines()[0].get_color()
leg = ax3.legend(h, l2,
                 bbox_to_anchor=(0, 1, 1, 0), loc='lower left',
                 mode='expand', ncol=ncol,
                 # fontsize=fontsize,
                 framealpha=0.85,
                 handletextpad=0.1,  # spacing between handle and label
                 columnspacing=0.5,
                 frameon=True,
                 edgecolor=color_grid,
                 prop={'weight':'bold',
                       'size': fontsize})
ax2.get_legend().remove()

# %% A3 top: Plot histogram + boxplots n_feats_opt
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False, figsize=(12.5, 4),
                         gridspec_kw={'height_ratios': [2, 6]})
fig.subplots_adjust(
top=0.99,
bottom=0.31,
left=0.05,
right=0.96,
hspace=0.04,
wspace=0.185)
for scenario, (row, col) in zip(scenario_options_top, grids):
    palette = sns.color_palette('muted', n_colors=len(df_opt_filter[scenario].unique()))

    ax1 = sns.histplot(ax=axes[row+1][col], data=df_opt_filter, x='n_feats_opt', alpha=0.3, color='#999999')
    ax1.set_ylim(bottom=0, top=330)
    ax1.yaxis.set_ticks(np.arange(0, 330, 100))
    if col >= 1:
        ax1.set(yticklabels=[], ylabel=None)
    else:
        ax1.set_ylabel('Count', weight='bold', fontsize='large')
        ax1.set_yticklabels(ax1.get_yticks(), weight='bold', fontsize='large')

    ax2 = ax1.twinx()
    ax2 = sns.kdeplot(ax=ax2, data=df_opt_filter, x='n_feats_opt', hue=scenario,
                      cut=0, bw_adjust=0.7, common_norm=True, common_grid=False,
                      label='temp', palette=palette)
    ax2.set_ylim(bottom=0)

    ax1.set_xlabel('Feature $\it{n}$ at optimum ' + '{0}'.format(obj),
                   fontweight='bold', fontsize='large')
    # set ticks visible, if using sharex = True. Not needed otherwise
    ax1.tick_params(labelbottom=True)
    ax1.xaxis.set_major_locator(mtick.FixedLocator(ax3.get_xticks()))
    ax1.set_xticklabels(['{:.0f}'.format(t) for t in ax2.get_xticks()],
                        fontweight='bold', fontsize='large')

    step_size = ax1.yaxis.get_ticklocs()[-1] - ax1.yaxis.get_ticklocs()[-2]
    extra = (ax1.get_ylim()[-1] - ax1.yaxis.get_ticklocs()[-1]) / step_size
    space = ax2.get_ylim()[-1] / (3.0 + extra)
    ax2.set_yticks(np.arange(ax3.get_ylim()[0], ax2.get_ylim()[-1]+(space*extra), space))
    ax2.grid(False)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold', rotation=90,
                        horizontalalignment='left', verticalalignment='center',
                        fontsize='large')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_ylabel('Density (%)', weight='bold', fontsize='large')

    l1 = list(df_opt_filter[scenario].unique())
    l2 = [scenario_options[scenario][uid] for uid in df_opt_filter[scenario].unique()]


    h, l1 = ax2.get_legend_handles_labels()
    h = list(reversed(h))
    ncol = 2 if len(h) > 4 else 1
    fontsize = 8.75 if len(h) > 3 else 'medium'
    label_color='#464646'
    color_grid = ax1.get_ygridlines()[0].get_color()
    leg = ax2.legend(h, l2,
                     bbox_to_anchor=(0, -0.24, 1, 0), loc='upper left',
                     mode='expand', ncol=ncol,
                     # fontsize=fontsize,
                     framealpha=0.85,
                     handletextpad=0.1,  # spacing between handle and label
                     columnspacing=0.5,
                     frameon=True,
                     edgecolor=color_grid,
                     prop={'weight':'bold',
                           'size': fontsize})

    # Finally, add boxplot now knowing the width
    box_width = 0.7 if len(h) > 4 else 0.65 if len(h) > 2 else 0.45
    ax3 = sns.boxplot(ax=axes[row][col], data=df_opt_filter, x='n_feats_opt', y=scenario,
                      width=box_width, fliersize=2, linewidth=1, palette=palette)
    ax3.set(yticklabels=[], ylabel=None)

# %% A3 bottom: Plot histogram + boxplots n_feats_opt
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False, figsize=(12.5, 5),
                         gridspec_kw={'height_ratios': [5, 6]})
fig.subplots_adjust(
top=0.99,
bottom=0.31,
left=0.05,
right=0.96,
hspace=0.04,
wspace=0.185)
for scenario, (row, col) in zip(scenario_options_bottom, grids):
    palette = sns.color_palette('muted', n_colors=len(df_opt_filter[scenario].unique()))

    ax1 = sns.histplot(ax=axes[row+1][col], data=df_opt_filter, x='n_feats_opt', alpha=0.3, color='#999999')
    ax1.set_ylim(bottom=0, top=330)
    ax1.yaxis.set_ticks(np.arange(0, 330, 100))
    if col >= 1:
        ax1.set(yticklabels=[], ylabel=None)
    else:
        ax1.set_ylabel('Count', weight='bold', fontsize='large')
        ax1.set_yticklabels(ax1.get_yticks(), weight='bold', fontsize='large')

    ax2 = ax1.twinx()
    ax2 = sns.kdeplot(ax=ax2, data=df_opt_filter, x='n_feats_opt', hue=scenario,
                      cut=0, bw_adjust=0.7, common_norm=True, common_grid=False,
                      label='temp', palette=palette)
    ax2.set_ylim(bottom=0)

    ax1.set_xlabel('Feature $\it{n}$ at optimum ' + '{0}'.format(obj),
                   fontweight='bold', fontsize='large')
    # set ticks visible, if using sharex = True. Not needed otherwise
    ax1.tick_params(labelbottom=True)
    ax1.xaxis.set_major_locator(mtick.FixedLocator(ax3.get_xticks()))
    ax1.set_xticklabels(['{:.0f}'.format(t) for t in ax2.get_xticks()],
                        fontweight='bold', fontsize='large')

    step_size = ax1.yaxis.get_ticklocs()[-1] - ax1.yaxis.get_ticklocs()[-2]
    extra = (ax1.get_ylim()[-1] - ax1.yaxis.get_ticklocs()[-1]) / step_size
    space = ax2.get_ylim()[-1] / (3.0 + extra)
    ax2.set_yticks(np.arange(ax3.get_ylim()[0], ax2.get_ylim()[-1]+(space*extra), space))
    ax2.grid(False)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold', rotation=90,
                        horizontalalignment='left', verticalalignment='center',
                        fontsize='large')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_ylabel('Density (%)', weight='bold', fontsize='large')

    l1 = list(df_opt_filter[scenario].unique())
    l2 = [scenario_options[scenario][uid] for uid in df_opt_filter[scenario].unique()]


    h, l1 = ax2.get_legend_handles_labels()
    h = list(reversed(h))
    ncol = 2 if len(h) > 4 else 1
    fontsize = 8.75 if len(h) > 3 else 'medium'
    label_color='#464646'
    color_grid = ax1.get_ygridlines()[0].get_color()
    leg = ax2.legend(h, l2,
                     bbox_to_anchor=(0, -0.24, 1, 0), loc='upper left',
                     mode='expand', ncol=ncol,
                     # fontsize=fontsize,
                     framealpha=0.85,
                     handletextpad=0.1,  # spacing between handle and label
                     columnspacing=0.5,
                     frameon=True,
                     edgecolor=color_grid,
                     prop={'weight':'bold',
                           'size': fontsize})

    # Finally, add boxplot now knowing the width
    box_width = 0.7 if len(h) > 4 else 0.65 if len(h) > 2 else 0.45
    ax3 = sns.boxplot(ax=axes[row][col], data=df_opt_filter, x='n_feats_opt', y=scenario,
                      width=box_width, fliersize=2, linewidth=1, palette=palette)
    ax3.set(yticklabels=[], ylabel=None)

# %% A0[not used]: ECDF for n_feats_opt
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(12.5, 6),
                         gridspec_kw={'height_ratios': [6, 6]})
fig.subplots_adjust(
top=0.998,
bottom=0.251,
left=0.051,
right=0.99,
hspace=0.825,
wspace=0.075)
for scenario, (row, col) in zip(scenario_options, grids_all):
    ax1 = sns.histplot(ax=axes[row][col], data=df_opt_filter, x='n_feats_opt', common_norm=False, cumulative=True, stat='density', alpha=0.3, color='#999999')
    palette=sns.color_palette('muted', n_colors=len(df_opt_filter[scenario].unique()))
    ax2 = sns.ecdfplot(ax=ax1, data=df_opt_filter, x='n_feats_opt', hue=scenario,
                      label='temp', palette=palette)
    ax1.set_ylim(bottom=0, top=1.05)
    # ax1.set_xlim(left=14, right=20)
    ax1.yaxis.set_ticks(np.arange(0, 1.05, 0.25))
    if col >= 1:
        ax1.set(yticklabels=[], ylabel=None)
        ax1.set_ylabel('')
    else:
        ax1.set_ylabel('Density', weight='bold', fontsize='large')
        ax1.set_yticklabels(ax1.get_yticks(), weight='bold', fontsize='large')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # legend
    h, _ = ax2.get_legend_handles_labels()
    h = list(reversed(h))
    l2 = [scenario_options[scenario][uid] for uid in df_opt_filter[scenario].unique()]
    ncol = 2 if len(h) > 4 else 1
    fontsize = 8.75 if len(h) > 3 else 'medium'
    label_color='#464646'
    color_grid = ax2.get_ygridlines()[0].get_color()
    leg = ax2.legend(h, l2,
                     bbox_to_anchor=(0, -0.3, 1, 0), loc='upper left',
                     mode='expand', ncol=ncol,
                     framealpha=0.85,
                     handletextpad=0.1,  # spacing between handle and label
                     columnspacing=0.5,
                     frameon=True,
                     edgecolor=color_grid,
                     prop={'weight':'bold',
                           'size': fontsize})
    width = 0.7 if len(h) > 4 else 0.6 if len(h) > 2 else 0.4
    # ax3 = sns.boxplot(ax=axes[row*2][col], data=df_opt_filter, x='value', y=scenario,
    #                   width=width, fliersize=2, linewidth=1, palette=palette)

    ax1.set_xlabel('Feature $\it{n}$ at optimum ' + '{0}'.format(obj),
                   fontweight='bold', fontsize='large')
    # set ticks visible, if using sharex = True. Not needed otherwise
    ax1.tick_params(labelbottom=True)
    ax1.xaxis.set_major_locator(mtick.FixedLocator(ax1.get_xticks()))
    ax1.set_xticklabels(['{:.0f}'.format(t) for t in ax1.get_xticks()],
                        fontweight='bold', fontsize='large')

# %% Fig 1: Get/organize NUP ground truth data

# base_dir_nup_data = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results\msi_2_000\nup_kgha'
# df_nup = pd.read_csv(os.path.join(base_dir_nup_data, 'msi_2_000_nup_kgha_data.csv'))

base_dir_nup_data = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results\msi_2_325\nup_kgha'
df_nup = pd.read_csv(os.path.join(base_dir_nup_data, 'msi_2_325_nup_kgha_data.csv'))

subset = ['dataset_id', 'study', 'date', 'plot_id', 'trt', 'rate_n_pp_kgha',
          'rate_n_sd_plan_kgha', 'rate_n_total_kgha', 'growth_stage', 'nup_kgha']
df_nup = df_nup[subset]

df_nup_stats = df_nup[['study', 'date', 'nup_kgha']].groupby(['study', 'date']).describe()
df_nup_stats.to_csv(os.path.join(r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2020_sip\data', 'nup_stats.csv'), index=True)

df_nup_stat_total = df_nup['nup_kgha'].describe()
df_nup_stat_total.to_csv(os.path.join(r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2020_sip\data', 'nup_stats_total.csv'), index=True)

# %% Fig 1: NUP histogram
dataset_dict = {
    0: 'Waseca whole field - 2019-06-29',
    1: 'Waseca whole field - 2019-07-08',
    2: 'Waseca whole field - 2019-07-23',
    3: 'Waseca small plot - 2019-06-29',
    4: 'Waseca small plot - 2019-07-09',
    5: 'Waseca small plot - 2019-07-23',
    6: 'Wells small plot - 2018-06-28',
    7: 'Wells small plot - 2019-07-08'}

fig, (ax_boxes, ax_box, ax1) = plt.subplots(3, sharex=True, figsize=(7, 5.5), gridspec_kw={"height_ratios": (5, 0.75, 5.25)})
fig.subplots_adjust(
top=0.808,
bottom=0.111,
left=0.120,
right=0.992,
hspace=0.05,
wspace=0.2)
palette = sns.color_palette('viridis', n_colors=len(df_nup['dataset_id'].unique())+5)
if len(df_nup['dataset_id'].unique()) == 7:
    hue_order = [6, 7, 3, 4, 0, 1, 2]
    ds_list = [0, 1, 2, 3, 4, 6, 7]
else:
    hue_order = [6, 7, 3, 4, 5, 0, 1, 2]
    n_order = [143, 144, 24, 24, 24, 16, 16, 16]
    ds_list = [0, 1, 2, 3, 4, 5, 6, 7]
    growthstage_ticks = [str(df_nup[df_nup['dataset_id'] == i]['growth_stage'].unique()[0]) + ' ({0})'.format(j) for i, j in zip(hue_order, n_order)]
    growthstage_ticks[1] = 'V8 (144)'
    # growthstage_ticks = [str(df_nup[df_nup['dataset_id'] == i]['growth_stage'].unique()[0]) + ' ($\it{n}$=' + '{0})'.format(j) for i, j in zip(hue_order, n_order)]
    # growthstage_ticks[1] = 'V8 ($\it{n}$=144)'

ax_boxes = sns.boxplot(ax=ax_boxes, data=df_nup.replace({'dataset_id': dataset_dict}),
                       x='nup_kgha', y='dataset_id',
                       order=[dataset_dict[i] for i in hue_order],
                        # order=hue_order,
                       width=0.7, fliersize=2, linewidth=1, palette=palette[2:-3])
ax_boxes.set_ylabel('')
ax_boxes.set_yticklabels(growthstage_ticks, fontweight='bold', fontsize='medium')
ax_boxes.set(xticklabels=[], xlabel=None)
for i, p in enumerate(ax_boxes.artists):
    p.set_alpha(0.8)
    p.set_edgecolor('#555555')
    for j in range(i*6, i*6+6):
        line = ax_boxes.lines[j]
        line.set_color('#555555')
        line.set_mfc('#555555')
        line.set_mec('#555555')

ax_box = sns.boxplot(x=df_nup['nup_kgha'], ax=ax_box, color='#C0C0C0', width=0.7, linewidth=1, fliersize=3)
for i, p in enumerate(ax_box.artists):
    p.set_alpha(0.8)
    p.set_edgecolor('#555555')
    for j in range(i*6, i*6+6):
        line = ax_box.lines[j]
        line.set_color('#555555')
        line.set_mfc('#555555')
        line.set_mec('#555555')
ax_box.set(xlabel='')
ax_box.set_yticklabels(['Total (407)'], fontweight='bold', fontsize=10.5)

ax1 = sns.histplot(ax=ax1, data=df_nup, x='nup_kgha', binwidth=3, multiple='stack', hue='dataset_id', hue_order=hue_order, palette=palette[2:-3], label='temp')
ax1.set_xlim(left=0)
ax1.set_xlabel('Nitrogen uptake ({0})'.format(units_print[response]), weight='bold', fontsize='x-large')
ax1.xaxis.set_major_locator(mtick.FixedLocator(ax1.get_xticks()))
ax1.set_xticklabels(['{:.0f}'.format(t) for t in ax1.get_xticks()],
                    fontweight='bold', fontsize='x-large')
ax1.set_ylabel('Count', weight='bold', fontsize='x-large')
ax1.yaxis.set_major_locator(mtick.FixedLocator(ax1.get_yticks()))
ax1.set_yticklabels(['{:.0f}'.format(t) for t in ax1.get_yticks()],
                    fontweight='bold', fontsize='x-large')

h, l1 = ax1.get_legend_handles_labels()
l2 = [dataset_dict[ds_id] for ds_id in hue_order]
h = list(reversed(h))
ncol = 2
fontsize = 12
label_color='#464646'
color_grid = ax1.get_ygridlines()[0].get_color()
leg = ax1.legend(h, l2,
                 bbox_to_anchor=(0, 2.15, 1, 0), loc='lower left',
                 mode='expand', ncol=ncol,
                 # fontsize=fontsize,
                 framealpha=0.85,
                 handletextpad=0.1,  # spacing between handle and label
                 columnspacing=0.5,
                 frameon=True,
                 edgecolor=color_grid,
                 prop={'weight':'bold',
                       'size': fontsize})

# %% Fig 2b: Spectral mimic demo using a single pixel from a Wells 2018 image
import os
from hs_process import hsio
from hs_process import spec_mod
from ast import literal_eval
from matplotlib.patches import Polygon

data_dir = r'F:\\nigo0024\Documents\hs_process_demo'
fname_hdr = os.path.join(data_dir, 'Wells_rep2_20180628_16h56m_pika_gige_7-Radiance Conversion-Georectify Airborne Datacube-Convert Radiance Cube to Reflectance from Measured Reference Spectrum.bip.hdr')

io = hsio()
io.read_cube(fname_hdr)
my_spec_mod = spec_mod(io.spyfile)

# Use spec_mod.spectral_mimic to mimic the Sentinel-2A spectral response function.
array_s2a, metadata_s2a = my_spec_mod.spectral_mimic(sensor='sentinel-2a', center_wl='weighted')
array_bin, metadata_bin = my_spec_mod.spectral_resample(bandwidth=20)

# Plot the mean spectral response of the hyperspectral image to that of the
# mimicked Sentinel-2A image bands (mean calculated across the entire image).
fwhm_s2a = [float(i) for i in metadata_s2a['fwhm'][1:-1].split(', ')]
spy_hs = my_spec_mod.spyfile.open_memmap()  # datacube before smoothing
meta_bands = list(io.tools.meta_bands.values())
meta_bands_s2a = sorted([float(i) for i in literal_eval(metadata_s2a['wavelength'])])
meta_bands_bin20 = sorted([float(i) for i in literal_eval(metadata_bin['wavelength'])])

# %% Fig 2b: Plot the spectral mimic demo
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
fig.subplots_adjust(
top=0.97,
bottom=0.14,
left=0.056,
right=0.99,
hspace=0.2,
wspace=0.178)
bands = [meta_bands_s2a, meta_bands_bin20]
arrays = [array_s2a, array_bin]
fwhms = [fwhm_s2a, [20]*(len(meta_bands_bin20)-1) + [(meta_bands_bin20[-1] - meta_bands_bin20[-2]) / 2]]
labels = ['Spectral "mimic" – Sentinel-2A',
          'Spectral "bin" - 20 nm']
for i, ax in enumerate(axes):
    if i == 0:
        zorders = [1, 1, 1, 1, 1, 1, 2, 1, 2]
        alpha=0.7
        ms=7
    else:
        zorders = [2]*len(bands[i])
        alpha=0.5
        ms=6

    ax1 = sns.lineplot(ax=ax, x=meta_bands, y=spy_hs[200][800]*100, label='Hyperspectral', linewidth=2, color=palette[2], zorder=0)
    ax2 = sns.lineplot(ax=ax1, x=bands[i], y=arrays[i][200][800]*100,
                       label=labels[i], linestyle='None',
                       marker='o', ms=ms, color=palette[0+i], zorder=2)

    ax1.set_ylim([0, 40])
    wedge_height = 1.5
    for wl, ref_pct, fwhm, zorder in zip(bands[i], arrays[i][200][800]*100, fwhms[i], zorders):
        wl_min = wl - (fwhm/2)
        wl_max = wl + (fwhm/2)
        verts = [(wl_min, 0), *zip([wl_min, wl, wl_max], [ref_pct-wedge_height, ref_pct, ref_pct-wedge_height]), (wl_max, 0)]
        poly = Polygon(verts, facecolor='0.9', edgecolor='0.5', alpha=alpha, zorder=zorder)
        ax2.add_patch(poly)

    ax1.set_xlabel('Wavelength (nm)', weight='bold', fontsize='large')
    ax1.set_ylabel('Reflectance (%)', weight='bold', fontsize='large')
    ax1.xaxis.set_major_locator(mtick.FixedLocator(ax1.get_xticks()))
    ax1.set_xticklabels(['{:.0f}'.format(t) for t in ax1.get_xticks()],
                        fontweight='bold', fontsize='large')
    ax1.yaxis.set_major_locator(mtick.FixedLocator(ax1.get_yticks()))
    ax1.set_yticklabels(['{:.0f}'.format(t) for t in ax1.get_yticks()],
                        fontweight='bold', fontsize='large')
    h, l1 = ax1.get_legend_handles_labels()
    leg = ax1.legend(h, l1,
                     framealpha=0.85,
                     handletextpad=0.5,  # spacing between handle and label
                     frameon=True,
                     edgecolor=ax1.get_ygridlines()[0].get_color(),
                     prop={'weight':'bold',
                           'size': 'medium'})


# %% Fig 3: Plot histogram + boxplots RMSE top
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False, figsize=(12.5, 4),
                         gridspec_kw={'height_ratios': [2, 6]})
fig.subplots_adjust(
top=0.99,
bottom=0.31,
left=0.05,
right=0.96,
hspace=0.04,
wspace=0.185)
for scenario, (row, col) in zip(scenario_options_top, grids):
    palette = sns.color_palette('muted', n_colors=len(df_opt_filter[scenario].unique()))

    ax2 = sns.histplot(ax=axes[row+1][col], data=df_opt_filter, x='value', alpha=0.3, color='#999999')
    ax2.yaxis.set_ticks(np.arange(0, 160, 50))
    if col >= 1:
        ax2.set(yticklabels=[], ylabel=None)
    else:
        ax2.set_ylabel('Count', weight='bold', fontsize='large')
        ax2.set_yticklabels(ax2.get_yticks(), weight='bold', fontsize='large')

    ax3 = ax2.twinx()
    ax3 = sns.kdeplot(ax=ax3, data=df_opt_filter, x='value', hue=scenario,
                      cut=0, bw_adjust=0.7, common_norm=True, common_grid=False,
                      label='temp', palette=palette)

    ax2.set_xlabel('{0} ({1})'.format(obj, units_print[response]),
                   fontweight='bold', fontsize='large')
    # set ticks visible, if using sharex = True. Not needed otherwise
    ax2.tick_params(labelbottom=True)
    ax2.xaxis.set_major_locator(mtick.FixedLocator(ax3.get_xticks()))
    ax2.set_xticklabels(['{:.0f}'.format(t) for t in ax2.get_xticks()],
                        fontweight='bold', fontsize='large')

    step_size = ax2.yaxis.get_ticklocs()[-1] - ax2.yaxis.get_ticklocs()[-2]
    extra = (ax2.get_ylim()[-1] - ax2.yaxis.get_ticklocs()[-1]) / step_size
    space = ax3.get_ylim()[-1] / (3.0 + extra)
    ax3.set_yticks(np.arange(ax3.get_ylim()[0], ax3.get_ylim()[-1]+(space*extra), space))
    ax3.grid(False)
    ax3.set_yticklabels(ax3.get_yticks(), weight='bold', rotation=90,
                        horizontalalignment='left', verticalalignment='center',
                        fontsize='large')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.set_ylabel('Density (%)', weight='bold', fontsize='large')

    l1 = list(df_opt_filter[scenario].unique())
    l2 = [scenario_options[scenario][uid] for uid in df_opt_filter[scenario].unique()]


    h, l1 = ax3.get_legend_handles_labels()
    h = list(reversed(h))
    ncol = 2 if len(h) > 4 else 1
    fontsize = 8.75 if len(h) > 3 else 'medium'
    label_color='#464646'
    color_grid = ax3.get_ygridlines()[0].get_color()
    leg = ax3.legend(h, l2,
                     bbox_to_anchor=(0, -0.24, 1, 0), loc='upper left',
                     mode='expand', ncol=ncol,
                     # fontsize=fontsize,
                     framealpha=0.85,
                     handletextpad=0.1,  # spacing between handle and label
                     columnspacing=0.5,
                     frameon=True,
                     edgecolor=color_grid,
                     prop={'weight':'bold',
                           'size': fontsize})

    # Finally, add boxplot now knowing the width
    box_width = 0.7 if len(h) > 4 else 0.65 if len(h) > 2 else 0.45
    ax1 = sns.boxplot(ax=axes[row][col], data=df_opt_filter, x='value', y=scenario,
                      width=box_width, fliersize=2, linewidth=1, palette=palette)
    ax1.set(yticklabels=[], ylabel=None)
    ax1.set_xlim(left=14, right=20)

# %% Fig 3: Plot histogram + boxplots RMSE bottom
fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False, figsize=(12.5, 5),
                         gridspec_kw={'height_ratios': [5, 6]})
fig.subplots_adjust(
top=0.99,
bottom=0.31,
left=0.05,
right=0.96,
hspace=0.04,
wspace=0.185)
for scenario, (row, col) in zip(scenario_options_bottom, grids):
    palette = sns.color_palette('muted', n_colors=len(df_opt_filter[scenario].unique()))

    ax2 = sns.histplot(ax=axes[row+1][col], data=df_opt_filter, x='value', alpha=0.3, color='#999999')
    ax2.yaxis.set_ticks(np.arange(0, 160, 50))
    if col >= 1:
        ax2.set(yticklabels=[], ylabel=None)
    else:
        ax2.set_ylabel('Count', weight='bold', fontsize='large')
        ax2.set_yticklabels(ax2.get_yticks(), weight='bold', fontsize='large')

    ax3 = ax2.twinx()
    ax3 = sns.kdeplot(ax=ax3, data=df_opt_filter, x='value', hue=scenario,
                      cut=0, bw_adjust=0.7, common_norm=True, common_grid=False,
                      label='temp', palette=palette)

    ax2.set_xlabel('{0} ({1})'.format(obj, units_print[response]),
                   fontweight='bold', fontsize='large')
    # set ticks visible, if using sharex = True. Not needed otherwise
    ax2.tick_params(labelbottom=True)
    ax2.xaxis.set_major_locator(mtick.FixedLocator(ax3.get_xticks()))
    ax2.set_xticklabels(['{:.0f}'.format(t) for t in ax2.get_xticks()],
                        fontweight='bold', fontsize='large')

    step_size = ax2.yaxis.get_ticklocs()[-1] - ax2.yaxis.get_ticklocs()[-2]
    extra = (ax2.get_ylim()[-1] - ax2.yaxis.get_ticklocs()[-1]) / step_size
    space = ax3.get_ylim()[-1] / (3.0 + extra)
    ax3.set_yticks(np.arange(ax3.get_ylim()[0], ax3.get_ylim()[-1]+(space*extra), space))
    ax3.grid(False)
    ax3.set_yticklabels(ax3.get_yticks(), weight='bold', rotation=90,
                        horizontalalignment='left', verticalalignment='center',
                        fontsize='large')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.set_ylabel('Density (%)', weight='bold', fontsize='large')

    l1 = list(df_opt_filter[scenario].unique())
    l2 = [scenario_options[scenario][uid] for uid in df_opt_filter[scenario].unique()]


    h, l1 = ax3.get_legend_handles_labels()
    h = list(reversed(h))
    ncol = 2 if len(h) > 4 else 1
    fontsize = 8.75 if len(h) > 3 else 'medium'
    label_color='#464646'
    color_grid = ax3.get_ygridlines()[0].get_color()
    leg = ax3.legend(h, l2,
                     bbox_to_anchor=(0, -0.28, 1, 0), loc='upper left',
                     mode='expand', ncol=ncol,
                     # fontsize=fontsize,
                     framealpha=0.85,
                     handletextpad=0.1,  # spacing between handle and label
                     columnspacing=0.5,
                     frameon=True,
                     edgecolor=color_grid,
                     prop={'weight':'bold',
                           'size': fontsize})

    # Finally, add boxplot now knowing the width
    box_width = 0.7 if len(h) > 4 else 0.65 if len(h) > 2 else 0.45
    ax1 = sns.boxplot(ax=axes[row][col], data=df_opt_filter, x='value', y=scenario,
                      width=box_width, fliersize=2, linewidth=1, palette=palette)
    ax1.set(yticklabels=[], ylabel=None)
    ax1.set_xlim(left=14, right=20)


# %% Fig 4: SALib functions
import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams

# seed = np.random.randint(0, 9999)
seed = 7818
def SALib_load_results():
    base_dir_salib = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta\salib'
    df = pd.read_csv(os.path.join(os.path.dirname(base_dir_salib), 'spotpy', 'results_nup_kgha_reflectance.csv'))
    Y = df['simulation_rmse'].to_numpy()
    df.rename(columns={'pardir_panels': 0,
                       'parcrop': 1,
                       'parclip': 2,
                       'parsmooth': 3,
                       'parbin': 4,
                       'parsegment': 5},
              inplace=True)
    return df, Y

def SALib_set_Y(param_values, df):
    Y = np.zeros(len(param_values))
    df['used'] = False
    for i, row in enumerate(param_values):
        df_filter = df[(df[0] == round(row[0])) &
                       (df[1] == round(row[1])) &
                       (df[2] == round(row[2])) &
                       (df[3] == round(row[3])) &
                       (df[4] == round(row[4])) &
                       (df[5] == round(row[5]))]
        if df.loc[df_filter.index[0], 'used'] == True:
            if df.loc[df_filter.index[1], 'used'] == True:  # get first one
                Y[i] = df.loc[df_filter.index[0], 'simulation_rmse']
            else:  # get second one
                Y[i] = df.loc[df_filter.index[0], 'simulation_rmse']
                df.loc[df_filter.index[1], 'used'] = True
        else:
            Y[i] = df.loc[df_filter.index[0], 'simulation_rmse']
            df.loc[df_filter.index[0], 'used'] = True
    print('Number of observations used: {0}'.format(len(df[df['used'] == True])))
    print('Number of observations NOT used: {0}'.format(len(df[df['used'] == False])))
    return df, Y

def SALib_get_problem(as_int=False):
    if as_int == True:
        problem = {
            'num_vars': 6,
            'names': ['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'],
            'bounds': [[0, 2],
                        [0, 2],
                        [0, 3],
                        [0, 2],
                        [0, 3],
                        [0, 9]]}
    else:
        problem = {
            'num_vars': 6,
            'names': ['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'],
            'bounds': [[0, 1],
                        [0, 1],
                        [0, 2],
                        [0, 1],
                        [0, 2],
                        [0, 8]]}

    return problem

scenario_options = {  # These indicate the legend labels
    'dir_panels': {
        'name': 'Reflectance panels',
        'closest': 'Closest panel',
        'all': 'All panels (mean)'},
    'crop': {
        'name': 'Crop',
        'plot_bounds': 'By plot boundary',
        'crop_buf': 'Edges cropped'},
    'clip': {
        'name': 'clip',
        'none': 'No spectral clipping',
        'ends': 'Ends clipped',
        'all': 'Ends + H2O and O2 absorption'},
    'smooth': {
        'name': 'Smooth',
        'none': 'No spectral smoothing',
        'sg-11': 'Savitzky-Golay smoothing'},
    'bin': {
        'name': 'Bin',
        'none': 'No spectral binning',
        'sentinel-2a_mimic': 'Spectral "mimic" - Sentinel-2A',
        'bin_20nm': 'Spectral "bin" - 20 nm'},
    'segment': {
        'name': 'Segment',
        'none': 'No segmenting',
        'ndi_upper_50': 'NDVI > 50th',
        'ndi_lower_50': 'NDVI < 50th',
        'mcari2_upper_50': 'MCARI2 > 50th',
        'mcari2_lower_50': 'MCARI2 < 50th',
        'mcari2_upper_90': 'MCARI2 > 90th',
        'mcari2_in_50-75': '50th > MCARI2 < 75th',
        'mcari2_in_75-95': '75th > MCARI2 < 95th',
        'mcari2_upper_90_green_upper_75': 'MCARI2 > 90th; green > 75th'},
    }

def get_df_results(Si, scenario_options, obj='rmse'):
    df_results = None
    for k, v in Si.items():
        sa_dict = {}
        sa_dict['step'] = list(scenario_options.keys())
        sa_dict['obj'] = [obj] * len(v)
        sa_dict['order'] = [k] * len(v)
        sa_dict['sensitivity_idx'] = v
        df_temp = pd.DataFrame.from_dict(sa_dict)
        if df_results is None:
            df_results = df_temp.copy()
        else:
            df_results = df_results.append(df_temp).reset_index(drop=True)
    return df_results

def plot_SA_bar(df_sa, ax1_str='S1', ax2_str='ST',
                ax1_title='First order', ax2_title='Total order',
                ylabel_str='Sensitivity Index',
                ax1_ylim=[0, 0.4], ax2_ylim=[0, 0.8]):
    # rcParams.update({'errorbar.capsize': 4})

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8, 4))
    fig.subplots_adjust(
    top=0.904,
    bottom=0.286,
    left=0.128,
    right=0.981,
    hspace=0.2,
    wspace=0.278)

    groups1 = df_sa[df_sa['order'] == ax1_str].groupby('step').sum()
    groups2 = df_sa[df_sa['order'] == ax2_str].groupby('step').sum()
    pal = sns.color_palette('Blues', len(groups1)+4)
    rank1 = groups1['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])
    rank2 = groups2['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])

    ax1 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa[df_sa['order'] == ax1_str],
                      # yerr=Si['S1_conf'], ecolor='#333333',
                      ax=axes[0], palette=np.array(pal[4:])[rank1])
    ax2 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa[df_sa['order'] == ax2_str],
                      # yerr=Si['ST_conf'], ecolor='#333333',
                      ax=axes[1], palette=np.array(pal[4:])[rank2])

    ax1.set_ylim(ax1_ylim[0], ax1_ylim[1])
    ax2.set_ylim(ax2_ylim[0], ax2_ylim[1])
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
    labels_tick = ['Reference panels', 'Crop', 'Clip', 'Smooth', 'Bin', 'Segment']
    for ax in [ax1, ax2]:
        ax.set_xlabel('', weight='bold', fontsize='large')
        ax.set_ylabel(ylabel_str, weight='bold', fontsize='large')
        ax.set_xticklabels(labels_tick, weight='bold', fontsize='medium')
        ax.set_yticklabels(['{0:.2f}'.format(x) for x in ax.get_yticks().tolist()], weight='bold', fontsize='medium')
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right',
                 rotation_mode='anchor')

    ax1.set_title(ax1_title, weight='bold', fontsize='x-large')
    ax2.set_title(ax2_title, weight='bold', fontsize='x-large')
    return fig, axes
# %% Fig 4: SALib SA FAST
from SALib.sample import fast_sampler
from SALib.analyze import fast

df, Y_file = SALib_load_results()
problem = SALib_get_problem()
param_values = fast_sampler.sample(problem, 1500, M=8, seed=seed)
df_FAST, Y = SALib_set_Y(param_values, df)

Si_FAST = fast.analyze(problem, Y, M=8, print_to_console=False, seed=seed)

df_sa_fast = get_df_results(Si_FAST, scenario_options, obj='rmse')

# %% Fig 4: SALib SA Sobel
from SALib.sample import saltelli
from SALib.analyze import sobol

df, Y = SALib_load_results()
param_values = saltelli.sample(problem, 400, calc_second_order=False, seed=seed)
param_values_round = param_values.round()
# param_values = saltelli.sample(problem, 162, calc_second_order=False, seed=seed).astype(int)
_, Y = SALib_set_Y(param_values_round, df)
problem = SALib_get_problem(as_int=False)
Si_sobol = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False, seed=seed)

# param_values_file = df[[0, 1, 2, 3, 4, 5]].to_numpy()

df_sa_sobol = get_df_results(Si_sobol, scenario_options, obj='rmse')
# df_sa_rmse = df_sa[df_sa['obj'] == 'rmse']

# %% Fig 4: Plot FAST, Sobol, and ranking for RMSE

def plot_FAST_Sobol_SA(df_sa_fast, df_sa_sobol,
                       ylabel_str='Sensitivity Index',
                       ax1_ylim=[0, 0.32], ax2_ylim=[0, 0.6]):
    # rcParams.update({'errorbar.capsize': 4})
    colors = sns.color_palette(['#a8a8a8', '#dcc457', '#57d4dc'])  # grey, gold, and cyan

    df_sa_fast['sa_type'] = 'fast'
    df_sa_sobol['sa_type'] = 'sobol'
    grp_fast_s1 = df_sa_fast[df_sa_fast['order'] == 'S1'].groupby('step').sum()
    grp_fast_st = df_sa_fast[df_sa_fast['order'] == 'ST'].groupby('step').sum()
    grp_sobol_s1 = df_sa_sobol[df_sa_sobol['order'] == 'S1'].groupby('step').sum()
    grp_sobol_st = df_sa_sobol[df_sa_sobol['order'] == 'ST'].groupby('step').sum()

    df_sa_fast_filter = df_sa_fast[df_sa_fast.order.isin(['S1','ST'])]
    df_sa = df_sa_fast_filter.append(df_sa_sobol[df_sa_sobol.order.isin(['S1','ST'])]).reset_index(drop=True)


    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(7.97, 4))
    fig.subplots_adjust(
    top=0.904,
    bottom=0.286,
    left=0.128,
    right=0.981,
    hspace=0.2,
    wspace=0.278)
    ax1 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa[df_sa['order'] == 'S1'],
                       hue='sa_type', ax=axes[0])
    ax2 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa[df_sa['order'] == 'ST'],
                       hue='sa_type', ax=axes[1])

    # pal_fast = sns.color_palette('Blues', len(grp_fast_s1)+8)
    # pal_sobol = sns.color_palette('YlOrBr', len(grp_sobol_s1)+16)
    pal_fast = sns.light_palette('#a8a8a8', len(grp_fast_s1)+3)
    pal_sobol = sns.light_palette('#597DBF', len(grp_sobol_s1)+3)  # blue

    # pal_fast = sns.light_palette('#dcc457', as_cmap=True)
    # pal_sobol = sns.light_palette('#57d4dc', as_cmap=True)

    rank_fast_s1 = grp_fast_s1['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])
    rank_fast_st = grp_fast_st['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])
    rank_sobol_s1 = grp_sobol_s1['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])
    rank_sobol_st = grp_sobol_st['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])

    # pal_fast_s1 = np.array(pal_fast[6:])[rank_fast_s1]
    # pal_fast_st = np.array(pal_fast[6:])[rank_fast_st]
    # pal_sobol_s1 = np.array(pal_sobol[7:])[rank_sobol_s1]
    # pal_sobol_st = np.array(pal_sobol[7:])[rank_sobol_st]
    pal_fast_s1 = np.array(pal_fast[3:])[rank_fast_s1]
    pal_fast_st = np.array(pal_fast[3:])[rank_fast_st]
    pal_sobol_s1 = np.array(pal_sobol[3:])[rank_sobol_s1]
    pal_sobol_st = np.array(pal_sobol[3:])[rank_sobol_st]
    for i, (p1, p2) in enumerate(zip(ax1.patches, ax2.patches)):
        if i <= 5:  # fast
            p1.set_color(pal_fast_s1[i])
            p2.set_color(pal_fast_st[i])
        else:  # sobol
            p1.set_color(pal_sobol_s1[i-6])
            p2.set_color(pal_sobol_st[i-6])

    ax1.set_ylim(ax1_ylim[0], ax1_ylim[1])
    ax2.set_ylim(ax2_ylim[0], ax2_ylim[1])
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5))


    labels_tick = ['Reference panels', 'Crop', 'Clip', 'Smooth', 'Bin', 'Segment']
    h, l = ax1.get_legend_handles_labels()
    for ax in [ax1, ax2]:
        ax.legend(handles=h, labels=['FAST', 'Sobol'], prop={'weight': 'bold'})
        # ax.legend(handles=label_list[0], labels=label_list[1])
        # labels_leg = ax.get_legend().get_texts()
        # labels_leg[0].set_text('FAST')
        # labels_leg[1].set_text('Sobol')
        ax.set_xlabel('', weight='bold', fontsize='large')
        ax.set_ylabel(ylabel_str, weight='bold', fontsize='large')
        ax.set_xticklabels(labels_tick, weight='bold', fontsize='medium')
        ax.set_yticklabels(['{0:.2f}'.format(x) for x in ax.get_yticks().tolist()], weight='bold', fontsize='medium')
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right',
                 rotation_mode='anchor')
    ax2.get_legend().remove()

    ax1.set_title('First order', weight='bold', fontsize='x-large')
    ax2.set_title('Total order', weight='bold', fontsize='x-large')
    return fig, axes

# %% Fig 4: Plot FAST, Sobol, and ranking for RMSE and modify
fig, axes = plot_FAST_Sobol_SA(df_sa_fast, df_sa_sobol,
                               ylabel_str='Sensitivity Index',
                               ax1_ylim=[0, 0.32], ax2_ylim=[0, 0.6])

# %% Fig 4b: Step importance via stepwise regression
df_stepwise = pd.read_csv(r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\var_importance_stepwise_all.csv')
df_stepwise['Variable Importance'] = df_stepwise['Variable Importance'] * 100
grp_res_var = df_stepwise[df_stepwise['method'] == 'residual_var'].groupby('step').sum()

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(3.5, 3.5))
fig.subplots_adjust(
top=0.992,
bottom=0.283,
left=0.259,
right=0.99,
hspace=0.2,
wspace=0.2)
ax1 = sns.barplot(x='step', y='Variable Importance', data=df_stepwise[df_stepwise['method'] == 'residual_var'],
                  order=['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'], ax=ax)

pal_res_var = sns.light_palette('#597DBF', len(grp_res_var)+3)  # blue
rank_res_var = grp_res_var['Variable Importance'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])

pal_res_var = np.array(pal_res_var[3:])[rank_res_var]
for i, p1 in enumerate(ax.patches):
    p1.set_color(pal_res_var[i])

# ax.set_ylim(ax_ylim[0], ax_ylim[1])
ax.yaxis.set_major_locator(plt.MaxNLocator(4))

labels_tick = ['Reference panels', 'Crop', 'Clip', 'Smooth', 'Bin', 'Segment']
h, l = ax.get_legend_handles_labels()
ax.set_xlabel('', weight='bold', fontsize='large')
ax.set_ylabel('Variable Importance (%)', weight='bold', fontsize='large')
ax.set_xticklabels(labels_tick, weight='bold', fontsize='medium')
ax.set_yticklabels(['{0:.0f}'.format(x) for x in ax.get_yticks().tolist()], weight='bold', fontsize='medium')
# ax.set_yticklabels(['{0:.2f}'.format(x) for x in mtick.FixedLocator(ax.get_yticks()).tolist()], weight='bold', fontsize='medium')
# ax.yaxis.set_major_locator(mtick.FixedLocator(ax.get_yticks()))
plt.setp(ax.get_xticklabels(), rotation=35, ha='right',
         rotation_mode='anchor')

# ax.get_legend().remove()

# %% Fig 5: Curate stats for cumulative density plots
classes = [14, 15, 16, 17, 18, 19]
array_props = {step: np.empty([len(scenario_options[step]), len(classes)]) for step in scenario_options}
array_props['segment'] = np.delete(array_props['segment'], (-1), axis=0)
cols = ['step', 'scenario', 'class_min_val', 'proportion']
df_props = None
for i_steps, step in enumerate(scenario_options):
    for i_scenarios, scenario in enumerate(scenario_options[step]):
        if scenario == 'name':
            continue
        df_scenario = df_opt_filter[df_opt_filter[step] == scenario]
        for i_classes, min_val in enumerate(classes):
            prop = (len(df_scenario[df_scenario['value'].between(min_val, min_val+1)]) / len(df_scenario)) * 100
            array_props[step][i_scenarios-1, i_classes] = prop
            data = [step, scenario, min_val, prop]
            if df_props is None:
                df_props = pd.DataFrame([data], columns=cols)
            else:
                df_props = df_props.append(pd.DataFrame([data], columns=cols))

figsize_top = (6, 1.5)
figsize_bottom = (6, 2.5)

# %% Fig 5: Heat plot dir_panels
step = 'dir_panels'
df_props_pivot = df_props[df_props['step'] == step].pivot('scenario', 'class_min_val', 'proportion')
grid = dict(height_ratios=[1]*(len(scenario_options[step])-1), width_ratios=[6])
fig, axes = plt.subplots(ncols=1, nrows=len(scenario_options[step])-1, figsize=figsize_top, gridspec_kw=grid)
fig.subplots_adjust(
    top=1.0,
    bottom=0.0,
    left=-0.001,
    right=1.0,
    hspace=0.0,
    wspace=0.2)
for i, scenario in enumerate(scenario_options[step]):
    if scenario == 'name':
        continue
    ax = sns.heatmap(
        ax=axes[i-1], data=df_props_pivot.loc[[scenario]],
        annot=True, fmt='.1f', annot_kws={'weight': 'bold', 'fontsize': 16},
        linewidths=4, yticklabels=False,
        cbar=False, cmap=sns.light_palette(palette[i-1], as_cmap=True))
    for t in ax.texts: t.set_text('') if float(t.get_text()) == 0.0 else t.set_text(t.get_text() + '%')
    ax.set_xlabel('')
    ax.set_ylabel('')

# %% Fig 5: Heat plot crop
step = 'crop'
df_props_pivot = df_props[df_props['step'] == step].pivot('scenario', 'class_min_val', 'proportion')
grid = dict(height_ratios=[1]*(len(scenario_options[step])-1), width_ratios=[6])
fig, axes = plt.subplots(ncols=1, nrows=len(scenario_options[step])-1, figsize=figsize_top, gridspec_kw=grid)
fig.subplots_adjust(
    top=1.0,
    bottom=0.0,
    left=-0.001,
    right=1.0,
    hspace=0.0,
    wspace=0.2)
for i, scenario in enumerate(scenario_options[step]):
    if scenario == 'name':
        continue
    ax = sns.heatmap(
        ax=axes[i-1], data=df_props_pivot.loc[[scenario]],
        annot=True, fmt='.1f', annot_kws={'weight': 'bold', 'fontsize': 16},
        linewidths=4, yticklabels=False,
        cbar=False, cmap=sns.light_palette(palette[i-1], as_cmap=True))
    for t in ax.texts: t.set_text('') if float(t.get_text()) == 0.0 else t.set_text(t.get_text() + '%')
    ax.set_xlabel('')
    ax.set_ylabel('')

# %% Fig 5: Heat plot clip (3x top)
step = 'clip'
df_props_pivot = df_props[df_props['step'] == step].pivot('scenario', 'class_min_val', 'proportion')
grid = dict(height_ratios=[1]*(len(scenario_options[step])-1), width_ratios=[6])
fig, axes = plt.subplots(ncols=1, nrows=len(scenario_options[step])-1, figsize=figsize_top, gridspec_kw=grid)
fig.subplots_adjust(
    top=1.0,
    bottom=0.0,
    left=-0.001,
    right=1.0,
    hspace=0.0,
    wspace=0.2)
for i, scenario in enumerate(scenario_options[step]):
    if scenario == 'name':
        continue
    ax = sns.heatmap(
        ax=axes[i-1], data=df_props_pivot.loc[[scenario]],
        annot=True, fmt='.1f', annot_kws={'weight': 'bold', 'fontsize': 16},
        linewidths=4, yticklabels=False,
        cbar=False, cmap=sns.light_palette(palette[i-1], as_cmap=True))
    for t in ax.texts: t.set_text('') if float(t.get_text()) == 0.0 else t.set_text(t.get_text() + '%')
    ax.set_xlabel('')
    ax.set_ylabel('')

# %% Fig 5: Heat plot smooth
step = 'smooth'
df_props_pivot = df_props[df_props['step'] == step].pivot('scenario', 'class_min_val', 'proportion')
grid = dict(height_ratios=[1]*(len(scenario_options[step])-1), width_ratios=[6])
fig, axes = plt.subplots(ncols=1, nrows=len(scenario_options[step])-1, figsize=figsize_bottom, gridspec_kw=grid)
fig.subplots_adjust(
    top=1.0,
    bottom=0.0,
    left=-0.001,
    right=1.0,
    hspace=0.0,
    wspace=0.2)
for i, scenario in enumerate(scenario_options[step]):
    if scenario == 'name':
        continue
    ax = sns.heatmap(
        ax=axes[i-1], data=df_props_pivot.loc[[scenario]],
        annot=True, fmt='.1f', annot_kws={'weight': 'bold', 'fontsize': 16},
        linewidths=4, yticklabels=False,
        cbar=False, cmap=sns.light_palette(palette[i-1], as_cmap=True))
    for t in ax.texts: t.set_text('') if float(t.get_text()) == 0.0 else t.set_text(t.get_text() + '%')
    ax.set_xlabel('')
    ax.set_ylabel('')

# %% Fig 5: Heat plot bin
step = 'bin'
df_props_pivot = df_props[df_props['step'] == step].pivot('scenario', 'class_min_val', 'proportion')
grid = dict(height_ratios=[1]*(len(scenario_options[step])-1), width_ratios=[6])
fig, axes = plt.subplots(ncols=1, nrows=len(scenario_options[step])-1, figsize=figsize_bottom, gridspec_kw=grid)
fig.subplots_adjust(
    top=1.0,
    bottom=0.0,
    left=-0.001,
    right=1.0,
    hspace=0.0,
    wspace=0.2)
for i, scenario in enumerate(scenario_options[step]):
    if scenario == 'name':
        continue
    ax = sns.heatmap(
        ax=axes[i-1], data=df_props_pivot.loc[[scenario]],
        annot=True, fmt='.1f', annot_kws={'weight': 'bold', 'fontsize': 16},
        linewidths=4, yticklabels=False,
        cbar=False, cmap=sns.light_palette(palette[i-1], as_cmap=True))
    for t in ax.texts: t.set_text('') if float(t.get_text()) == 0.0 else t.set_text(t.get_text() + '%')
    ax.set_xlabel('')
    ax.set_ylabel('')

# %% Fig 5: Heat plot segment
step = 'segment'
df_props_pivot = df_props[df_props['step'] == step].pivot('scenario', 'class_min_val', 'proportion')
grid = dict(height_ratios=[1]*(len(scenario_options[step])-1), width_ratios=[6])
fig, axes = plt.subplots(ncols=1, nrows=len(scenario_options[step])-1, figsize=figsize_bottom, gridspec_kw=grid)
fig.subplots_adjust(
    top=1.0,
    bottom=0.0,
    left=-0.0,
    right=1.0,
    hspace=0.0,
    wspace=0.0)
for i, scenario in enumerate(scenario_options[step]):
    if scenario == 'name':
        continue
    ax = sns.heatmap(
        ax=axes[i-1], data=df_props_pivot.loc[[scenario]],
        annot=True, fmt='.1f', annot_kws={'weight': 'bold', 'fontsize': 12},
        linewidths=4, yticklabels=False,
        cbar=False, cmap=sns.light_palette(palette[i-1], as_cmap=True))
    for t in ax.texts: t.set_text('') if float(t.get_text()) == 0.0 else t.set_text(t.get_text() + '%')
    # for t in ax.texts: t.set_text(t.get_text() + '%')
    ax.set_xlabel('')
    ax.set_ylabel('')

# %% Fig 5: Plot cumulative density functions

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(12.5, 6),
                         gridspec_kw={'height_ratios': [6, 6]})
fig.subplots_adjust(
top=0.998,
bottom=0.251,
left=0.051,
right=0.99,
hspace=0.825,
wspace=0.075)
for scenario, (row, col) in zip(scenario_options, grids_all):
    ax1 = sns.histplot(ax=axes[row][col], data=df_opt_filter, x='value', common_norm=False, cumulative=True, stat='density', alpha=0.3, color='#999999')
    palette=sns.color_palette('muted', n_colors=len(df_opt_filter[scenario].unique()))
    ax2 = sns.ecdfplot(ax=ax1, data=df_opt_filter, x='value', hue=scenario,
                      label='temp', palette=palette)
    ax1.set_ylim(bottom=0, top=1.05)
    ax1.set_xlim(left=14, right=20)
    ax1.yaxis.set_ticks(np.arange(0, 1.05, 0.25))
    if col >= 1:
        ax1.set(yticklabels=[], ylabel=None)
        ax1.set_ylabel('')
    else:
        ax1.set_ylabel('Density', weight='bold', fontsize='large')
        ax1.set_yticklabels(ax1.get_yticks(), weight='bold', fontsize='large')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # legend
    h, _ = ax2.get_legend_handles_labels()
    h = list(reversed(h))
    l2 = [scenario_options[scenario][uid] for uid in df_opt_filter[scenario].unique()]
    ncol = 2 if len(h) > 4 else 1
    fontsize = 8.75 if len(h) > 3 else 'medium'
    label_color='#464646'
    color_grid = ax2.get_ygridlines()[0].get_color()
    leg = ax2.legend(h, l2,
                     bbox_to_anchor=(0, -0.3, 1, 0), loc='upper left',
                     mode='expand', ncol=ncol,
                     framealpha=0.85,
                     handletextpad=0.1,  # spacing between handle and label
                     columnspacing=0.5,
                     frameon=True,
                     edgecolor=color_grid,
                     prop={'weight':'bold',
                           'size': fontsize})
    width = 0.7 if len(h) > 4 else 0.6 if len(h) > 2 else 0.4
    # ax3 = sns.boxplot(ax=axes[row*2][col], data=df_opt_filter, x='value', y=scenario,
    #                   width=width, fliersize=2, linewidth=1, palette=palette)

    ax1.set_xlabel('{0} ({1})'.format(obj, units_print[response]),
                    fontweight='bold', fontsize='large')
    # set ticks visible, if using sharex = True. Not needed otherwise
    ax1.tick_params(labelbottom=True)
    ax1.xaxis.set_major_locator(mtick.FixedLocator(ax2.get_xticks()))
    ax1.set_xticklabels(['{:.0f}'.format(t) for t in ax2.get_xticks()],
                        fontweight='bold', fontsize='large')

# %% Fig 5b: Show only two segment lines
# for _ in range(len(ax2.lines)-2):
#     # ax2.lines[0].remove()
for i, l in enumerate(ax2.lines):
    if i == 7 or i == 8:
        continue
    else:
        l.set_alpha(.2)
# %% Fig 6: Calculate percentile of
from scipy import stats

# Nigon et al. (2020) scenario:
idx = 140
subset_nigon_2020 = df_opt_filter[df_opt_filter['grid_idx'] == idx][['grid_idx', 'dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment']].drop_duplicates()
rmse_last_paper = df_opt_filter[df_opt_filter['grid_idx'] == idx]['value'].mean()
stats.percentileofscore(df_opt_filter['value'], rmse_last_paper)

a = df_opt_filter[
    ['grid_idx', 'dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment', 'value']].groupby(
        ['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment']).mean()
idx_min = a[a['value'] == a['value'].min()]['grid_idx'].values[0]
subset_min_rmse = df_opt_filter[df_opt_filter['grid_idx'] == idx_min][['grid_idx', 'dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment']].drop_duplicates()
rmse_min = df_opt_filter[df_opt_filter['grid_idx'] == idx_min]['value'].mean()
stats.percentileofscore(df_opt_filter['value'], rmse_min)

idx_max = a[a['value'] == a['value'].max()]['grid_idx'].values[0]
subset_max_rmse = df_opt_filter[df_opt_filter['grid_idx'] == idx_max][['grid_idx', 'dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment']].drop_duplicates()
rmse_max = df_opt_filter[df_opt_filter['grid_idx'] == idx_max]['value'].mean()
stats.percentileofscore(df_opt_filter['value'], rmse_max)

subset_max_rmse = df_opt_filter[df_opt_filter['value'] == df_opt_filter['value'].max()][['grid_idx', 'dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment']]
rmse_max = df_opt_filter[df_opt_filter['grid_idx'] == subset_max_rmse['grid_idx'].values[0]]['value'].mean()
stats.percentileofscore(df_opt_filter['value'], rmse_max)

# %% Results: Calculate percentile of threshold
stats.percentileofscore(df_opt_filter[df_opt_filter['dir_panels'] == 'closest']['value'], 15)
stats.percentileofscore(df_opt_filter[df_opt_filter['dir_panels'] == 'all']['value'], 15)

# %% Results: Segment min/max/range
df_segment_describe = df_opt_filter.groupby('segment')['value'].describe()
df_segment_describe.to_csv(r'F:\nigo0024\Dropbox\UMN\UMN_Publications\2020_sip\data\segment_stats.csv', index=True)

# %% Results: Calculate RRMSE for best and worst scenarios
import pandas as pd

fname1 = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results\msi_2_521\nup_kgha\reflectance\testing\msi_2_521_nup_kgha_test-preds-lasso.csv'
df_b_las = pd.read_csv(fname1)
df_b_las['nup_kgha'].mean()

df_opt_filter['rrmse'] = df_opt_filter['value'] / df_b_las['nup_kgha'].mean() * 100

df_opt_filter[df_opt_filter['grid_idx'] == 472]['rrmse'].mean()  # worst
df_opt_filter[df_opt_filter['grid_idx'] == 521]['rrmse'].mean()  # best

# %% Plot cumulative density functions

fig, axes = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=False, figsize=(12.5, 9),
                         gridspec_kw={'height_ratios': [6, 2.5, 6, 6]})
fig.subplots_adjust(
# top row of plots
# top=0.993,
# bottom=0.061,
# left=0.051,
# right=0.990,
# hspace=0.50,
# wspace=0.185)
# bottom row of plots
top=0.993,
bottom=0.061,
left=0.051,
right=0.990,
hspace=0.805,
wspace=0.075)
for scenario, (row, col) in zip(scenario_options, grids):
    ax1 = sns.histplot(ax=axes[row*2][col], data=df_opt_filter, x='value', common_norm=False, cumulative=True, stat='density', alpha=0.3, color='#999999')
    palette=sns.color_palette('muted', n_colors=len(df_opt_filter[scenario].unique()))
    ax2 = sns.ecdfplot(ax=ax1, data=df_opt_filter, x='value', hue=scenario,
                      label='temp', palette=palette)
    ax1.set_ylim(bottom=0, top=1.05)
    ax1.set_xlim(left=14, right=20)
    ax1.yaxis.set_ticks(np.arange(0, 1.05, 0.25))
    if col >= 1:
        ax1.set(yticklabels=[], ylabel=None)
        ax1.set_ylabel('')
    else:
        ax1.set_ylabel('Density', weight='bold', fontsize='large')
        ax1.set_yticklabels(ax1.get_yticks(), weight='bold', fontsize='large')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # legend
    h, _ = ax2.get_legend_handles_labels()
    h = list(reversed(h))
    l2 = [scenario_options[scenario][uid] for uid in df_opt_filter[scenario].unique()]
    ncol = 2 if len(h) > 4 else 1
    fontsize = 8.75 if len(h) > 3 else 'medium'
    label_color='#464646'
    color_grid = ax2.get_ygridlines()[0].get_color()
    leg = ax2.legend(h, l2,
                     bbox_to_anchor=(0, 0, 1, 0), loc='upper left',
                     mode='expand', ncol=ncol,
                     framealpha=0.85,
                     handletextpad=0.1,  # spacing between handle and label
                     columnspacing=0.5,
                     frameon=True,
                     edgecolor=color_grid,
                     prop={'weight':'bold',
                           'size': fontsize})
    # ax2.get_legend().remove()

    width = 0.7 if len(h) > 4 else 0.6 if len(h) > 2 else 0.4
    ax3 = sns.boxplot(ax=axes[row*2+1][col], data=df_opt_filter, x='value', y=scenario,
                      width=width, fliersize=2, linewidth=1, palette=palette)

    ax3.set_xlabel('{0} ({1})'.format(obj, units_print[response]),
                    fontweight='bold', fontsize='large')
    # set ticks visible, if using sharex = True. Not needed otherwise
    # for tick in ax3.get_xticklabels():
    #     tick.set_visible(True)
    ax3.tick_params(labelbottom=True)
    ax3.xaxis.set_major_locator(mtick.FixedLocator(ax3.get_xticks()))
    ax3.set_xticklabels(['{:.0f}'.format(t) for t in ax3.get_xticks()],
                        fontweight='bold', fontsize='large')
    # for tick in ax3.get_xticklabels():
    #     tick.set_visible(True)


    ax3.set(yticklabels=[], ylabel=None)
    # ax3.set_ylabel('{0}'.format(scenario_options[scenario]['name']), fontsize='large', fontweight='bold')
    ax3.set_ylabel('', fontsize='large', fontweight='bold')

    # ncol = 2 if len(h) > 3 else 2 if len(h) > 2 else 1
    # ncol = 2 if len(h) > 4 else 1
    # fontsize = 8.75 if len(h) > 3 else 'medium'
    # label_color='#464646'
    # color_grid = ax1.get_ygridlines()[0].get_color()
    # leg = ax3.legend(h, l2,
    #                  bbox_to_anchor=(0, 1, 1, 0), loc='lower left',
    #                  mode='expand', ncol=ncol,
    #                  framealpha=0.85,
    #                  handletextpad=0.1,  # spacing between handle and label
    #                  columnspacing=0.5,
    #                  frameon=True,
    #                  edgecolor=color_grid,
    #                  prop={'weight':'bold',
    #                        'size': fontsize})
    # ax2.get_legend().remove()
# just save figure manually

# %% Number of observations
import os
import pandas as pd

base_dir_results = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta'
df_obs = pd.read_csv(os.path.join(base_dir_results, 'msi_2_n_observations.csv'))
df_grid = pd.read_csv(os.path.join(base_dir_results, 'msi_2_hs_settings_short.csv'))

df_obs = df_grid.merge(df_obs, left_index=True, on='grid_idx')


arrays = [np.array(['dir_panels', 'dir_panels', 'crop', 'crop', 'clip', 'clip', 'clip',
                    'smooth', 'smooth', 'bin', 'bin', 'bin',
                    'segment', 'segment', 'segment', 'segment', 'segment', 'segment', 'segment', 'segment', 'segment']),
          np.array(list(df_obs['dir_panels'].unique()) +
                   list(df_obs['crop'].unique()) +
                   list(df_obs['clip'].unique()) +
                   list(df_obs['smooth'].unique()) +
                   list(df_obs['bin'].unique()) +
                   list(df_obs['segment'].unique()))]

df_a = pd.DataFrame(np.zeros((21, 0)), index=pd.MultiIndex.from_arrays(arrays, names=('step', 'scenario')))

for scenario in ['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment']:
    df_a.loc[pd.IndexSlice[scenario, :], 'median'] = list(df_obs[[scenario, 'obs_n']].groupby([scenario])['obs_n'].median())
    df_a.loc[pd.IndexSlice[scenario, :], 'min'] = list(df_obs[[scenario, 'obs_n']].groupby([scenario])['obs_n'].min())
    df_a.loc[pd.IndexSlice[scenario, :], 'max'] = list(df_obs[[scenario, 'obs_n']].groupby([scenario])['obs_n'].max())
    df_a.loc[pd.IndexSlice[scenario, :], 'std'] = list(df_obs[[scenario, 'obs_n']].groupby([scenario])['obs_n'].std())
    df_a.loc[pd.IndexSlice[scenario, :], 'mean'] = list(df_obs[[scenario, 'obs_n']].groupby([scenario])['obs_n'].mean())


df = df_obs[(df_obs['dir_panels'] == 'closest') &
            (df_obs['segment'] == 'mcari2_upper_90_green_upper_75')]
# %% Spotpy Sensitivity analysis
import os
from spotpy import analyser

base_dir_spotpy = r'G:\BBE\AGROBOT\Shared Work\hs_process_results\results\msi_2_results_meta\spotpy'
options = [[response, feature, obj] for response in ['biomass_kgha', 'nup_kgha', 'tissue_n_pct']
           for feature in ['reflectance', 'derivative_1', 'derivative_2']
           for obj in ['MAE', 'RMSE', 'R2']]

# for response, feature, obj in options:
#     break
response = 'nup_kgha'
feature = 'reflectance'
obj = 'RMSE'

results = analyser.load_csv_results(
    os.path.join(base_dir_spotpy, 'results_{0}_{1}'.format(response, feature)))

# %% FAST sensitivity for RMSE, MAE, and R2
sa_rmse = analyser.get_sensitivity_of_fast(results, like_index=1)
sa_mae = analyser.get_sensitivity_of_fast(results, like_index=2)
sa_r2 = analyser.get_sensitivity_of_fast(results, like_index=3)

df_sa = None
for d, obj in zip([sa_rmse, sa_mae, sa_r2], ['rmse', 'mae', 'r2']):
    sa_dict = {}
    for k, v in d.items():
        sa_dict['step'] = list(scenario_options.keys())
        sa_dict['obj'] = [obj] * len(v)
        sa_dict['order'] = [k] * len(v)
        sa_dict['sensitivity_idx'] = v
        df_temp = pd.DataFrame.from_dict(sa_dict)
        if df_sa is None:
            df_sa = df_temp.copy()
        else:
            df_sa = df_sa.append(df_temp).reset_index(drop=True)

df_sa_rmse = df_sa[df_sa['obj'] == 'rmse']

# %% Plot FAST 1st and total order SA for RMSE
import seaborn as sns
from matplotlib import pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10, 5))
fig.subplots_adjust(
top=0.936,
bottom=0.233,
left=0.137,
right=0.995,
hspace=0.2,
wspace=0.391)

groups1 = df_sa_rmse[df_sa_rmse['order'] == 'S1'].groupby('step').sum()
groups2 = df_sa_rmse[df_sa_rmse['order'] == 'ST'].groupby('step').sum()
pal = sns.color_palette('Blues', len(groups1)+4)
rank1 = groups1['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])
rank2 = groups2['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])

ax1 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa_rmse[df_sa_rmse['order'] == 'S1'], ax=axes[0], palette=np.array(pal[4:])[rank1])
ax2 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa_rmse[df_sa_rmse['order'] == 'ST'], ax=axes[1], palette=np.array(pal[4:])[rank2])

ax1.set_ylim(0, 0.2)
ax2.set_ylim(0, 0.6)
ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
labels_tick = ['Reference panels', 'Crop', 'Clip', 'Smooth', 'Bin', 'Segment']
for ax in [ax1, ax2]:
    ax.set_xlabel('', weight='bold', fontsize='large')
    ax.set_ylabel('FAST sensitivity index', weight='bold', fontsize='large')
    ax.set_xticklabels(labels_tick, weight='bold', fontsize='medium')
    ax.set_yticklabels(['{0:.2f}'.format(x) for x in ax.get_yticks().tolist()], weight='bold', fontsize='medium')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right',
             rotation_mode='anchor')

ax1.set_title('First order', weight='bold', fontsize='x-large')
ax2.set_title('Total order', weight='bold', fontsize='x-large')

# %% Plot SALib SA FAST
fig, axes = plot_SA_bar(
    df_sa_fast, ax1_str='S1', ax2_str='ST',
    ax1_title='First order', ax2_title='Total order',
    ylabel_str='FAST sensitivity index',
    ax1_ylim=[0, 0.24], ax2_ylim=[0, 0.6])

# import seaborn as sns
# from matplotlib import pyplot as plt
# from matplotlib import rcParams

# rcParams.update({'errorbar.capsize': 4})

# fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8, 5))
# fig.subplots_adjust(
# top=0.923,
# bottom=0.229,
# left=0.128,
# right=0.981,
# hspace=0.2,
# wspace=0.354)

# groups1 = df_sa_fast[df_sa_fast['order'] == 'S1'].groupby('step').sum()
# groups2 = df_sa_fast[df_sa_fast['order'] == 'ST'].groupby('step').sum()
# pal = sns.color_palette('Blues', len(groups1)+4)
# rank1 = groups1['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])
# rank2 = groups2['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])

# ax1 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa_fast[df_sa_fast['order'] == 'S1'],
#                   # yerr=Si['S1_conf'], ecolor='#333333',
#                   ax=axes[0], palette=np.array(pal[4:])[rank1])
# ax2 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa_fast[df_sa_fast['order'] == 'ST'],
#                   # yerr=Si['ST_conf'], ecolor='#333333',
#                   ax=axes[1], palette=np.array(pal[4:])[rank2])

# ax1.set_ylim(0, 0.3)
# ax2.set_ylim(0, 0.6)
# ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
# ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
# labels_tick = ['Reference panels', 'Crop', 'Clip', 'Smooth', 'Bin', 'Segment']
# for ax in [ax1, ax2]:
#     ax.set_xlabel('', weight='bold', fontsize='large')
#     ax.set_ylabel('FAST sensitivity index', weight='bold', fontsize='large')
#     ax.set_xticklabels(labels_tick, weight='bold', fontsize='medium')
#     ax.set_yticklabels(['{0:.2f}'.format(x) for x in ax.get_yticks().tolist()], weight='bold', fontsize='medium')
#     plt.setp(ax.get_xticklabels(), rotation=35, ha='right',
#              rotation_mode='anchor')

# ax1.set_title('First order', weight='bold', fontsize='x-large')
# ax2.set_title('Total order', weight='bold', fontsize='x-large')


# %% Plot Sobol SA for RMSE

fig, axes = plot_SA_bar(
    df_sa_sobol, ax1_str='S1', ax2_str='ST',
    ax1_title='First order', ax2_title='Total order',
    ylabel_str="Sobol sensitivity index",
    ax1_ylim=[0, 0.4], ax2_ylim=[0, 0.6])

# rcParams.update({'errorbar.capsize': 4})

# fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8, 5))
# fig.subplots_adjust(
# top=0.923,
# bottom=0.229,
# left=0.128,
# right=0.981,
# hspace=0.2,
# wspace=0.354)

# groups1 = df_sa_sobol[df_sa_sobol['order'] == 'S1'].groupby('step').sum()
# groups2 = df_sa_sobol[df_sa_sobol['order'] == 'ST'].groupby('step').sum()
# pal = sns.color_palette('Blues', len(groups1)+4)
# rank1 = groups1['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])
# rank2 = groups2['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])

# ax1 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa_sobol[df_sa_sobol['order'] == 'S1'],
#                   # yerr=Si['S1_conf'], ecolor='#333333',
#                   ax=axes[0], palette=np.array(pal[4:])[rank1])
# ax2 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa_sobol[df_sa_sobol['order'] == 'ST'],
#                   # yerr=Si['ST_conf'], ecolor='#333333',
#                   ax=axes[1], palette=np.array(pal[4:])[rank2])

# ax1.set_ylim(0, 0.4)
# ax2.set_ylim(0, 0.6)
# ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
# ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
# labels_tick = ['Reference panels', 'Crop', 'Clip', 'Smooth', 'Bin', 'Segment']
# for ax in [ax1, ax2]:
#     ax.set_xlabel('', weight='bold', fontsize='large')
#     ax.set_ylabel("Sobel' sensitivity index", weight='bold', fontsize='large')
#     ax.set_xticklabels(labels_tick, weight='bold', fontsize='medium')
#     ax.set_yticklabels(['{0:.2f}'.format(x) for x in ax.get_yticks().tolist()], weight='bold', fontsize='medium')
#     plt.setp(ax.get_xticklabels(), rotation=35, ha='right',
#              rotation_mode='anchor')

# ax1.set_title('First order', weight='bold', fontsize='x-large')
# ax2.set_title('Total order', weight='bold', fontsize='x-large')

# %% SALib SA Morris
from SALib import sample
from SALib.analyze import morris
from SALib.analyze import delta
from SALib.analyze import ff

df, Y = SALib_load_results()
problem = SALib_get_problem(as_int=False)
param_values = sample.morris.sample(problem, 400, num_levels=1000, seed=seed)
param_values_round = param_values.round()
# param_values = saltelli.sample(problem, 162, calc_second_order=False, seed=seed).astype(int)
_, Y = SALib_set_Y(param_values_round, df)
Si_morris = morris.analyze(problem, param_values_round, Y,
                           num_resamples=100,
                           conf_level=0.95,
                           print_to_console=False,
                           num_levels=1000,
                           seed=seed)
df_sa_morris = get_df_results(Si_morris, scenario_options, obj='rmse')

Si_delta = delta.analyze(
    problem, param_values_round, Y, num_resamples=100, conf_level=0.95,
    print_to_console=False, seed=seed)
df_sa_delta = get_df_results(Si_delta, scenario_options, obj='rmse')


# param_values = sample.ff.sample(problem, seed=seed)
# param_values_round = param_values.round()
# _, Y = SALib_set_Y(param_values_round, df)
# Si_ff = ff.analyze(problem, param_values_round, Y, second_order=True, print_to_console=False)

# %% Plot Delta Moment-Independent Measure SA for RMSE
fig, axes = plot_SA_bar(
    df_sa_delta, ax1_str='S1', ax2_str='delta',
    ax1_title='S1', ax2_title='Delta',
    ylabel_str='DM-IM sensitivity index',
    ax1_ylim=[0, 0.24], ax2_ylim=[0, 0.24])

# fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8, 5))
# fig.subplots_adjust(
# top=0.923,
# bottom=0.229,
# left=0.128,
# right=0.981,
# hspace=0.2,
# wspace=0.354)

# groups1 = df_sa_delta[df_sa_delta['order'] == 'S1'].groupby('step').sum()
# groups2 = df_sa_delta[df_sa_delta['order'] == 'delta'].groupby('step').sum()
# pal = sns.color_palette('Blues', len(groups1)+4)
# rank1 = groups1['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])
# rank2 = groups2['sensitivity_idx'].sort_values().argsort().reindex(['dir_panels', 'crop', 'clip', 'smooth', 'bin', 'segment'])

# ax1 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa_delta[df_sa_delta['order'] == 'S1'],
#                   # yerr=Si['S1_conf'], ecolor='#333333',
#                   ax=axes[0], palette=np.array(pal[4:])[rank1])
# ax2 = sns.barplot(x='step', y='sensitivity_idx', data=df_sa_delta[df_sa_delta['order'] == 'delta'],
#                   # yerr=Si['ST_conf'], ecolor='#333333',
#                   ax=axes[1], palette=np.array(pal[4:])[rank2])

# ax1.set_ylim(0, 0.3)
# ax2.set_ylim(0, 0.2)
# ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
# ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
# labels_tick = ['Reference panels', 'Crop', 'Clip', 'Smooth', 'Bin', 'Segment']
# for ax in [ax1, ax2]:
#     ax.set_xlabel('', weight='bold', fontsize='large')
#     ax.set_ylabel('DM-IM sensitivity index', weight='bold', fontsize='large')
#     ax.set_xticklabels(labels_tick, weight='bold', fontsize='medium')
#     ax.set_yticklabels(['{0:.2f}'.format(x) for x in ax.get_yticks().tolist()], weight='bold', fontsize='medium')
#     plt.setp(ax.get_xticklabels(), rotation=35, ha='right',
#              rotation_mode='anchor')

# ax1.set_title('S1', weight='bold', fontsize='x-large')
# ax2.set_title('Delta', weight='bold', fontsize='x-large')


# %% Plot Morris SA for RMSE
fig, axes = plot_SA_bar(
    df_sa_morris, ax1_str='mu_star', ax2_str='sigma',
    ax1_title='Mu_star', ax2_title='Sigma',
    ylabel_str='Morris sensitivity index',
    ax1_ylim=[0, 2], ax2_ylim=[0, 2])