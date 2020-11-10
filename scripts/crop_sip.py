# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:35:32 2020

@author: nigo0024
"""

# In[Load data]
from sip_functions import *

dir_base = r'G:\BBE\AGROBOT\Shared Work\hs_process_results'
dir_data = os.path.join(dir_base, 'data')

# In[Crop ref_closest_panel-crop_plot]
dir_panels = 'ref_closest_panel'
crop_type = 'crop_plot'
n_files = 835  # ref_closest_panel has 24 less images because aerfsmall-20190723 did not have any reference panels in its images

dir_out_crop = os.path.join(dir_data, dir_panels, crop_type)
df_crop, gdf_wells, gdf_aerf = retrieve_files(
    dir_data, panel_type=dir_panels, crop_type=crop_type)
print('Cropping {0}-{1}..'.format(dir_panels, crop_type))
crop(df_crop, panel_type=dir_panels, dir_out_crop=dir_out_crop,
     out_force=False, n_files=n_files, gdf_aerf=gdf_aerf, gdf_wells=gdf_wells)

# In[Crop ref_closest_panel-crop_buf]
dir_panels = 'ref_closest_panel'
crop_type = 'crop_buf'
n_files = 835

dir_out_crop = os.path.join(dir_data, dir_panels, crop_type)
df_crop, gdf_wells, gdf_aerf = retrieve_files(
    dir_data, panel_type=dir_panels, crop_type=crop_type)
print('Cropping {0}-{1}..'.format(dir_panels, crop_type))
crop(df_crop, panel_type=dir_panels, dir_out_crop=dir_out_crop,
     out_force=False, n_files=n_files, gdf_aerf=gdf_aerf, gdf_wells=gdf_wells)

# In[Crop ref_closest_panel-crop_plot MISSING]
dir_panels = 'ref_closest_panel'
crop_type = 'crop_plot'
n_files = 835

dir_out_crop = os.path.join(dir_data, dir_panels, crop_type)
_, gdf_wells, gdf_aerf = retrieve_files(
    dir_data, panel_type=dir_panels, crop_type=crop_type)

fname_crop_info = os.path.join(dir_data, 'crop_plot_missing.csv')
df_crop = pd.read_csv(fname_crop_info)
df_crop['date'] = pd.to_datetime(df_crop['date'])
df_crop['directory'] = df_crop.apply(lambda row : os.path.join(
    row['directory'], dir_panels), axis = 1)

print('Cropping {0}-{1}..'.format(dir_panels, crop_type))
crop(df_crop, panel_type=dir_panels, dir_out_crop=dir_out_crop,
     out_force=False, n_files=n_files, gdf_aerf=gdf_aerf, gdf_wells=gdf_wells)


# In[]
dir_panels = 'ref_all_panels'
crop_type = 'crop_plot'
n_files = 859
