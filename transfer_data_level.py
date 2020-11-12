# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:23:45 2020

@author: nigo0024
"""

# In[Import libraries]
from sip_functions import *
import globus_sdk

CLIENT_ID = ''
TRANSFER_TOKEN = ''
TRANSFER_REFRESH_TOKEN = ''
msi_run_id = None
idx_grid = None
level = 'segment'

# In[Get arguments]
if __name__ == "__main__":  # required on Windows, so just do on all..
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--CLIENT_ID',
                        help='Globus client ID; can be found at https://auth.globus.org/v2/web/developers on the "MSI SIP" app')
    parser.add_argument('-t', '--TRANSFER_TOKEN',
                        help='Transfer access token.')
    parser.add_argument('-r', '--TRANSFER_REFRESH_TOKEN',
                        help='Transfer refresh token.')
    parser.add_argument('-m', '--msi_run_id',
                        help='The MSI run ID; use 0 to run on local machine.')
    parser.add_argument('-i', '--idx_grid',
                        help='Row index from df_grid to transfer image data for.')
    parser.add_argument('-l', '--level',
                        help='Directory level to transfer data for; must be one '
                        'of ["segment", "smooth", "clip", "crop"].')
    args = parser.parse_args()
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
    if args.msi_run_id is not None:
        msi_run_id = eval(args.msi_run_id)
    else:
        msi_run_id = 0
    if args.idx_grid is not None:
        idx_grid = eval(args.idx_grid)
    else:
        idx_grid = None
    if args.level is not None:
        level = args.level.lower()

    # In[Prep I/O]
    if msi_run_id > 0:  # be sure to set keyrings
        dir_base = '/panfs/roc/groups/5/yangc1/public/hs_process'
    else:
        msi_run_id = 0
        dir_base = r'G:\BBE\AGROBOT\Shared Work\hs_process_results'

    dir_data = os.path.join(dir_base, 'data')
    dir_results = os.path.join(dir_base, 'results')
    dir_results_msi = os.path.join(dir_results, 'msi_' + str(msi_run_id) + '_results')
    dir_results_meta = os.path.join(dir_results, 'msi_' + str(msi_run_id) + '_results_meta')

    df_grid = pd.read_csv(os.path.join(dir_results_meta, 'msi_' + str(msi_run_id) + '_hs_settings.csv'), index_col=0)
    df_grid = clean_df_grid(df_grid)

    row = df_grid.loc[idx_grid]
    print_details(row)
    label_base = 'idx_grid_' + str(idx_grid).zfill(3)

    client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    dir_source_data, dir_dest_data = get_globus_data_dir(
        dir_base, msi_run_id, row, level=level)
    transfer_result, delete_result = globus_transfer(
        dir_source_data, dir_dest_data, TRANSFER_REFRESH_TOKEN, client, TRANSFER_TOKEN,
        label=label_base + '-data', delete=True)
