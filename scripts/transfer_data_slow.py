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
idx_min = 0
idx_max = 1

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

    parser.add_argument('-i', '--idx_min',
                        help='Minimum idx to consider in df_grid')
    parser.add_argument('-d', '--idx_max',
                        help='Minimum idx to consider in df_grid')
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

    if args.idx_min is not None:
        idx_min = eval(args.idx_min)
    else:
        idx_min = 0
    if args.idx_max is not None:
        idx_max = eval(args.idx_max)
    else:
        idx_max = idx_min + 1

    # In[Prep I/O]
    msi_run_id = 1
    dir_base = '/panfs/roc/groups/5/yangc1/public/hs_process'

    dir_data = os.path.join(dir_base, 'data')
    dir_results = os.path.join(dir_base, 'results')
    dir_results_msi = os.path.join(dir_results, 'msi_' + str(msi_run_id) + '_results')

    df_grid = pd.read_csv(os.path.join(dir_results, 'msi_' + str(msi_run_id) + '_hs_settings.csv'), index_col=0)
    df_grid = clean_df_grid(df_grid)

    for idx_grid, row in df_grid.iterrows():
        # if idx_grid >= 2:
        #     break
        if idx_grid < idx_min:
            print('Skipping past idx_grix {0}...'.format(idx_grid))
            continue
        if idx_grid >= idx_max:
            sys.exit('All processing scenarios are finished. Exiting program.')
        print_details(row)
        label_base = 'idx_grid_' + str(idx_grid).zfill(3)

        # tier2_data_transfer(dir_base, row)
        # tier2_results_transfer(dir_base, folder_list_test[0])  # msi_result_dir
        client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
        dir_source_data, dir_dest_data = get_globus_data_dir(
            dir_base, msi_run_id, row)
        transfer_result, delete_result = globus_transfer(
            dir_source_data, dir_dest_data, TRANSFER_REFRESH_TOKEN, client, TRANSFER_TOKEN,
            label=label_base + '-data', delete=True)

        dir_source_results, dir_dest_results = get_globus_results_dir(
            dir_base, msi_run_id, row)
        transfer_result, delete_result = globus_transfer(
            dir_source_results, dir_dest_results, TRANSFER_REFRESH_TOKEN, client,
            TRANSFER_TOKEN, label=label_base + '-results', delete=True)