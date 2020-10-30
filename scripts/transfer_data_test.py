# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:39:25 2020

@author: nigo0024
"""

# In[Import libraries]
from sip_functions import *
import globus_sdk
import boto3

def s3_delete_dir(bucket='hs_process', prefix='test_globus_transfer/', recursive=True):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    if prefix[-1] != '/':
        prefix += '/'
    bucket.objects.filter(Prefix=prefix).delete()

CLIENT_ID = ''
TRANSFER_TOKEN = ''
TRANSFER_REFRESH_TOKEN = ''

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
    parser.add_argument('-m', '--msi_test',
                        help='Whether to run on MSI (True) or on local machin (False).')

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
    if args.msi_test is not None:
        msi_test = args.msi_test
    else:
        msi_test = True

    # In[Prep I/O]
    if msi_test is True or msi_test == 'True':  # be sure to set keyrings
        msi_run_id = 1
        dir_base = '/panfs/roc/groups/5/yangc1/public/hs_process'
    else:
        msi_run_id = 0
        dir_base = r'G:\BBE\AGROBOT\Shared Work\hs_process_results'

    name = 'test_globus_transfer'
    dir_source_test = os.path.join(dir_base, name + '/')
    fname_test = os.path.join(dir_source_test, name + '.txt')

    if not os.path.isdir(dir_source_test):
        pathlib.Path(dir_source_test).mkdir(parents=True, exist_ok=True)
    with open(fname_test, "w") as file:
        file.write('This file was created as part of a test to ensure GLOBUS '
                   'client credentials are valid. This file can be deleted.')

    client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    dir_dest_test = '/hs_process/test_globus_transfer/'
    dir_dest_test = '/'.join(
        ('/' + os.path.basename(dir_base), name + '/'))
    transfer_result, delete_result = globus_transfer(
        dir_source_test, dir_dest_test, TRANSFER_REFRESH_TOKEN, client, TRANSFER_TOKEN,
        label=name, delete=True)

    # Delete test file from tier2
    authorizer = globus_sdk.RefreshTokenAuthorizer(
        TRANSFER_REFRESH_TOKEN, client, access_token=TRANSFER_TOKEN)
    tc = globus_sdk.TransferClient(authorizer=authorizer)
    # dest_endpoint = tc.endpoint_search(filter_fulltext='umnmsi#tier2')
    tier2_id = 'fb6f1c6b-86b1-11e8-9571-0a6d4e044368'
    dest_endpoint = tc.get_endpoint(tier2_id)

    print('Waiting for transfer/delete {0} to complete...'
          ''.format(delete_result['task_id']))
    c = it.count(1)
    while not tc.task_wait(delete_result['task_id'], timeout=60):
        print('Transfer/delete {0} has not yet finished; transfer submitted {1} '
              'minute(s) ago'.format(transfer_result['task_id'], next(c)))
    print('DONE.')
    # ddata = globus_sdk.DeleteData(tc, dest_endpoint, label=name + '_delete_tier2', recursive=True)
    # ddata.add_item(dir_dest_test)
    # delete_result_t2 = tc.submit_delete(ddata)
    # print("GLOBUS tier2 DELETE task_id:\n", delete_result_t2['task_id'])

    if transfer_result['code'] == 'Accepted':
        print('(1 of 2): Test folder successfully transferred from MSI HPC '
              'storage to tier2 storage.')
    if delete_result['code'] == 'Accepted':
        print('(2 of 2): Test folder successfully deleted from MSI HPC storage.')

    # s3_delete_dir(bucket='hs_process', prefix=name+'/', recursive=True)
    # if delete_result_t2['code'] == 'Accepted':
    #     print('(3 of 3): Test folder successfully deleted from tier2 storage.')
