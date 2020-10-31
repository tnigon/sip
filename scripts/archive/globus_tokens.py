# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:38:38 2020

@author: nigo0024
"""

import globus_sdk

# In[Getting started - get transfer tokens]
# See Globus-SDK Python: https://globus-sdk-python.readthedocs.io/

CLIENT_ID = '5d573462-2134-4900-970d-6e7a5e0f2b1e'  # can be found at https://auth.globus.org/v2/web/developers on the "MSI SIP" app

client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
client.oauth2_start_flow()
client.oauth2_start_flow(refresh_tokens=True)

print('Please go to this URL and login: {0}'
      .format(client.oauth2_get_authorize_url()))

# In[Get unique transfer tokens]
# auth_code = 'nF7HmyJJU55vq0QK01YDL8MXy8doAR'.strip()  # March 2020
auth_code = 'kX0NyF4tVcVlG4JCs3yedKZJal9BaZ'.strip()  # May 2020

token_response = client.oauth2_exchange_code_for_tokens(auth_code)
str(token_response.by_resource_server)

globus_auth_data = token_response.by_resource_server['auth.globus.org']
globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']

AUTH_TOKEN = globus_auth_data['access_token']
TRANSFER_TOKEN = globus_transfer_data['access_token']  # transfer access token
TRANSFER_REFRESH_TOKEN = globus_transfer_data['refresh_token']  # transfer refresh token

expires_at_s = globus_transfer_data['expires_at_seconds']

# treat these like passwords
AUTH_TOKEN = 'AgxJrp6OqbgDY01KBXN0DNpr3bKkQ3qVw1jXEKNgEQ7oMnXNYeT8C1Xzn3nVvxKlk1bXXEodPxJvrXIqDw26JHa66NtgVVDCarrd'
TRANSFER_TOKEN = 'AgBXVNMKXoOKa6XBlympD0pVKq3EkXxl03NkGB56YPqweBayGeFyClPv0n86GkyPK7PP0mgNM4GqNCk32vwoclEEN'
TRANSFER_REFRESH_TOKEN = 'AgxdYn0G06XqDYKmy21kpjmr7xkoP3kdBdrVzPOYag6z3vXp6eCeUNX4Pb4qwa4x9oW5O8DPo3KN684yoNKp05pKrGmyQ'

keyring.set_password('CLIENT_ID', 'nigo0024', '684eb60a-9c5e-48af-929d-0880fd829173')
keyring.set_password('globus_transfer_access_token', 'nigo0024', 'Aggwn2QG9WQb7v5QmBe1KbgBDmBKd3mEwXx3wYQxdQDg80xnoXSJCx57ywM1qpV6NNgPJOPWgdMYYJT1Q5Bg2flGOG')
keyring.set_password('globus_transfer_refresh_token', 'nigo0024', 'AgeQ96Dj433owed0EK799GkB4jNbl5xkWDrMjjqgQzVbNoPBKHqUM1bGngqy5DB0elpkv637z3qM8pMy27BMJdxyeE5n')

# keyring set CLIENT_ID nigo0024 684eb60a-9c5e-48af-929d-0880fd829173
# keyring set globus_transfer_access_token nigo0024 Aggwn2QG9WQb7v5QmBe1KbgBDmBKd3mEwXx3wYQxdQDg80xnoXSJCx57ywM1qpV6NNgPJOPWgdMYYJT1Q5Bg2flGOG
# keyring set globus_transfer_refresh_token nigo0024 AgeQ96Dj433owed0EK799GkB4jNbl5xkWDrMjjqgQzVbNoPBKHqUM1bGngqy5DB0elpkv637z3qM8pMy27BMJdxyeE5n

# In[View available endpoints]
authorizer = globus_sdk.RefreshTokenAuthorizer(
    TRANSFER_REFRESH_TOKEN, client, access_token=TRANSFER_TOKEN)
# authorizer = globus_sdk.AccessTokenAuthorizer(TRANSFER_TOKEN)  # TRANSFER_TOKEN is only valid for a limited time. You’ll have to login again when it expires.
tc = globus_sdk.TransferClient(authorizer=authorizer)

# high level interface; provides iterators for list responses
print("My Endpoints:")  # these are my "private" endpoints on globus.org
for ep in tc.endpoint_search(filter_scope="my-endpoints"):
    print("[{}] {}".format(ep["id"], ep["display_name"]))

for ep in tc.endpoint_search('umnmsi#tier2', num_results=10):
    print(ep['display_name'])

# In[Transfer data]
authorizer = globus_sdk.RefreshTokenAuthorizer(
    TRANSFER_REFRESH_TOKEN, client, access_token=TRANSFER_TOKEN, expires_at=expires_at_s)
tc = globus_sdk.TransferClient(authorizer=authorizer)

# msi_endpoint = 'd865fc6a-2db3-11e6-8070-22000b1701d1'
# tier2_endpoint = 'fb6f1c6b-86b1-11e8-9571-0a6d4e044368'

# msi_endpoint = 'cf388dcc-7aaa-11ea-9698-0afc9e7dd773'
# tier2_endpoint = 'd9f66f04-7aaa-11ea-9698-0afc9e7dd773'

label = 'SDK example'
submission_id = None

# see: https://globus-sdk-python.readthedocs.io/en/stable/clients/transfer/#helper-objects
tdata = globus_sdk.TransferData(
    tc, source_endpoint=msi_endpoint, destination_endpoint=tier2_endpoint,
    label=label, submission_id=submission_id,
    sync_level=2, verify_checksum=False, preserve_timestamp=False,
    encrypt_data=False, deadline=None, recursive_symlinks='ignore')

dir_str_source = '/home/yangc1/public/hs_process/results/msi_1_results/'
dir_str_dest = 'hs_process/results/msi_1_results'

tdata.add_item(dir_str_source, dir_str_dest, recursive=True)

transfer_result = tc.submit_transfer(tdata)
print("task_id =", transfer_result["task_id"])



# In[Auto-activate endpoints (add credentials)]
authorizer = globus_sdk.RefreshTokenAuthorizer(
    TRANSFER_REFRESH_TOKEN, client, access_token=TRANSFER_TOKEN, expires_at=expires_at_s)
tc = globus_sdk.TransferClient(authorizer=authorizer)

# msi_endpoint = 'd865fc6a-2db3-11e6-8070-22000b1701d1'
# tier2_endpoint = 'fb6f1c6b-86b1-11e8-9571-0a6d4e044368'
msi_endpoint = tc.endpoint_search(filter_fulltext='umnmsi#home')
tier2_endpoint = tc.endpoint_search(filter_fulltext='umnmsi#tier2')

r = tc.endpoint_autoactivate(msi_endpoint, if_expires_in=3600)
while (r["code"] == "AutoActivationFailed"):
    print("Endpoint requires manual activation, please open "
          "the following URL in a browser to activate the "
          "endpoint:")
    print("https://app.globus.org/file-manager?origin_id=%s"
          % msi_endpoint)
    # For python 2.X, use raw_input() instead
    input("Press ENTER after activating the endpoint:")
    r = tc.endpoint_autoactivate(msi_endpoint, if_expires_in=3600)


r = tc.endpoint_autoactivate(tier2_endpoint, if_expires_in=3600)
while (r["code"] == "AutoActivationFailed"):
    print("Endpoint requires manual activation, please open "
          "the following URL in a browser to activate the "
          "endpoint:")
    print("https://app.globus.org/file-manager?origin_id=%s"
          % msi_endpoint)
    # For python 2.X, use raw_input() instead
    input("Press ENTER after activating the endpoint:")
    r = tc.endpoint_autoactivate(msi_endpoint, if_expires_in=3600)

r = tc.endpoint_autoactivate(msi_endpoint, if_expires_in=3600)
if r['code'] == 'AutoActivationFailed':
    print('Endpoint({}) Not Active! Error! Source message: {}'
          .format(msi_endpoint, r['message']))
    sys.exit(1)
elif r['code'] == 'AutoActivated.CachedCredential':
    print('Endpoint({}) autoactivated using a cached credential.'
          .format(msi_endpoint))
elif r['code'] == 'AutoActivated.GlobusOnlineCredential':
    print(('Endpoint({}) autoactivated using a built-in Globus '
           'credential.').format(msi_endpoint))
elif r['code'] == 'AlreadyActivated':
    print('Endpoint({}) already active until at least {}'
          .format(msi_endpoint, 3600))


# In[Create endpoint for MSI]
authorizer = globus_sdk.RefreshTokenAuthorizer(
    TRANSFER_REFRESH_TOKEN, client, access_token=TRANSFER_TOKEN)
tc = globus_sdk.TransferClient(authorizer=authorizer)

ep_data_msi = {
  "DATA_TYPE": "endpoint",
  # "id": "d865fc6a-2db3-11e6-8070-22000b1701d1",
  "display_name": "umnmsi#home",
  "organization": "University of Minnesota",
  # "owner_string": "umnmsi@globusid.org",
  # "subscription_id": "d865fc6a-2db3-11e6-8070-22000b1701d1",
  "default_directory": "/~/",
  "myproxy_server": 'walibb40h303627.msi.umn.edu:7512',
  "myproxy_dn": '/C=US/O=Globus Consortium/OU=Globus Connect Service/CN=4a7e26e0-16f1-11ea-b94c-0e16720bb42f',
  "DATA": [
    {
      "DATA_TYPE": "server",
      "hostname": 'walibb40h303620.msi.umn.edu',
      "uri": "gsiftp://walibb40h303620.msi.umn.edu:2811",
      "port": 2811,
      "scheme": "gsiftp",
      # "id": 985,
      "subject": "/C=US/O=Globus Consortium/OU=Globus Connect Service/CN=aee0715c-16f0-11ea-ab3a-0a7959ea6081"
    },
  ],
}
create_result = tc.create_endpoint(ep_data_msi)
msi_endpoint = create_result["id"]

ep_data_tier2 = {
  "DATA_TYPE": "endpoint",
  # "id": "fb6f1c6b-86b1-11e8-9571-0a6d4e044368",
  "display_name": "umnmsi#tier2",
  "organization": "University of Minnesota",
  # "owner_id": "nigo0024@umn.edu",
  # "subscription_id": "fb6f1c6b-86b1-11e8-9571-0a6d4e044368",
  "default_directory": "/",
  "myproxy_server": 'walibb40h303622.msi.umn.edu',
  "myproxy_dn": '/C=US/O=Globus Consortium/OU=Globus Connect Service/CN=d4a3f292-f4f4-11e9-be92-02fcc9cdd752',
  "DATA": [
    {
      "DATA_TYPE": "server",
      "hostname": 'walibb40h303622.msi.umn.edu',
      "uri": "gsiftp://walibb40h303622.msi.umn.edu:2811",
      "port": 2811,
      "scheme": "gsiftp",
      # "id": 985,
      "subject": "/C=US/O=Globus Consortium/OU=Globus Connect Service/CN=d4a3f292-f4f4-11e9-be92-02fcc9cdd752"
    },
  ],
}
create_result = tc.create_endpoint(ep_data_tier2)
tier2_endpoint = create_result["id"]

# delete endpoint
# tc.delete_endpoint(tier2_endpoint)







