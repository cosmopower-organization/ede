import argparse
import numpy as np
import re
import yaml
import os
from pkg_resources import resource_filename
import time
from time import gmtime
from time import strftime
import pprint
import copy
from itertools import groupby
from sklearn.model_selection import train_test_split
import pandas as pd

nsamps_stop = 500000 # decrease just for testing
num_groups = 1000
all_groups = np.arange(1,num_groups+1)
nonempty_groups = all_groups[100:110] # set e.g. nonempty_groups[0:1] for testing

frac_training = 0.8
test_size = 1.-frac_training
derived_params_names = ['100*theta_s',  # replace h by '100*theta_s' oct 26 22
                        'sigma8',
                        'YHe',
                        'z_reio',
                        'Neff',
                        'tau_rec',
                        'z_rec',
                        'rs_rec',
                        'ra_rec',
                        'tau_star',
                        'z_star',
                        'rs_star',
                        'ra_star',
                        'rs_drag',# added rs drag for baos 21oct22
                        'age', ## added 24 nov 23 
                        'log10m_scf', ## added 24 nov 23
                        'log10f_scf' ## added 31 may 24
                           ] 
nder = len(derived_params_names)
print(f'we have {nder} derived parameters')


cosmo_params = ['logA',
                'n_s',
                'H0',
                'omega_b',
                'omega_cdm',
                'tau_reio',
                'm_ncdm',
                'N_ur',
                'thetai_scf',
                'log10z_c',
                'fEDE',
                'r',
               ] # oct 26: change thetas to H0 -- dec 9: added m_ncdm, nov 23 23: added eded and Nur

npars = len(cosmo_params)
print(f'we have {npars} input parameters')

nscalars = npars+nder
print(f'In total we have {npars+nder} scalar parameters stored at the start of each data vector')

path_to_data_file = '/mnt/ceph/users/bbolliet/cosmopower_training_data/ACTPol_lite_DR4_baseLCDM_taup_hip_bao_pks_ede-final_500000_by_1'

cobaya_yaml_file = '/mnt/ceph/users/bbolliet/cosmopower_training_data/ACTPol_lite_DR4_baseLCDM_taup_hip_bao_pks_ede-final_500000_by_1/ACTPol_lite_DR4_baseLCDM_taup_hip_bao_pks_ede-final.yaml'
with open(cobaya_yaml_file) as f:
    dict_from_yaml_file = yaml.load(f,Loader=yaml.FullLoader)
classy_precision = dict_from_yaml_file['theory']['classy']['extra_args']
lmax = dict_from_yaml_file['theory']['classy']['extra_args']['l_max_scalars']
nz = 5000 # number of z-points in redshift data [21oct22]
zmax = 20. # max redshift of redshift data [was 5 in 21oct22# now 20
z_arr = np.linspace(0.,zmax,nz) # z-array of redshift data [21oct22] oct 26 22: nz = 1000
k_arr = np.geomspace(5e-4,10.,1000)  #may 31 24 : (1e-4,50.,1000)
l_arr = np.arange(2,lmax+1)

# Create the dictionary
func_dict = {
    'l_func': l_arr,
    'k_func': k_arr,
    'z_func': z_arr,
    'discrete_func':  np.arange(nder)
}


print(f'processing data from {path_to_data_file}')

obstypes = ['TT', 'TE', 'EE', 'BB', 'PP','uTT', 'uTE', 'uEE', 'uBB', 'S8Z', 'HZ', 'DAZ', 'PKL', 'PKLCB', 'PKNL', 'PKNLCB']
nobs = len(obstypes)
npostpros = nobs + 1
# print(f'{len(obstypes)} obseravbles to emulate seperately + the set of comological parameters, i.e., {npostpros} cases:')
obstypes.append('DER')
# Initialize the dictionary with default values
obsdict = {obstype: {'obs_key_temp': None, 
                     'obs_params_exclude': [], 
                     'obs_familly': None, 
                     'apply_log1p': None,
                     'apply_log10': None, # apply log10 to absolute value, post rescaling (see scaling power)
                     'apply_abs' : None,
                     'use_pca' : None,
                     'scaling_power': 0 # scale by x^scaling_power where scaling power is k, l or z. 
                     
                    } for obstype in obstypes}

# Set specific values for 'obs_familly' based on the obstype
l_func_obstypes = ['TT', 'TE', 'EE', 'BB', 'uTT', 'uTE', 'uEE', 'uBB', 'PP']
k_func_obstypes = ['PKL', 'PKLCB', 'PKNL', 'PKNLCB']
z_func_obstypes = ['S8Z', 'HZ', 'DAZ']
discrete_func_obstypes = ['DER']

for obstype in l_func_obstypes:
    obsdict[obstype]['obs_familly'] = 'l_func'

for obstype in k_func_obstypes:
    obsdict[obstype]['obs_familly'] = 'k_func'

for obstype in z_func_obstypes:
    obsdict[obstype]['obs_familly'] = 'z_func'

for obstype in discrete_func_obstypes:
    obsdict[obstype]['obs_familly'] = 'discrete_func'
    
for obstype in obsdict:
    if obsdict[obstype]['obs_familly'] == 'z_func':
        obsdict[obstype]['obs_params_exclude'] = ['tau_reio','r']

    if obstype in ['TE','uTE']:
        obsdict[obstype]['use_pca'] = True
    else:
        obsdict[obstype]['use_pca'] = False

    if obstype in ['TE','uTE']:
        obsdict[obstype]['apply_log10'] = False
    else:
        obsdict[obstype]['apply_log10'] = True
        
    if obstype in ['DER']:
        obsdict[obstype]['apply_abs'] = True
    else:
        obsdict[obstype]['apply_abs'] = False
        
    if obstype in []:
        obsdict[obstype]['apply_log1p'] = True
    else: 
        obsdict[obstype]['apply_log1p'] = False
        
    if obstype == 'DER':
        obsdict[obstype]['obs_key_temp'] = 'TT'
    else:
        obsdict[obstype]['obs_key_temp'] = obstype
        
    if obstype in ['TT','uTT','TE','uTE','EE','uEE','BB','uBB']:
        obsdict[obstype]['scaling_power'] = 2
        
    if obstype in ['PP']:
        obsdict[obstype]['scaling_power'] = 3
    
    if obstype in ['PKL', 'PKLCB', 'PKNL', 'PKNLCB']:
        obsdict[obstype]['scaling_power'] = 3
        
    if obsdict[obstype]['obs_familly'] == 'l_func':
        obsdict[obstype]['obs_params_exclude'].append('z_pk_save_nonclass')

        
    if obsdict[obstype]['obs_familly'] == 'z_func':
        obsdict[obstype]['obs_params_exclude'].append('z_pk_save_nonclass')
        obsdict[obstype]['obs_params_exclude'].append('tau_reio')
    
    if obsdict[obstype]['obs_familly'] == 'k_func':
        obsdict[obstype]['obs_params_exclude'].append('tau_reio')
        
    if obstype in ['DAZ','HZ']:
        obsdict[obstype]['obs_params_exclude'].extend(['logA','n_s'])
        
    if obstype in ['DER']:
        obsdict[obstype]['obs_params_exclude'].extend(['z_pk_save_nonclass'])




do_plot = False # do you want to plot a few samples for testing?
if do_plot: 
    import matplotlib.pyplot as plt

aggregate = {}




def get_lhs_indices(group_id):
    # Construct the directory path based on the provided group_id
    directory_path = f"{path_to_data_file}/group_{group_id}"

    # Initialize an empty list to store the indices
    indices = []

    # Define the pattern for matching filenames
    pattern = re.compile(r"LHS_parameter_file_(\d+)\.npz")

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        match = pattern.match(filename)
        if match:
            # Extract the integer index from the filename
            index = int(match.group(1))
            indices.append(index)
    indices.sort()
    return indices

def run(args):
    idx_process = args.pa-1
    this_obs = obstypes[idx_process]
    key = this_obs
    
    
    
    aggregate[this_obs] = {'failed_samples':None,'data':None,'params':None}
    
    notfound_sample_innonempty_groups = []
    data_rows = []
    params_rows = []
    idx_sample_row = []
    
    # if this_obs != 'DAZ':
    #     continue
    this_obs_key = obsdict[this_obs]['obs_key_temp']
    print(f'dealing with case: {this_obs}, using data from key: {this_obs_key}')
    
    if do_plot:
        fig,ax = plt.subplots()
    params_keys = None
    for nonempty_group_id in nonempty_groups: 
        group_idx = nonempty_group_id
        print(f'doing group: {group_idx}')
        param_lh_indices = get_lhs_indices(nonempty_group_id)
    
        nsamps = 0
        for idx_sample in param_lh_indices:
            success = False
            
            # print(f'looking at samples from group {group_idx}')

            try: 
                this_sample_data = {
                    this_obs : {
                        'path_to_data_file': f"{path_to_data_file}/group_{group_idx}/{this_obs_key}/data_{idx_sample}.npy",
                        'y': {},
                        'x': {},
                        'input_params': {},
                        'derived_params': {},
                        'input_params_keys'  : {}
                    }
                }
                # print(f'loading data from:',path_to_data_file+f'/group_{group_idx}'+'/LHS_parameter_file_{}.npz'.format(idx_sample))
                this_sample_data[this_obs]['input_params_keys'] = np.load(path_to_data_file+f'/group_{group_idx}'+'/LHS_parameter_file_{}.npz'.format(idx_sample)).files
                # print('data loded:',this_sample_data[this_obs]['input_params_keys'])
                if not params_keys:
                    params_keys = this_sample_data[this_obs]['input_params_keys'] 

                # print('keys loaded')
                this_sample_data[this_obs]['y'] = np.load(this_sample_data[this_obs]['path_to_data_file'],allow_pickle=True).item()[0][nscalars+1:]
                this_sample_data[this_obs]['x'] = func_dict[obsdict[this_obs]['obs_familly']]
                this_sample_data[this_obs]['input_params'] =  np.load(this_sample_data[this_obs]['path_to_data_file'],allow_pickle=True).item()[0][:npars+1]
                this_sample_data[this_obs]['derived_params'] =  np.load(this_sample_data[this_obs]['path_to_data_file'],allow_pickle=True).item()[0][npars+1:nscalars+1]

                if this_obs == 'DER':
                    this_sample_data[this_obs]['y'] = this_sample_data[this_obs]['derived_params'].copy()

                this_sample_data[this_obs]['y'] *= this_sample_data[this_obs]['x']**obsdict[this_obs]['scaling_power']
                
                if obsdict[this_obs]['apply_log10']:
                    this_sample_data[this_obs]['y'] = np.log10(np.abs(this_sample_data[this_obs]['y']))
                if obsdict[this_obs]['apply_log1p']:
                    this_sample_data[this_obs]['y'] = np.log1p(this_sample_data[this_obs]['y'])
                
                success = True
                data_rows.append(this_sample_data[this_obs]['y'])
                params_rows.append(this_sample_data[this_obs]['input_params'])
                idx_sample_row.append(idx_sample)
                

            except: 
                print(f'---->data not found for sample {idx_sample} in group_{group_idx}')
                notfound_sample_innonempty_groups.append(idx_sample)

            if do_plot and success:

                ax.plot(this_sample_data[this_obs]['x'],this_sample_data[this_obs]['y'])
                ax.set_title(f"{this_obs} Data")
                
            if success:
                nsamps += 1
                if nsamps>nsamps_stop:
                    break
                
    if do_plot:        
        if this_obs != 'DER':
            ax.set_xscale('log')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
    # Convert the lists to pandas DataFrames and store in the aggregate dictionary
    aggregate[this_obs]['data'] = pd.DataFrame(data_rows)
    aggregate[this_obs]['params'] = pd.DataFrame(params_rows)
    aggregate[this_obs]['idx_samples'] = pd.DataFrame(idx_sample_row)
    aggregate[this_obs]['failed_samples'] = notfound_sample_innonempty_groups
    aggregate[this_obs]['params'].columns = params_keys
    if this_obs == 'DER':
        aggregate[this_obs]['data'].columns = derived_params_names
        
        
    # find the relevant features
    series_data = aggregate[key]['data'].iloc[0]
    # Create a boolean mask for non-nan, non-inf, and non-zero values
    mask = np.isfinite(series_data) & (series_data != 0)

    # Get the indices of the filtered data
    filtered_indices = series_data[mask].index
    aggregate[key]['filtered_indices'] = filtered_indices
    
    # build filtered aggregate
    # unfiltered_aggregate = copy.deepcopy(aggregate)
    
    ##### end aggregate build
    if 'KNL' in key: 
        key_PKL = key.replace('N', '')
        aggregate[key_PKL] = {'failed_samples':None,'data':None,'params':None}

        notfound_sample_innonempty_groups = []
        data_rows = []
        params_rows = []
        idx_sample_row = []

        # if key_PKL != 'DAZ':
        #     continue
        key_PKL_key = obsdict[key_PKL]['obs_key_temp']
        print(f'dealing with case: {key_PKL}, using data from key: {key_PKL_key}')

        if do_plot:
            fig,ax = plt.subplots()
        params_keys = None
        for nonempty_group_id in nonempty_groups: 
            group_idx = nonempty_group_id
            print(f'doing group: {group_idx}')
            param_lh_indices = get_lhs_indices(nonempty_group_id)

            nsamps = 0
            for idx_sample in param_lh_indices:
                success = False

                # print(f'looking at samples from group {group_idx}')

                try: 
                    this_sample_data = {
                        key_PKL : {
                            'path_to_data_file': f"{path_to_data_file}/group_{group_idx}/{key_PKL_key}/data_{idx_sample}.npy",
                            'y': {},
                            'x': {},
                            'input_params': {},
                            'derived_params': {},
                            'input_params_keys'  : {}
                        }
                    }
                    # print(f'loading data from:',path_to_data_file+f'/group_{group_idx}'+'/LHS_parameter_file_{}.npz'.format(idx_sample))
                    this_sample_data[key_PKL]['input_params_keys'] = np.load(path_to_data_file+f'/group_{group_idx}'+'/LHS_parameter_file_{}.npz'.format(idx_sample)).files
                    # print('data loded:',this_sample_data[key_PKL]['input_params_keys'])
                    if not params_keys:
                        params_keys = this_sample_data[key_PKL]['input_params_keys'] 

                    # print('keys loaded')
                    this_sample_data[key_PKL]['y'] = np.load(this_sample_data[key_PKL]['path_to_data_file'],allow_pickle=True).item()[0][nscalars+1:]
                    this_sample_data[key_PKL]['x'] = func_dict[obsdict[key_PKL]['obs_familly']]
                    this_sample_data[key_PKL]['input_params'] =  np.load(this_sample_data[key_PKL]['path_to_data_file'],allow_pickle=True).item()[0][:npars+1]
                    this_sample_data[key_PKL]['derived_params'] =  np.load(this_sample_data[key_PKL]['path_to_data_file'],allow_pickle=True).item()[0][npars+1:nscalars+1]
                    this_sample_data[key_PKL]['y'] *= this_sample_data[key_PKL]['x']**obsdict[key_PKL]['scaling_power']

                    if obsdict[key_PKL]['apply_log10']:
                        this_sample_data[key_PKL]['y'] = np.log10(np.abs(this_sample_data[key_PKL]['y']))
                    if obsdict[key_PKL]['apply_log1p']:
                        this_sample_data[key_PKL]['y'] = np.log1p(this_sample_data[key_PKL]['y'])

                    success = True
                    data_rows.append(this_sample_data[key_PKL]['y'])
                    params_rows.append(this_sample_data[key_PKL]['input_params'])
                    idx_sample_row.append(idx_sample)


                except: 
                    print(f'---->data not found for sample {idx_sample} in group_{group_idx}')
                    notfound_sample_innonempty_groups.append(idx_sample)

                if success:
                    nsamps += 1
                    if nsamps>nsamps_stop:
                        break

        # Convert the lists to pandas DataFrames and store in the aggregate dictionary
        aggregate[key_PKL]['data'] = pd.DataFrame(data_rows)
        aggregate[key_PKL]['params'] = pd.DataFrame(params_rows)
        aggregate[key_PKL]['idx_samples'] = pd.DataFrame(idx_sample_row)
        aggregate[key_PKL]['failed_samples'] = notfound_sample_innonempty_groups
        aggregate[key_PKL]['params'].columns = params_keys
        if key_PKL == 'DER':
            aggregate[key_PKL]['data'].columns = derived_params_names


        # find the relevant features
        series_data = aggregate[key_PKL]['data'].iloc[0]
        # Create a boolean mask for non-nan, non-inf, and non-zero values
        mask = np.isfinite(series_data) & (series_data != 0)

        # Get the indices of the filtered data
        filtered_indices = series_data[mask].index
        aggregate[key_PKL]['filtered_indices'] = filtered_indices

    # build filtered aggregate
    unfiltered_aggregate = copy.deepcopy(aggregate)
    
    
    
    if key != 'DER':
        aggregate[key]['data'] = aggregate[key]['data'].iloc[:,  aggregate[key]['filtered_indices']]
        
        
    # deal with Pk save non-class
    # Function to find rows with NaN or Inf values
    def find_nan_or_inf_rows(dataframe):
        mask = dataframe.applymap(lambda x: pd.isna(x) or np.isinf(x))
        indices_with_nan_or_inf = mask.any(axis=1).index[mask.any(axis=1)]
        return indices_with_nan_or_inf

    if 'KNL' in key: 
        
        key_PKL = key.replace('N', '')
        print(f'dealing with {key} will substitute {key_PKL}')
        nan_indices = find_nan_or_inf_rows(unfiltered_aggregate[key]['data'])
        substitute_samples_ids = unfiltered_aggregate[key]['params']['z_pk_save_nonclass'][nan_indices][unfiltered_aggregate[key]['params']['z_pk_save_nonclass']>5.].index
        aggregate[key]['data'] = unfiltered_aggregate[key]['data']
        aggregate[key]['data'].loc[substitute_samples_ids] = unfiltered_aggregate[key_PKL]['data'].loc[substitute_samples_ids] 
        
        idx_samps_to_remove = find_nan_or_inf_rows(aggregate[key]['data'])
        aggregate[key]['data'] = aggregate[key]['data'].drop(idx_samps_to_remove)
        aggregate[key]['params'] = aggregate[key]['params'].drop(idx_samps_to_remove)
        aggregate[key]['idx_samples'] = aggregate[key]['idx_samples'].drop(idx_samps_to_remove)
        # break
        
        
    #use only params of interest
    aggregate[key]['params'] = aggregate[key]['params'].drop(columns=[col for col in obsdict[key]['obs_params_exclude'] if col in aggregate[key]['params'].columns])
    params_indices = aggregate[key]['params'].index
    data_indices = aggregate[key]['data'].index

    if params_indices.equals(data_indices):
        print(f"The {key} DataFrames have the same indices.")
    else:
        print(f"------> CAREFUL: The {key} DataFrames do not have the same indices. This indicates problem in preprocessing above. I stop here.")
        print(f"params_indices: {params_indices}")
        print(f"data_indices: {data_indices}")
        return
    
    # Split the DataFrames into train and test sets
    params_train, params_test = train_test_split(aggregate[key]['params'], test_size=test_size, random_state=42)
    data_train, data_test = train_test_split(aggregate[key]['data'], test_size=test_size, random_state=42)

    # Creating the corresponding dictionaries
    aggregate[key]['train'] = {'params': params_train, 'data': data_train}
    aggregate[key]['test'] = {'params': params_test, 'data': data_test}


    ls = np.arange(1, aggregate[key]['train']['data'].shape[1]+1)
    data_train_array = aggregate[key]['train']['data'].to_numpy()
    data_test_array = aggregate[key]['test']['data'].to_numpy()
    params_train_array = aggregate[key]['train']['params'].to_numpy()
    params_test_array = aggregate[key]['test']['params'].to_numpy()

    processed_spectra_dict_train = {'modes': ls,
                              'features': data_train_array}
    processed_spectra_dict_test = {'modes': ls,
                             'features': data_test_array}

    params_name_list = aggregate[key]['test']['params'].columns.tolist()

    parameters_dict_train = {params_name_list[i]: params_train_array[:, i] for i in range(len(params_name_list))}
    parameters_dict_test = {params_name_list[i]: params_test_array[:, i] for i in range(len(params_name_list))}

    path_to_data_file

    train_params_path = f'{path_to_data_file}/{key}_training_params.npz'
    test_params_path = f'{path_to_data_file}/{key}_testing_params.npz'
    train_processed_spectra_path = f'{path_to_data_file}/{key}_training_processed_data.npz'
    test_processed_spectra_path = f'{path_to_data_file}/{key}_testing_processed_data.npz'


    # Save the data
    print('saving.....')
    np.savez(train_params_path, **parameters_dict_train)
    np.savez(test_params_path, **parameters_dict_test)
    np.savez(train_processed_spectra_path, **processed_spectra_dict_train)
    np.savez(test_processed_spectra_path, **processed_spectra_dict_test)
    print('success, bye!')

    
def main():
    parser=argparse.ArgumentParser(description="reprocess spectra")
    parser.add_argument("-pa",help="pa" ,dest="pa", type=int, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
	main()
