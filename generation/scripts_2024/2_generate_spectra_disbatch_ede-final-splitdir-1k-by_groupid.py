import argparse
import numpy as np
import re
import yaml
import os
import classy
from classy import Class
from pkg_resources import resource_filename
import time
from time import gmtime
from time import strftime

from itertools import groupby


cobaya_yaml_file = '/mnt/ceph/users/bbolliet/cosmopower_training_data/ACTPol_lite_DR4_baseLCDM_taup_hip_bao_pks_ede-final_500000_by_1/ACTPol_lite_DR4_baseLCDM_taup_hip_bao_pks_ede-final.yaml'
with open(cobaya_yaml_file) as f:
    dict_from_yaml_file = yaml.load(f,Loader=yaml.FullLoader)
classy_precision = dict_from_yaml_file['theory']['classy']['extra_args']
lmax = dict_from_yaml_file['theory']['classy']['extra_args']['l_max_scalars']
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


nz = 5000 # number of z-points in redshift data [21oct22]
zmax = 20. # max redshift of redshift data [was 5 in 21oct22# now 20
z_arr = np.linspace(0.,zmax,nz) # z-array of redshift data [21oct22] oct 26 22: nz = 1000
k_arr = np.geomspace(5e-4,10.,1000)  #may 31 24 : (1e-4,50.,1000)


obstypes = ['TT', 'TE', 'EE', 'BB', 'PP','uTT', 'uTE', 'uEE', 'uBB', 'S8Z', 'HZ', 'DAZ', 'PKL', 'PKLCB', 'PKNL', 'PKNLCB']
params = {
    'output': 'tCl pCl lCl mPk',
    'z_max_pk': zmax,
    'modes': 's,t',
    'non_linear': 'hmcode',
    'tensor method': 'exact',
}

n_jobs = 500 # number of jobs
n_samples_per_job = 1000 # has to be 1000 so that 100 x 500 jobs = 500k samples 

def run(args):

    def process_parameters(params_lhs, idx_sample):
        class_params_dict = {}
        non_class_params_dict = {}
        for k in params_lhs:
            if 'nonclass' not in k:
                class_params_dict[k] = params_lhs[k][idx_sample]
            else:
                non_class_params_dict[k] = params_lhs[k][idx_sample]
            if '3state' in cobaya_yaml_file or 'ede' in cobaya_yaml_file:
                if k == 'm_ncdm':
                    class_params_dict[k] = f"{params_lhs[k][idx_sample]},{params_lhs[k][idx_sample]},{params_lhs[k][idx_sample]}"
        return class_params_dict, non_class_params_dict

    def merge_dictionaries(params, class_params_dict, classy_precision):
        params_keys = set(params.keys())
        class_params_dict_keys = set(class_params_dict.keys())
        classy_precision_keys = set(classy_precision.keys())
        common_keys = (params_keys & class_params_dict_keys) | (params_keys & classy_precision_keys) | (class_params_dict_keys & classy_precision_keys)
        if not common_keys:
            merged_params_dict = {**params, **class_params_dict, **classy_precision}
        return merged_params_dict

    def cosmology_calculations(merged_params_dict):
        cosmo = Class()
        cosmo.set(merged_params_dict)
        try:
            cosmo.compute()
            return cosmo
        except:
            return None

    def extract_and_stack_params(cosmo, params_lhs, idx_sample, derived_params_names, z_arr, k_arr, z_pk_save):
        cls = cosmo.lensed_cl(lmax=lmax)
        unlensed_cls = cosmo.raw_cl(lmax=lmax)
        sigma8_of_z = np.vectorize(cosmo.sigma)
        Hubble_of_z = np.vectorize(cosmo.Hubble)
        da_of_z = np.vectorize(cosmo.angular_distance)
        pkl_at_z = np.vectorize(cosmo.pk_lin)
        pknl_at_z = np.vectorize(cosmo.pk)
        pkl_cb_at_z = np.vectorize(cosmo.pk_cb_lin)
        pknl_cb_at_z = np.vectorize(cosmo.pk_cb)

        powers = cls
        derived_params = {}
        for p in derived_params_names:
            if p == 'rs_drag':
                derived_params[p] = cosmo.rs_drag()
            else:
                derived_params.update(cosmo.get_current_derived_parameters([p]))

        try:
            pknlarr = pknl_at_z(k_arr,z_pk_save)
        except:
            pknlarr = 0.*k_arr
        try:
            pknlarr_cb = pknl_cb_at_z(k_arr,z_pk_save)
        except:
            pknlarr_cb = 0.*k_arr

        arrays = {
            'TT': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], powers['tt'][2:])),
            'TE': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], powers['te'][2:])),
            'EE': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], powers['ee'][2:])),
            'BB': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], powers['bb'][2:])),
            'PP': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], powers['pp'][2:])),
            'uTT': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], unlensed_cls['tt'][2:])),
            'uTE': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], unlensed_cls['te'][2:])),
            'uEE': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], unlensed_cls['ee'][2:])),
            'uBB': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], unlensed_cls['bb'][2:])),
            'S8Z': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], sigma8_of_z(8./cosmo.h(), z_arr))),
            'HZ': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], Hubble_of_z(z_arr))),
            'DAZ': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], da_of_z(z_arr))),
            'PKL': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], pkl_at_z(k_arr, z_pk_save))),
            'PKLCB': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], pkl_cb_at_z(k_arr, z_pk_save))),
            'PKNL': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], pknlarr)),
            'PKNLCB': np.hstack(([params_lhs[k][idx_sample] for k in params_lhs], [derived_params[k] for k in derived_params.keys()], pknlarr_cb)),
        }
        return arrays

    def populate_output_data(all_obsdata, arrays, idx_sample):
        for key in arrays:
            all_obsdata[key]['data'][idx_sample] = arrays[key].astype(np.float32)

    def save_output_data(all_obsdata):
        for key in all_obsdata:
            np.save(all_obsdata[key]['path_to_data_file'], all_obsdata[key]['data'])

    # Main processing loop
    # Initialization and Loop Setup
    home_dir = '/mnt/home/bbolliet/cosmopower/cosmopower/training/spectra_generation_scripts/'
    empty_groups_id = np.loadtxt(home_dir+'empty_groups_ede-final_500000_by_1.txt', dtype=int) ## initially, this should be all groups from 1 to 10000
    job_array_id = args.pa

    def get_lhs_indices(group_id):
        # Construct the directory path based on the provided group_id
        directory_path = f"{args.dir}/group_{group_id}"

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

    group_idx = empty_groups_id[job_array_id-1]
    param_lh_indices = get_lhs_indices(group_idx)

    for idx_sample in param_lh_indices:


        all_obsdata = {
            key: {
                'path_to_data_file': f"{args.dir}/group_{group_idx}/{key}/data_{idx_sample}.npy",
                'data': {}
            }
            for key in obstypes
        }

        params_lhs  = np.load(args.dir+f'/group_{group_idx}'+'/LHS_parameter_file_{}.npz'.format(idx_sample))


        class_params_dict, non_class_params_dict = process_parameters(params_lhs, 0)
        z_pk_save = non_class_params_dict['z_pk_save_nonclass']

        merged_params_dict = merge_dictionaries(params, class_params_dict, classy_precision)

        cosmo = cosmology_calculations(merged_params_dict)
        if cosmo is None:
            continue

        arrays = extract_and_stack_params(cosmo, params_lhs, 0, derived_params_names, z_arr, k_arr, z_pk_save)
        populate_output_data(all_obsdata, arrays, 0)
        save_output_data(all_obsdata)


    
    
def main():
    parser=argparse.ArgumentParser(description="generate spectra")
    parser.add_argument("-dir",help="dir" ,dest="dir", type=str, required=True)
    parser.add_argument("-pa",help="pa" ,dest="pa", type=int, required=False)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
	main()
