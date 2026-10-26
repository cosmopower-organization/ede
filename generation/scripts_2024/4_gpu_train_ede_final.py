import tensorflow as tf
import numpy as np
from cosmopower import cosmopower_PCA
from cosmopower import cosmopower_NN
from cosmopower import cosmopower_PCAplusNN
import argparse
import yaml
from pkg_resources import resource_filename

tf.config.list_physical_devices('GPU')
tf.test.is_gpu_available()
device = 'gpu:0' if tf.config.list_physical_devices('GPU') else 'cpu'
print('using', device, 'device \n')


dirstr = 'ACTPol_lite_DR4_baseLCDM_taup_hip_bao_pks_ede-final_500000_by_1'
path_to_training_data = '/mnt/ceph/users/bbolliet/cosmopower_training_data/'
hmdir = '/mnt/home/bbolliet/'





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

def run(args):
    idx_process = args.pa-1
    Nless = args.Nless
    VAR = obstypes[idx_process]


    print('loading data')
    load_training_features = np.load(path_to_training_data+dirstr+f'/{VAR}_training_processed_data.npz')
    training_features = load_training_features['features']
    modes = load_training_features['modes']

    training_parameters = np.load(path_to_training_data+dirstr+'/%s_training_params.npz'%(VAR))
    
    if Nless>0:
        # Create a dictionary to store the modified arrays
        modified_training_parameters = {}

        # Loop through each item in the .npz file
        for key in training_parameters.files:
            modified_training_parameters[key] = training_parameters[key][:Nless]

        # Save the modified arrays back to a new .npz file
        np.savez(path_to_training_data+dirstr+'/%s_lesss_training_params.npz'%(VAR), **modified_training_parameters)
        training_parameters = np.load(path_to_training_data+dirstr+'/%s_lesss_training_params.npz'%(VAR))
        
        training_features = training_features[:Nless]

    
    print('data loaded')
    use_PCA = obsdict[VAR]['use_pca']
    if use_PCA:
        print('starting training with pca')
        n_pcas = 64
        cp_pca = cosmopower_PCA(parameters=training_parameters.files,
                                modes=modes,
                                n_pcas=n_pcas,
                                parameters_filenames=[path_to_training_data+dirstr+'/%s_training_params'%(VAR)],
                                features_filenames=[path_to_training_data+dirstr+'/%s_training_processed_data'%(VAR)],
                                verbose=True, # useful to follow the various steps
                                )
        cp_pca.transform_and_stack_training_data()
        cp_pca_nn = cosmopower_PCAplusNN(cp_pca=cp_pca,
                                         n_hidden=[512,512,512,512], # 4 hidden layers, each with 512 nodes
                                         verbose=True, # useful to understand the different steps in initialisation and training
                                         )
        with tf.device('/device:GPU:0'): # ensures we are running on a GPU
            # train
            cp_pca_nn.train(filename_saved_model='%s_pca_ede-final'%VAR,
                            # cooling schedule
                            validation_split=0.1,
                            learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                            batch_sizes=[1024, 1024, 1024, 1024, 1024],
                            gradient_accumulation_steps = [1, 1, 1, 1, 1],
                            # early stopping set up
                            patience_values = [100,100,100,100,100],
                            max_epochs = [1000,1000,1000,1000,1000],
                            )
    else:
        print('starting training without pca')
        cp_nn = cosmopower_NN(parameters=training_parameters.files,
                              modes=modes,
                              n_hidden = [512, 512, 512, 512], # 4 hidden layers, each with 512 nodes
                              verbose=True, # useful to understand the different steps in initialisation and training
                              )
        with tf.device(device):
            # train
            cp_nn.train(training_parameters=training_parameters,
                        training_features=training_features,
                        filename_saved_model=hmdir+'%s_ede-final'%VAR,
                        # cooling schedule
                        validation_split=0.1,
                        learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                        batch_sizes=[1024, 1024, 1024, 1024, 1024],
                        gradient_accumulation_steps = [1, 1, 1, 1, 1],
                        # early stopping set up
                        patience_values = [100,100,100,100,100],
                        max_epochs = [1000,1000,1000,1000,1000],
                        )
    
    
    
    print('success, bye!')

def main():
    parser=argparse.ArgumentParser(description="reprocess spectra")
    parser.add_argument("-pa",help="pa" ,dest="pa", type=int, required=True)
    parser.add_argument("-Nless",help="Nless" ,dest="Nless", type=int, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main()