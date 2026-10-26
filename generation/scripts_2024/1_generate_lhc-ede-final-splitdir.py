# this should be run with:


import argparse
import numpy as np
import yaml
import os
# from classy import Class
from pkg_resources import resource_filename
import pyDOE as pyDOE

set_width_from_plc18 = True
test_lh_boundaries = False
with_classy_precision = False #relevant only when requesting test of boundaries

def run(args):
    print(">>>> generating lhc for EDE!")
    n_samples_per_process = args.n_samples_per_process #int(sys.argv[1])
    n_processes  = args.n_processes #int(sys.argv[2]) # should be the same as number of "-np" processes
    n_samples = n_samples_per_process*n_processes

    path_to_training_data_dir = args.path_to_training_data_dir
    test_lh_boundaries = args.test_lh_boundaries




    path_to_cosmopower_dir = '/mnt/home/bbolliet/cosmopower'


    cobaya_yaml_file = args.yaml_file
    # path_to_cosmopower_dir+'/cosmopower/training/spectra_generation_scripts/yaml_files/ACTPol_lite_DR4_baseLCDM_taup_hip.yaml'
    # data_dir_name = 'ACTPol_lite_DR4_baseLCDM_taup_hip'

    # print(cobaya_yaml_file.split('/'))
    sp_str = cobaya_yaml_file.split('/')

    data_dir_name = sp_str[len(sp_str)-1]
    yaml_file_name = data_dir_name
    data_dir_name = data_dir_name.replace('.yaml','')



    data_dir_name += '_'+str(n_processes)+'_by_'+str(n_samples_per_process)

    with open(cobaya_yaml_file) as f:
        dict_from_yaml_file = yaml.load(f,Loader=yaml.FullLoader)

    # get path to the folder of this script
    # folder_path = os.path.abspath(os.path.dirname(__file__))
    if (path_to_training_data_dir is None):
        folder_path = path_to_cosmopower_dir+'/cosmopower/training/training_data'
        try:
            os.mkdir(folder_path)
        except FileExistsError:
            print("File exist")
        folder_path = path_to_cosmopower_dir+'/cosmopower/training/training_data'
    else:
        folder_path = path_to_training_data_dir
    folder_path += '/'+data_dir_name 
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print(folder_path)
        print("File exist")
        exit(0)


    with open(folder_path+'/'+yaml_file_name, 'w') as file:
        yaml.dump(dict_from_yaml_file, file)







    # it doesnt matter if these params are in the yaml or not.
    # but order needs to be preserved
    # cosmo_params = ['logA','n_s','theta_s_1e2','omega_b','omega_cdm','tau_reio']
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
    # cosmo_params_names_in_class = ['ln10^{10}A_s','n_s','100*theta_s','omega_b','omega_cdm','tau_reio']
    cosmo_params_names_in_class = ['ln10^{10}A_s',
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
                                   ] # oct 26: change thetas to H0 -- dec 9: added m_ncdm

    # last column of Table 1 of https://arxiv.org/pdf/1807.06209.pdf
    # p18_sigmas = [0.014,0.0042,0.00031,0.00015,0.0012,0.0074]
    # p18_values = [3.043,0.9652,1.04089,0.02233,0.1198,0.0540]

    p18_sigmas = [0.014,0.0042,-1,0.00015,0.0012,0.0074,0.06,1.,1.,1.,1.,1.] # oct 26: change thetas to H0-- dec 9: added m_ncdm
    p18_values = [3.043,0.9652,-1,0.02233,0.1198,0.0540,0.06,1.,1.,1.,1.,1.] # oct 26: change thetas to H0-- dec 9: added m_ncdm



    param_list_all = []
    param_list_varied = {}
    param_list_varied_cosmo = {}
    param_list_other = []
    pdict = dict_from_yaml_file['params']
    for (k,v) in zip(pdict.keys(),pdict.values()):
        #print(k,v)
        #print(k)
        if 'prior' in v:
            #print('varied param')
    #         print(v['prior'])
            param_list_varied[k] = {}
            param_list_varied[k]['bounds'] ={}
            if ('min' and 'max') in v['prior']:
    #             print(v['prior']['min'])
                param_list_varied[k]['bounds']={'min':v['prior']['min'],'max':v['prior']['max']}
            elif 'dist' in v['prior']:
    #             print(k,v['prior'])
                param_list_varied[k]['bounds']={'min':v['prior']['loc']-5.*v['prior']['scale'],'max':v['prior']['loc']+5.*v['prior']['scale']}
                if k == 'tau_reio':
                    if param_list_varied[k]['bounds']['min']<0:
                        param_list_varied[k]['bounds'] = {'min':0.001,'max':v['prior']['loc']+5.*v['prior']['scale']}

        else:
            param_list_other.append(k)

    for (p,v) in zip(param_list_varied.keys(),param_list_varied.values()):
        if p in cosmo_params:
    #         param_list_varied_cosmo[p]['bounds']={}
    #         param_list_varied_cosmo[p]['bounds']=v
            param_list_varied_cosmo[p]=v
            param_list_varied_cosmo[p]['class_name'] = {}
            param_list_varied_cosmo[p]['class_name'] = cosmo_params_names_in_class[cosmo_params.index(p)]


    if set_width_from_plc18:
        for p in param_list_varied_cosmo.keys():
            param_list_varied_cosmo[p]['bounds']['min'] = p18_values[cosmo_params.index(p)]-20.*p18_sigmas[cosmo_params.index(p)]
            param_list_varied_cosmo[p]['bounds']['max'] = p18_values[cosmo_params.index(p)]+20.*p18_sigmas[cosmo_params.index(p)]
            if p == 'tau_reio':
                if param_list_varied_cosmo[p]['bounds']['min']<0:
                    param_list_varied_cosmo[p]['bounds']['min'] = 0.02
                # added 19th october 2022:
                param_list_varied_cosmo[p]['bounds']['min'] = 0.02 # bounds on tau do not need to be wider than that...
                param_list_varied_cosmo[p]['bounds']['max'] = 0.12 # bounds on tau do not need to be wider than that...
            # added 19th october 2022
            # wider bounds on density
            if p == 'omega_cdm':
                param_list_varied_cosmo[p]['bounds']['min'] = 0.08
                param_list_varied_cosmo[p]['bounds']['max'] = 0.20
            # added 19th october 2022
            # wider bounds on scalar amplitude
            if p == 'logA':
                param_list_varied_cosmo[p]['bounds']['min'] = 2.5
                param_list_varied_cosmo[p]['bounds']['max'] = 3.5
            if p == 'H0': # oct 26 22
                param_list_varied_cosmo[p]['bounds']['min'] = 39.99
                param_list_varied_cosmo[p]['bounds']['max'] = 100.01
            if p == 'n_s': # dec 9 22
                param_list_varied_cosmo[p]['bounds']['min'] = 0.8 
                param_list_varied_cosmo[p]['bounds']['max'] = 1.2
            if p == 'm_ncdm': # dec 9 22
                param_list_varied_cosmo[p]['bounds']['min'] = 0.
                param_list_varied_cosmo[p]['bounds']['max'] = 0.33333 ## avoid errors
            
            if p == 'N_ur': #nov 24 23
                param_list_varied_cosmo[p]['bounds']['min'] = -2. # broad enough... relation between nur to neff=3.044 but holds for all neff: print(nur = 3.044-i*(0.71611/(4./11.)**(1./3.))**4) where i=N_ncdm
                param_list_varied_cosmo[p]['bounds']['max'] = 2.5  # broad enough...
                
            ### notes 10/06/2024. Idea, vary Neff set N_ur, T_ncdm accordingly as a function Neff (with N_ncdm =3)
            if p == 'fEDE': #nov 24 23
                param_list_varied_cosmo[p]['bounds']['min'] = 1e-3
                param_list_varied_cosmo[p]['bounds']['max'] = 0.5
            if p == 'log10z_c': #nov 24 23
                param_list_varied_cosmo[p]['bounds']['min'] = 3.
                param_list_varied_cosmo[p]['bounds']['max'] = 4.3
            if p == 'thetai_scf': #nov 24 23
                param_list_varied_cosmo[p]['bounds']['min'] = 0.1
                param_list_varied_cosmo[p]['bounds']['max'] = 3.1                
            if p == 'r': #nov 24 23
                param_list_varied_cosmo[p]['bounds']['min'] = 0.
                param_list_varied_cosmo[p]['bounds']['max'] = 0.3   
                
    n_params = len(param_list_varied_cosmo)
    
    # oct 26 22: add other parameters here
    # add z_pk_save
    k = 'z_pk_save'
    param_list_varied_cosmo[k] = {}
    param_list_varied_cosmo[k]['bounds'] = {}
    param_list_varied_cosmo[k]['bounds']={'min':-1,'max':-1}
    param_list_varied_cosmo[k]['bounds']['min'] = 0.
    param_list_varied_cosmo[k]['bounds']['max'] = 20.
    param_list_varied_cosmo[k]['class_name'] = 'z_pk_save_nonclass'
    n_params += 1 
    


    print('doing %d times %d calculations'%(n_processes,n_samples_per_process))

    for (k,v) in zip(param_list_varied_cosmo.keys(),param_list_varied_cosmo.values()):
        param_list_varied_cosmo[k]['array'] = {}
        param_list_varied_cosmo[k]['array'] = np.linspace(param_list_varied_cosmo[k]['bounds']['min'],param_list_varied_cosmo[k]['bounds']['max'],n_samples)

    all_params_list = []
    for (k,v) in zip(param_list_varied_cosmo.keys(),param_list_varied_cosmo.values()):
        all_params_list.append(v['array'])

    AllParams = np.vstack(all_params_list)
    lhd = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
    idx = (lhd * n_samples).astype(int)

    AllCombinations = np.zeros((n_samples, n_params))
    for i in range(n_params):
        AllCombinations[:, i] = AllParams[i][idx[:, i]]
    # AllCombinations

    # saving
    lh_param_dict = {}
    ip = 0
    for (k,v) in zip(param_list_varied_cosmo.keys(),param_list_varied_cosmo.values()):
        lh_param_dict[v['class_name']] = AllCombinations[:,ip]
        ip+=1


    # np.savez(path_to_cosmopower_dir+'/cosmopower/training/training_data/'+data_dir_name+'/LHS_parameter_file.npz', **lh_param_dict)
    np.savez(folder_path+'/LHS_parameter_file.npz', **lh_param_dict)


    # saving
    for idx_process in range(n_processes):
        group_idx = idx_process // 500
        group_folder = os.path.join(folder_path, f'group_{group_idx + 1}')

        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        params = dict(zip(lh_param_dict.keys(), AllCombinations[idx_process * n_samples_per_process:(idx_process + 1) * n_samples_per_process, :].T))
        np.savez(os.path.join(group_folder, f'LHS_parameter_file_{idx_process + 1}.npz'), **params)
    print('all files saved')

def main():
    parser=argparse.ArgumentParser(description="generate spectra")
    parser.add_argument("-n_samples_per_process",help="n_samples_per_process" ,dest="n_samples_per_process", type=int, required=True)
    parser.add_argument("-n_processes",help="n_processes" ,dest="n_processes", type=int, required=True)
    parser.add_argument("-test_lh_boundaries",help="test_lh_boundaries" ,dest="test_lh_boundaries", type=int, required=False)
    parser.add_argument("-yaml_file",help="yaml_file" ,dest="yaml_file", type=str, required=True)
    parser.add_argument("-path_to_training_data_dir",help="path_to_training_data_dir" ,dest="path_to_training_data_dir", type=str, required=False)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
	main()
