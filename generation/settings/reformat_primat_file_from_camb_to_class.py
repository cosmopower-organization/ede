import numpy as np

primat_file = 'PRIMAT_Yp_DH_ErrorMC_2021.dat'
output_file = 'PRIMAT_Yp_DH_ErrorMC_2021_CLASS.dat'

# extract the columns that CLASS uses: ombh2 DeltaNnu Y_P
bbn_data = np.loadtxt(primat_file, usecols = (0,2,3))
print(len(bbn_data))

# Important: the DeltaN	values here are	computed w.r.t.	3.044, *not* 3.046 as in previous CLASS	BBN files!
#
# In order to avoid having to change internal workings of CLASS, let's just re-express the DeltaN values
#   s.t. they are computed w.r.t. 3.046.
# i.e.,
#
# Delta_N_CAMB = N - 3.044   -->   N = Delta_N_CAMB + 3.044
# Delta_N_CLASS = N - 3.046   -->   Delta_N_CLASS = (Delta_N_CAMB + 3.044) - 3.046
# --> Delta_N_CLASS = Delta_N_CAMB - 0.002

bbn_data[:,1] -= 0.002

np.savetxt(output_file, bbn_data, fmt='%.6f')
