import os
import numpy as np
from nemesispy_scatter.common.info_svp import vp_data_dict
from nemesispy_scatter.common.read_nemesis import *
from nemesispy_scatter.common.parameterisations import *

def stack_and_pad(arrays):
    # Determine the maximum length
    max_length = max(len(arr) for arr in arrays)
    
    # Initialize a zero-padded array
    padded_array = np.zeros((len(arrays), max_length))
    
    # Fill the array
    for i, arr in enumerate(arrays):
        padded_array[i, :len(arr)] = arr
    
    return padded_array

def remove_file(input_string):
    segments = input_string.split('/')
    segments.pop()  # Remove the last segment
    return '/'.join(segments)+'/'
                      
def parse_apr_file(file_path, npro, vpdict):
    """
    This reads in a .apr file and returns the apriori state vector with error,
    a function to map state vectors to atmospheric profiles, and some useful flags.
    """
    
    
    xa = []
    xa_err = []
    gas_ids = []
    iso_ids = []
    params = []
    haze_wave_grids = []
    calc_wave_grids = []
    nreals = []
    vrefs = []
    vnorms = []
    cont_flags = []
    var_flags = []  # Initialize var_flags list
    size_flags = []
    n_real_flags = []
    folder = remove_file(file_path)
    
    with open(file_path, "r") as f:
        f.readline()  # Skip header
        nvar = int(f.readline().split('!')[0])  # Number of profiles
        var_start_index = 0
        while True:
            
            line = f.readline().split('!')[0].split()
            if not line:
                break

            if len(line) == 3:
                var_id, iso_id, param_id = map(int, line)
                gas_ids.append(var_id)
                iso_ids.append(iso_id)

                if param_id == 0:
                    # Processing Continuum
                    cont_filename = f.readline().replace('\n', '').strip()
                    with open(folder+cont_filename, "r") as cont_f:
                        _, clen = cont_f.readline().split()
                        cont_flags.append((len(xa),len(xa)+npro,0, float(clen), 0))
                        
                        for _ in range(npro):
                            _, xai, xa_erri = cont_f.readline().split()
                            xa.append(float(xai))
                            xa_err.append(float(xa_erri))
                    param = get_parameterisation(0, npro)
                    if iso_id == -1:
                        size_flags.append(var_id) # Could we move this outside param_id == 0?
                    if iso_id == -2:
                        n_real_flags.append(var_id)
                elif var_id == 444:
                    # Processing Haze
                    haze_filename = f.readline().replace('\n', '').strip()
                    haze_params, haze_param_errs, haze_waves, calc_waves, clen,\
                                        vref, nreal_ref, v_od_norm = parse_hazef_file(folder+haze_filename)
                    
                    cont_flags.append((len(xa)+len(haze_params)-len(haze_waves),\
                                       len(xa)+len(haze_params),1, clen, len(haze_wave_grids)))
                    haze_wave_grids.append(haze_waves)
                    calc_wave_grids.append(calc_waves)
                    vrefs.append(vref)
                    vnorms.append(v_od_norm)
                    nreals.append(nreal_ref)
                    
                    
                    for i in range(len(haze_params)):
                        xa.append(haze_params[i])
                        xa_err.append(haze_param_errs[i])
                    
                    param = get_parameterisation(0, len(haze_params))
                else:
                    # Processing Other Variables
                    param = get_parameterisation(param_id, npro)
                
                var_end_index = var_start_index + param[1]
                
                params.append((param,var_start_index,var_end_index))
                
                var_flags.append((var_start_index, var_end_index, len(var_flags)+1, var_id, iso_id, param_id))  
                var_start_index += param[1]

            else:
                xa.append(float(line[0]))
                try:
                    xa_err.append(float(line[1]))
                except:
                    xa_err.append(xa[-1]*1e-7)
                    
    get_profile = profile_factory(gas_ids, iso_ids, npro, params, haze_wave_grids, calc_wave_grids, \
                                  vrefs, nreals, vnorms, size_flags, n_real_flags, vpdict)
    return np.array(xa), np.array(xa_err), get_profile, cont_flags,\
           np.array(var_flags, dtype=int), haze_wave_grids, size_flags, n_real_flags

SENTINEL = np.array([999])
def add_segment(arr_list, new_segment):
    return arr_list + [new_segment, SENTINEL]        
        
def profile_factory(apr_gas_id, apr_iso_id, npro, param_list, haze_wave_grids, calc_wave_grids,
                    vrefs, nreals, vnorms, apr_size_flags, apr_n_real_flags,vpdict, iscat = 1):    # add iscat from .inp
    """
    Returns the function get_profile, which uses information from the .apr file
    to convert state vectors into atmospheric profiles.
    """
       
    
    apr_gas_id = np.array(apr_gas_id)
    apr_iso_id = np.array(apr_iso_id)

    num_gas = (apr_gas_id > 0).sum()
    num_aer = (apr_gas_id < 0).sum()
                              
    T_flag = 0 in apr_gas_id
    if T_flag:
        T_index = np.argmin(np.abs(apr_gas_id))
            
    def get_profile(H_model, P_model, T_model, VMR_model, A_model, PARA_model, gas_id, iso_id, planet_parameters, xn):
        nmodes = A_model.shape[1]
        imode = 0
        xn_index = 0
        if len(apr_size_flags) > 0:
            size_model = np.zeros_like(A_model)
        else:
            size_model = None
        
        if len(apr_n_real_flags) > 0:
            n_real_model = np.zeros_like(A_model)
        else:
            n_real_model = None
        
        
        if T_flag:
            T_param = param_list[T_index]
            T_model = T_param[0][0](npro, H_model, P_model, T_model,\
                                 VMR_model, A_model, PARA_model, gas_id, iso_id, planet_parameters,\
                                 T_model, xn[T_param[1]: T_param[2]])
        
        A_info = []
        renorms = []
        
        for i,param in enumerate(param_list):
            
            if apr_gas_id[i]>0 and apr_gas_id[i]!=444: # Gas
                
                gas_index = np.argmin(np.abs(gas_id - apr_gas_id[i]))
                if gas_id[gas_index] != apr_gas_id[i]:
                    print('GAS MISMATCH')
                
                VMR_model[:,gas_index] = param[0][0](npro, H_model, P_model, T_model,\
                                                  VMR_model, A_model, PARA_model, gas_id, iso_id, planet_parameters,\
                                            VMR_model[:,gas_index], xn[param[1]: param[2]])
                
                
            if apr_gas_id[i]<0 and apr_iso_id[i]==0: # Aerosol
                if imode == nmodes:
                    PARA_model = param[0][0](npro, H_model, P_model, T_model,\
                                          VMR_model, A_model, PARA_model, gas_id, iso_id, planet_parameters,\
                                           PARA_model, xn[param[1]: param[2]])
                else:
                    A_model[:,int(-apr_gas_id[i]-1)] = param[0][0](npro, H_model, P_model, T_model,\
                                                       VMR_model, A_model, PARA_model, gas_id, iso_id, planet_parameters,\
                                                       A_model[:,int(-apr_gas_id[i]-1)], xn[param[1]: param[2]])
                    if param[0][2] >= 0:
                        renorms.append(xn[param[1]+param[0][2]])
                    else:
                        renorms.append(0)
                imode += 1
            elif apr_gas_id[i]<0 and apr_iso_id[i]==-1: # Continuous size profile
                size_model[:,int(-apr_gas_id[i]-1)] = param[0][0](npro, H_model, P_model, T_model,\
                                                            VMR_model, A_model, PARA_model, gas_id, iso_id, planet_parameters,\
                                            A_model[:,int(-apr_gas_id[i]-1)], xn[param[1]: param[2]])
                
            elif apr_gas_id[i]<0 and apr_iso_id[i]==-2: # Continuous n_real profile
                n_real_model[:,int(-apr_gas_id[i]-1)] = param[0][0](npro, H_model, P_model, T_model,\
                                                            VMR_model, A_model, PARA_model, gas_id, iso_id, planet_parameters,\
                                            A_model[:,int(-apr_gas_id[i]-1)], xn[param[1]: param[2]])
        
            elif apr_gas_id[i]==444: # N_imag
                a = xn[param[1]] 
                b = xn[param[1]+1]
                
                if iscat == 1: # add others
                    alpha = (1-3*b)/b
                
                
                dsize = np.array([a, b, alpha])
                haze_wave_grid = haze_wave_grids[apr_iso_id[i]-1]
                calc_wave_grid = calc_wave_grids[apr_iso_id[i]-1]
                
                r_int = np.array([0.015*np.min(haze_wave_grid), 0.0, 0.015*np.min(haze_wave_grid)]) # improve this
                
                nwave = len(haze_wave_grid)
                # Initialize list for concatenation
                A_info_param_list = []
                # Add segments
                A_info_param_list = add_segment(A_info_param_list, np.array([iscat]))
                A_info_param_list = add_segment(A_info_param_list, dsize)
                A_info_param_list = add_segment(A_info_param_list, r_int)
                A_info_param_list = add_segment(A_info_param_list, haze_wave_grid)
                A_info_param_list = add_segment(A_info_param_list, calc_wave_grid)
                A_info_param_list = add_segment(A_info_param_list, np.array([nreals[apr_iso_id[i]-1]]))
                A_info_param_list = add_segment(A_info_param_list, xn[param[1]+2: param[2]])
                A_info_param_list = add_segment(A_info_param_list, np.array([vrefs[apr_iso_id[i]-1]]))
                A_info_param_list = add_segment(A_info_param_list, np.array([vnorms[apr_iso_id[i]-1]]))
                A_info_param_list = add_segment(A_info_param_list, np.array([renorms[apr_iso_id[i]-1]]))

                A_info_param = np.concatenate(A_info_param_list)
                
                
                
                A_info.append(A_info_param) # could be an issue here if isos are not in the right order?
        
        for i in range(len(gas_id)):
            try:
                vp,svpflag = vpdict[(gas_id[i],iso_id[i])] 
                a,b,c,d = vp_data_dict[gas_id[i]]
                svp = vp*np.exp(a + b/T_model + c*T_model + d*T_model**2)
                pp = VMR_model[:,i]*P_model/101325
                VMR_model[:,i] = np.where(pp > svp, svp/(P_model/101325), VMR_model[:,i])
                
            except:
                continue
        
        
        
        mask = np.zeros(len(gas_id), dtype = bool)
        for i in range(len(apr_gas_id)):
            if apr_gas_id[i] > 0 and apr_gas_id[i] != 444:
                gas_index = np.where((gas_id == apr_gas_id[i]) & (iso_id == apr_iso_id[i]))[0][0]
                mask[gas_index] = True
        sum0 = VMR_model.sum(axis=1)
        sum1 = VMR_model[:,mask].sum(axis=1)
        VMR_model[:,~mask] *= ((1-sum1)/(sum0-sum1))[:,None]
        PARA_model = np.clip(PARA_model,0,1)
        return T_model, VMR_model, A_model, A_info, PARA_model, size_model, n_real_model

    return get_profile