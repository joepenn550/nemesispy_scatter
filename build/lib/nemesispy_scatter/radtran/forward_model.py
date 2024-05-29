#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Interface class for running forward models.
"""
import numpy as np
from numba import njit
from nemesispy_scatter.radtran.calc_mmw import calc_mmw
from nemesispy_scatter.radtran.read import read_tables
from nemesispy_scatter.radtran.calc_radiance_scatter import calc_radiance
from nemesispy_scatter.radtran.read import read_cia
from nemesispy_scatter.radtran.calc_layer import calc_layer
from nemesispy_scatter.common.calc_hydrostat import calc_hydrostat
from nemesispy_scatter.common.get_gas_info import get_gas_id, get_gas_name
from nemesispy_scatter.radtran.scatter import makephase
from nemesispy_scatter.common.constants import N_A, AMU
from scipy.interpolate import CubicSpline

# def indices_to_sort(first, second):
#     order = {val: idx for idx, val in enumerate(first)}
#     default_index = len(first)
#     sorting_indices = sorted(range(len(second)), key=lambda i: order.get(second[i], default_index))
#     return sorting_indices
def indices_to_sort(gas_ref, iso_ref, gas, iso):
    sorted_indices = []
    used_indices = set()

    # First pass: Match and collect indices
    for index_ref in range(len(gas_ref)):
        for index in range(len(gas)):
            if gas[index] == gas_ref[index_ref] and iso[index] == iso_ref[index_ref]:
                if index not in used_indices:
                    sorted_indices.append(index)
                    used_indices.add(index)
                    break

    # Second pass: Append remaining indices
    sorted_indices.extend([i for i in range(len(gas)) if i not in used_indices])

    return sorted_indices

def find_info_segments(arr, sentinel):
    """
    Finds and returns the segments in arr separated by the sentinel value.
    """
    # Initialize variables
    start_index = 0
    segments = []
    sentinel_indices = np.where(arr == sentinel)[0]  # Find indices of all sentinels
    
    # Iterate through each sentinel index to slice segments
    for index in sentinel_indices:
        # Extract segment from start_index to current sentinel index
        segment = arr[start_index:index]
        segments.append(segment)
        start_index = index + 1  # Update start_index for next segment
    # Handle last segment if there's no sentinel at the end
    if start_index < arr.size:
        segments.append(arr[start_index:])
    
    return segments

def get_planet_parameters(planet_id, radrepo):
    with open(radrepo+'gravity.dat', 'r') as file: 
        for line in file:
            # Skip comments and headers
            if line.startswith('#') or line.strip() == '' or len(line)<5:
                continue
            # Split line into components
            
            parts = line.split()
            pid = parts[0] 
            if int(pid) == int(planet_id):
                # Extract parameters
                parameters = np.array([
                    float(parts[2])*1e24,
                    float(parts[3])/1e3,
                    float(parts[4])/1e6,
                    float(parts[5])/1e8,
                    float(parts[6])*1e3,
                    float(parts[7]),
                    float(parts[8])*24*60*60,
                    float(parts[9])
                ])
                return parameters
    return None

@njit
def lblconv(fwhm,ishape,nwave,vwave,y,nconv,vconv):

    yout = np.zeros(nconv)
    ynor = np.zeros(nconv)

    if fwhm>0.0:
        #Set total width of Hamming/Hanning function window in terms of
        #numbers of FWHMs for ISHAPE=3 and ISHAPE=4
        nfw = 3.

        for j in range(nconv):
            yfwhm = fwhm
            vcen = vconv[j]
            if ishape==0:
                v1 = vcen-0.5*fwhm
                v2 = v1 + yfwhm
            elif ishape==1:
                v1 = vcen-fwhm
                v2 = vcen+fwhm
            elif ishape==2:
                sig = 0.5*yfwhm/np.sqrt( np.log(2.0)  )
                v1 = vcen - 3.*sig
                v2 = vcen + 3.*sig
            else:
                v1 = vcen - nfw*yfwhm
                v2 = vcen + nfw*yfwhm


            #Find relevant points in tabulated files
            inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
            inwave = inwave1[0]

            np1 = len(inwave)
            for i in range(np1):
                f1=0.0
                if ishape==0:
                    #Square instrument lineshape
                    f1=1.0
                elif ishape==1:
                    #Triangular instrument shape
                    f1=1.0 - abs(vwave[inwave[i]] - vcen)/yfwhm
                elif ishape==2:
                    #Gaussian instrument shape
                    f1 = np.exp(-((vwave[inwave[i]]-vcen)/sig)**2.0)

                if f1>0.0:
                    yout[j] = yout[j] + f1*y[inwave[i]]
                    ynor[j] = ynor[j] + f1

            yout[j] = yout[j]/ynor[j]

    return yout


class ForwardModel():

    def __init__(self):
        """
        Attributes to store planet and opacity data.
        These attributes shouldn't change during a retrieval.
        """
        # planet and planetary system data
        self.M_plt = None
        self.R_plt = None
        self.M_star = None # currently not used
        self.R_star = None # currently not used
        self.T_star = None # currently not used
        self.semi_major_axis = None # currently not used
        self.NLAYER = None
        self.is_planet_model_set = False

        # opacity data
        self.gas_id_list = None
        self.gas_name_list = None
        self.iso_id_list = None
        self.wave_grid = None
        self.g_ord = None
        self.del_g = None
        self.k_table_P_grid = None
        self.k_table_T_grid = None
        self.k_gas_w_g_p_t = None
        self.k_wave_grid = None
        self.cia_nu_grid = None
        self.cia_frac_grid = None
        self.cia_T_grid = None
        self.k_cia_pair_t_w = None
        self.is_opacity_data_set = False

        self.fours = {}
        
        self.mu = None
        self.wtmu = None
        
        self.phase_array = None
        self.xextnorms = None
        self.refwaves = None
        self.remake_phase = False
        self.makephase_downscaling = 1
        
    def sanity_check(self):
        assert self.M_plt > 0
        assert self.R_plt > 0
        assert self.NLAYER > 0
        assert self.is_planet_model_set
        assert self.is_opacity_data_set
        
        
  
    def set_planet_model(self, planet, gas_id_list, iso_id_list, NLAYER, radrepo,latitude,
        semi_major_axis=None):
        """
        Store the planetary system parameters
        """
        
        self.plt_params = get_planet_parameters(planet,radrepo)
        self.plt_params = np.concatenate([self.plt_params,[latitude]])
        self.M_plt = self.plt_params[0]
        self.R_plt = self.plt_params[4]

        gas_name_list = []
        for index, id in enumerate(gas_id_list):
            gas_name_list.append(get_gas_name(id))
        self.gas_name_list = np.array(gas_name_list)
        self.gas_id_list = gas_id_list
        self.iso_id_list = iso_id_list
        self.NLAYER = NLAYER

        self.is_planet_model_set = True
    
    def set_angles(self,mu,wtmu):
        self.mu = mu
        self.wtmu = wtmu
    
    
    def set_opacity_data(self, kta_file_paths, cia_file_path,wave_grid,dnu,npara):
        """
        Read gas ktables and cia opacity files and store as class attributes.
        """
        k_gas_id_list, k_iso_id_list, k_wave_grid, g_ord, del_g, k_table_P_grid,\
            k_table_T_grid, k_gas_w_g_p_t, lta_flags = read_tables(kta_file_paths)
        """
        Some gases (e.g. H2 and He) have no k table data so gas id lists need
        to be passed somewhere else.
        """
        
        self.k_gas_id_list = k_gas_id_list
        self.k_iso_id_list = k_iso_id_list
        self.wave_grid = wave_grid
        self.k_wave_grid = k_wave_grid

        self.g_ord = g_ord
        self.del_g = del_g
        self.k_table_P_grid = k_table_P_grid
        self.k_table_T_grid = k_table_T_grid
        self.k_gas_w_g_p_t = k_gas_w_g_p_t
#         self.k_gas_w_g_p_t = linear_interpolate_along_axis(self.k_gas_w_g_p_t,k_wave_grid,wave_grid,1)
        self.lta_flags = lta_flags
        
        cia_nu_grid, cia_frac_grid, cia_T_grid, k_cia_pair_t_w = read_cia(cia_file_path,dnu,npara)
        self.cia_frac_grid = cia_frac_grid
        self.cia_nu_grid = cia_nu_grid
        self.cia_T_grid = cia_T_grid
        self.k_cia_pair_t_w = k_cia_pair_t_w

        self.is_opacity_data_set = True

    def read_input_dict(self,input_dict):
        NLAYER = input_dict['NLAYER']
        M_plt = input_dict['M_plt']
        R_plt = input_dict['R_plt']
        gas_id_list = input_dict['gas_id']
        iso_id_list = input_dict['iso_id']
        kta_file_paths = input_dict['kta_file_paths']
        cia_file_path = input_dict['cia_file_path']
        self.set_planet_model(
            M_plt, R_plt, gas_id_list, iso_id_list, NLAYER
        )
        self.set_opacity_data(
            kta_file_paths, cia_file_path
        )

    def calc_point_spectrum(self, H_model, P_model, T_model, VMR_model,A_model,A_info,PARA_model,\
                            size_model,size_flags,n_real_model,n_real_flags,H_0,
                            path_angle,sol_ang,aphi, solspec=[], remake_phase=True, f_flag = False):
        """
        Calculate average layer properties from model inputs,
        then compute the spectrum of a plane parallel atmosphere.
        """
        mmw = []
        for i in range(len(P_model)):
            mmw.append(calc_mmw(self.gas_id_list,VMR_model[i],self.iso_id_list))
        mmw = np.array(mmw)
        
        AVOGAD=6.022045E23
        
        A_model = A_model[:,:len(A_info)] # shouldn't need to do this - change apr parsing
        # New aerosol logic - clean this up
        info_stats = [[] for i in range(10)]
        npro = len(H_model)
        nmodes = A_model.shape[1] + (npro - 1)*max(len(size_flags),len(n_real_flags))
        split_A_model = np.zeros((npro,nmodes))
        split_counter = 0
        for i in range(len(A_info)):
            if -(i+1) in size_flags or -(i+1) in n_real_flags:
                for ipro in range(npro):
                    segments = find_info_segments(A_info[i], sentinel=999)
                    for j,segment in enumerate(segments):
                        info_stats[j].append(segment)
                    if -(i+1) in size_flags:
                        info_stats[1][-1][0] = size_model[ipro,i] # setting dsize[0]
                    if -(i+1) in n_real_flags:
                        info_stats[5][-1][0] = n_real_model[ipro,i] # setting n_real
                        
                    split_A_model[ipro,split_counter+ipro] = A_model[ipro,i]
                split_counter += npro
            else:
                segments = find_info_segments(A_info[i], sentinel=999)
                for j,segment in enumerate(segments):
                    info_stats[j].append(segment)
                    
                split_A_model[:,split_counter] = A_model[:,i]
                split_counter += 1
                
        A_model = split_A_model 

        H_model = calc_hydrostat(P_model, T_model, mmw, self.M_plt, self.R_plt, 
                                 self.plt_params, H=H_model)
        
        H_layer,P_layer,T_layer,VMR_layer,U_layer,A_layer,PARA_layer,dH,scale \
            = calc_layer(
            self.R_plt, H_model, P_model, T_model, VMR_model,A_model,PARA_model,
            self.gas_id_list, self.NLAYER, path_angle=0.0, layer_type=self.layer_type,
            H_0=H_0, custom_P_base = self.custom_p_base
            )
        
        for i in range(nmodes):
            for j in range(A_layer.shape[0]):
                A_layer[j,i] = A_layer[j,i] * mmw[j] / AVOGAD / AMU * 1e-4 
            
        if self.FWHM > 0:
            wave_grid = self.k_wave_grid[(self.k_wave_grid<self.wave_grid.max()+1.5*self.FWHM) \
                                           * (self.k_wave_grid>self.wave_grid.min()-1.5*self.FWHM)]
            solspec = np.interp(wave_grid,self.wave_grid,solspec)
        else:
            wave_grid = self.wave_grid
            
            
        if len(solspec)==0:
            solspec = np.ones(len(self.wave_grid))
        
        if len(A_info) > 0:
            if self.phase_array is None or remake_phase:
                self.phase_array = np.zeros((nmodes,len(wave_grid),6))
                self.xextnorms = np.zeros(nmodes)
                
            for imode in range(nmodes):
                if imode >= len(info_stats[0]): # unused aerosol mode
                    break

                iscat, dsize, rs, nimag_wave_grid, calc_wave_grid, nreal_ref,\
                                  nimag, v_ref, v_norm,renorm = [info_stats[i][imode] for i in range(10)]

                if renorm > 0:
                    A_layer[:,imode] = A_layer[:,imode] * renorm/(A_layer[:,imode].sum())
                    
                if self.phase_array is None or remake_phase:
                    self.phase_array[imode] = makephase(wave_grid,
                         iscat = iscat,
                         dsize = dsize,
                         rs = rs,
                         nimag_wave_grid = nimag_wave_grid,
                         calc_wave_grid = calc_wave_grid,
                         nreal_ref = nreal_ref,
                         nimag = nimag,
                         refwave = v_ref,
                         normwave = v_norm,
                         downscaling = self.makephase_downscaling,
                         iwave = self.ispace)

                    
                    if self.makephase_downscaling > 1:

                        downsampled_wave_grid = wave_grid[::self.makephase_downscaling]
                        
                        spline = CubicSpline(downsampled_wave_grid, 
                              self.phase_array[imode,::self.makephase_downscaling, :2], 
                              axis=0)

                        self.phase_array[imode,:, :2] = spline(wave_grid)
                        self.xextnorms[imode] = spline(v_norm)[0][0]
                        
                        for iphas in range(2,5):
                            self.phase_array[imode,:, iphas] = np.interp(wave_grid,downsampled_wave_grid,\
                                                               self.phase_array[imode,::self.makephase_downscaling, iphas])
                            
                            
                    elif self.makephase_downscaling == 0:
                        
                        spline = CubicSpline(calc_wave_grid, 
                              self.phase_array[imode,:len(calc_wave_grid), :2], 
                              axis=0)

                        self.phase_array[imode,:, :2] = spline(wave_grid)
                        self.xextnorms[imode] = spline(v_norm)[0][0]
                        
                        for iphas in range(2,5):
                            self.phase_array[imode,:, iphas] = np.interp(wave_grid,calc_wave_grid,\
                                                               self.phase_array[imode,:len(calc_wave_grid), iphas])
                    
        else:
            self.phase_array = np.zeros((nmodes,len(wave_grid),6))
            self.xextnorms = np.ones(nmodes)
#         try:
#             indices = [np.concatenate([[np.argmin(np.abs(wave_grid - nimag_wave_grid[i]))] for i in range(len(nimag_wave_grid))])]
#             print(wave_grid[indices],repr(self.phase_array[0][indices]),flush=True)
#             print(self.xextnorms,flush=True)
#         except:
#             pass
        
        del_gases = []
        for igas, (gas, iso) in enumerate(zip(self.k_gas_id_list,self.k_iso_id_list)):
            if gas not in self.gas_id_list:
                print(f'Missing {gas,iso}', flush=True)
                del_gases.append(igas)
            elif iso not in self.iso_id_list[np.where(self.gas_id_list == gas)]:
                print(f'Missing {gas,iso}', flush=True)
                del_gases.append(igas)
                
        self.k_gas_w_g_p_t = np.delete(self.k_gas_w_g_p_t,del_gases,axis=0)
        self.k_gas_id_list = np.delete(self.k_gas_id_list,del_gases)
        self.k_iso_id_list = np.delete(self.k_iso_id_list,del_gases)
              
        gas_sorting_indices = indices_to_sort(self.k_gas_id_list, self.k_iso_id_list, 
                                              self.gas_id_list, self.iso_id_list)
        sorted_gas_id_list = self.gas_id_list[gas_sorting_indices]
        sorted_VMR_layer = VMR_layer[:,gas_sorting_indices]
        
#         np.save('vmr.npy',VMR_layer[:,:]*P_layer[:,None]/101325)
#         print(repr(A_layer[:,1]),flush=True)

        try:
            nf = self.fours[path_angle]
        except:
            nf = -1
        point_spectrum, fours = calc_radiance(wave_grid, U_layer, P_layer, T_layer,
            sorted_VMR_layer, A_layer, PARA_layer, self.phase_array, self.k_gas_w_g_p_t,
            self.k_wave_grid, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            R_plt=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=sorted_gas_id_list,cia_nu_grid=self.cia_nu_grid, cia_frac_grid = self.cia_frac_grid,
            cia_T_grid=self.cia_T_grid, dH=dH, emiss_ang = path_angle, sol_ang = sol_ang,
            aphi = aphi, lta = self.lta_flags[0], 
            xextnorms = self.xextnorms, mu = self.mu, wtmu = self.wtmu, 
            IRAY = self.iray, INORMAL = self.inormal, ISPACE = self.ispace, f_flag = f_flag, fours = nf)
        if f_flag:
            self.fours[path_angle] = fours
            
#         except ZeroDivisionError:
#             print('P_model',P_model)
#             print('H_model',H_model)
#             print('T_model',T_model)
#             print('VMR_model',VMR_model)
#             print('A_model', A_model)
#             print('path_angle',path_angle)
#             print('H_layer',H_layer)
#             print('U_layer',U_layer)
#             print('P_layer',P_layer)
#             print('T_layer',T_layer)
#             print('VMR_layer',VMR_layer)
#             print('A_layer', A_layer)
#             print('scale',scale)
#             raise(Exception('Division by zero'))
        if self.FWHM > 0:
            point_spectrum = lblconv(self.FWHM,self.ins_shape,len(wave_grid),wave_grid,point_spectrum,
                                                              len(self.wave_grid),self.wave_grid)
        return point_spectrum
