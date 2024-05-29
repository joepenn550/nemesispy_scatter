#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate the optical path due to atomic and molecular lines.
The opacity of gases is calculated by the correlated-k method
using pre-tabulated ktables, assuming random overlap of lines.
"""
import numpy as np
from numba import jit

@jit(nopython=True, cache = False)
def calc_tau_gas(k_gas_w_g_p_t, k_wave_grid, wave_grid, P_layer, T_layer, VMR_layer, U_layer,
    P_grid, T_grid, del_g, lta):
    """
    Parameters
    ----------
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Unit: cm^2 (per particle)
        Has dimension: NGAS x NWAVE x NG x NPRESSK x NTEMPK.
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    U_layer(NLAYER) : ndarray
        Total number of gas particles in each layer.
        Unit: (no. of particle) m^-2
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.
    n_active : int
        Number of spectrally active gases.

    Returns
    -------
    tau_w_g_l(NWAVE,NG,NLAYER) : ndarray
        Optical path due to spectral line absorptions.

    Notes
    -----
    Absorber amounts (U_layer) is scaled down by a factor 1e-20 because Nemesis
    k-tables are scaled up by a factor of 1e20.
    """
    # convert from per m^2 to per cm^2 and downscale Nemesis opacity by 1e-20
    Scaled_U_layer = U_layer  * 1.0e-4
    Ngas, Nwavek, Ng = k_gas_w_g_p_t.shape[0:3]
    Nwave = len(wave_grid)
    Nlayer = len(P_layer)
    tau_w_g_l = np.zeros((Nwave,Ng,Nlayer))
    # if only has 1 active gas, skip random overlap
    if Ngas == 1:
        if lta:
            k_w_g_l = interp_k_lta(P_grid, T_grid, P_layer, T_layer,
                k_gas_w_g_p_t[0], k_wave_grid, wave_grid) 
        else:
            k_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer,
                k_gas_w_g_p_t[0], k_wave_grid, wave_grid, del_g,lta)
            
#             for i in range(Nlayer):
#                 print(i,k_w_g_l[0,:,i]* VMR_layer[i,0], flush = True)
            
        for ilayer in range(Nlayer):
            tau_w_g_l[:,:,ilayer]  = k_w_g_l[:,:,ilayer] \
                * Scaled_U_layer[ilayer] * VMR_layer[ilayer,0]

    # if there are multiple gases, combine their opacities
    else:
        k_gas_w_g_l = np.zeros((Ngas,Nwave,Ng,Nlayer))
        for igas in range(Ngas):
            if lta:
                k_gas_w_g_l[igas,:,:,:,] \
                    = interp_k_lta(P_grid, T_grid, P_layer, T_layer,
                        k_gas_w_g_p_t[igas,:,:,:,:], k_wave_grid, wave_grid)
                
                amount_layer = Scaled_U_layer * VMR_layer[:,igas]                
                tau_w_g_l += k_gas_w_g_l[igas]*amount_layer
                    
            else:
                k_gas_w_g_l[igas,:,:,:,] \
                    = interp_k(P_grid, T_grid, P_layer, T_layer,
                        k_gas_w_g_p_t[igas,:,:,:,:], k_wave_grid, wave_grid, del_g, lta)
        if not lta:
            for iwave in range (Nwave):
                for ilayer in range(Nlayer):
                    amount_layer = Scaled_U_layer[ilayer] * VMR_layer[ilayer,:Ngas]
                    tau_w_g_l[iwave,:,ilayer]\
                        = noverlapg(k_gas_w_g_l[:,iwave,:,ilayer],
                            amount_layer,del_g)
    tau_w_g_l = np.clip(tau_w_g_l*1e-20,0,1e10)
    return tau_w_g_l

@jit(nopython=True)
def interp_k(P_grid, T_grid, P_layer, T_layer, k_w_g_p_t, k_wave_grid, wave_grid, del_g, lta):
    """
    Interpolate the k coeffcients at given atmospheric pressures, temperatures, and wavelengths
    using a k-table.

    Parameters
    ----------
    P_grid(NPRESSKTA) : ndarray
        Pressure grid of the k-tables.
        Unit: Pa
    T_grid(NTEMPKTA) : ndarray
        Temperature grid of the ktables.
        Unit: Kelvin
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: Kelvin
    k_w_g_p_t(NGAS,NWAVEKTA,NG,NPRESSKTA,NTEMPKTA) : ndarray
        Array storing the k-coefficients.

    Returns
    -------
    k_w_g_l(NGAS,NWAVEKTA,NG,NLAYER) : ndarray
        The interpolated-to-atmosphere k-coefficients.
        Has dimension: NWAVE x NG x NLAYER.

    Notes
    -----
    Code breaks if P_layer/T_layer is out of the range of P_grid/T_grid.
    Mainly need to worry about max(T_layer)>max(T_grid).
    No extrapolation outside of the TP grid of ktable.
    """
    NWAVEK, NG, NPRESS, NTEMP = k_w_g_p_t.shape
    NLAYER = len(P_layer)
    NWAVE = len(wave_grid)
    k_w_g_l = np.zeros((NWAVE,NG,NLAYER))
    
    T_grid = T_grid[0]
    # Interpolate the k values at the layer temperature and pressure
    for ilayer in range(NLAYER):
        p = P_layer[ilayer]
        t = T_layer[ilayer]

        # Find pressure grid points above and below current layer pressure
        ip = np.abs(P_grid-p).argmin()
        if P_grid[ip] >= p:
            ip_high = ip
            if ip == 0:
                p = P_grid[0]
                ip_low = 0
                ip_high = 1
            else:
                ip_low = ip-1
        elif P_grid[ip]<p:
            ip_low = ip
            if ip == NPRESS-1:
                p = P_grid[NPRESS-1]
                ip_high = NPRESS-1
                ip_low = NPRESS-2
            else:
                ip_high = ip + 1

        # Find temperature grid points above and below current layer temperature
        it = np.abs(T_grid-t).argmin()
        if T_grid[it] >= t:
            it_high = it
            if it == 0:
                t = T_grid[0]
                it_low = 0
                it_high = 1
            else:
                it_low = it -1
        elif T_grid[it] < t:
            it_low = it
            if it == NTEMP-1:
                t = T_grid[-1]
                it_high = NTEMP - 1
                it_low = NTEMP -2
            else:
                it_high = it + 1

        # Set up arrays for interpolation
        lnp = np.log(p)
        lnp_low = np.log(P_grid[ip_low])
        lnp_high = np.log(P_grid[ip_high])
        t_low = T_grid[it_low]
        t_high = T_grid[it_high]
        for iwave in range(NWAVE):
            wave = wave_grid[iwave]
            # Find indices of the two closest wavenumbers in k_wave_grid
            iw_closest = np.searchsorted(k_wave_grid, wave)  # Find insertion point
            iw_low = max(iw_closest - 1, 0)
            iw_high = min(iw_closest, len(k_wave_grid) - 1)
            if iw_high == iw_low:
                iw_high = min(iw_high + 1, len(k_wave_grid) - 1)

            # Calculate weights for wavenumber interpolation
            wave_low = k_wave_grid[iw_low]
            wave_high = k_wave_grid[iw_high]
            w = (wave - wave_low) / (wave_high - wave_low) if wave_high != wave_low else 0
            # Interpolate k-values across pressure, temperature, and wavenumber
#             for ig in range(NG):
            f111 = k_w_g_p_t[iw_low, :, ip_low, it_low]
            f112 = k_w_g_p_t[iw_low, :, ip_low, it_high]
            f121 = k_w_g_p_t[iw_low, :, ip_high, it_low]
            f122 = k_w_g_p_t[iw_low, :, ip_high, it_high]
            f211 = k_w_g_p_t[iw_high, :, ip_low, it_low]
            f212 = k_w_g_p_t[iw_high, :, ip_low, it_high]
            f221 = k_w_g_p_t[iw_high, :, ip_high, it_low]
            f222 = k_w_g_p_t[iw_high, :, ip_high, it_high]

            v = (lnp-lnp_low)/(lnp_high-lnp_low)
            u = (t-t_low)/(t_high-t_low)
            # Bilinear interpolation in P-T 
            k_interpolated_1 = ((1 - v) * (1 - u) * f111 + v * (1 - u) * f121 + (1 - v) * u * f112 + v * u * f122)
            k_interpolated_2 = ((1 - v) * (1 - u) * f211 + v * (1 - u) * f221 + (1 - v) * u * f212 + v * u * f222)
            
            k_interp = np.zeros(NG*2)
            weight = np.zeros(NG*2)
            
            k_interp[:NG] = k_interpolated_1
            weight[:NG] = del_g*(1-w)
            
            k_interp[NG:] = k_interpolated_2
            weight[NG:] = del_g*w
#             print(k_interp.shape,weight.shape,del_g.shape,flush=True)
            if w != 0 and w != 1:
                k_w_g_l[iwave, :, ilayer] =  rank(weight,k_interp,del_g)
            elif w == 0:
                k_w_g_l[iwave, :, ilayer] = k_interpolated_1
            elif w == 1:
                k_w_g_l[iwave, :, ilayer] = k_interpolated_2
                

    return k_w_g_l

@jit(nopython=True)
def rank(weight, cont, del_g):
    """
    Combine the randomly overlapped k distributions of two gases into a single
    k distribution.

    Parameters
    ----------
    weight(NG) : ndarray
        Weights of points in the random k-dist
    cont(NG) : ndarray
        Random k-coeffs in the k-dist.
    del_g(NG) : ndarray
        Required weights of final k-dist.

    Returns
    -------
    k_g(NG) : ndarray
        Combined k-dist.
        Unit: cm^2 (per particle)
    """
    ng = len(del_g)
    nloop = len(weight.flatten())

    # sum delta gs to get cumulative g ordinate
    g_ord = np.zeros(ng+1)
    g_ord[1:] = np.cumsum(del_g)
    g_ord[ng] = 1
    
    # Sort random k-coeffs into ascending order. Integer array ico records
    # which swaps have been made so that we can also re-order the weights.
    ico = np.argsort(cont)
    cont = cont[ico]
    weight = weight[ico] # sort weights accordingly
    gdist = np.cumsum(weight)
    k_g = np.zeros(ng)
    ig = 0
    sum1 = 0.0
    cont_weight = cont * weight
    for iloop in range(nloop):
        if gdist[iloop] < g_ord[ig+1] and ig < ng:
            k_g[ig] = k_g[ig] + cont_weight[iloop]
            sum1 = sum1 + weight[iloop]
        else:
            frac = (g_ord[ig+1] - gdist[iloop-1])/(gdist[iloop]-gdist[iloop-1])
            k_g[ig] = k_g[ig] + frac*cont_weight[iloop]

            sum1 = sum1 + frac * weight[iloop]
            k_g[ig] = k_g[ig]/sum1

            ig = ig +1
            if ig < ng:
                sum1 = (1.0-frac)*weight[iloop]
                k_g[ig] = (1.0-frac)*cont_weight[iloop]

    if ig == ng-1:
        k_g[ig] = k_g[ig]/sum1

    return k_g

@jit(nopython=True)
def noverlapg(k_gas_g, amount, del_g):
    """
    Combine k distributions of multiple gases given their number densities.

    Parameters
    ----------
    k_gas_g(NGAS,NG) : ndarray
        K-distributions of the different gases.
        Each row contains a k-distribution defined at NG g-ordinates.
        Unit: cm^2 (per particle)
    amount(NGAS) : ndarray
        Absorber amount of each gas,
        i.e. amount = VMR x layer absorber per area
        Unit: (no. of partiicles) cm^-2
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.

    Returns
    -------
    tau_g(NG) : ndarray
        Opatical path from mixing k-distribution weighted by absorber amounts.
        Unit: dimensionless
    """
    NGAS = len(amount)
    NG = len(del_g)
    tau_g = np.zeros(NG)
    random_weight = np.zeros(NG*NG)
    random_tau = np.zeros(NG*NG)
    cutoff = 1e-12
    for igas in range(NGAS-1):
        # first pair of gases
        if igas == 0:
            # if opacity due to first gas is negligible
            if k_gas_g[igas,:][-1] * amount[igas] < cutoff:
                tau_g = k_gas_g[igas+1,:] * amount[igas+1]
            # if opacity due to second gas is negligible
            elif k_gas_g[igas+1,:][-1] * amount[igas+1] < cutoff:
                tau_g = k_gas_g[igas,:] * amount[igas]
            # else resort-rebin with random overlap approximation
            else:
                iloop = 0
                for ig in range(NG):
                    for jg in range(NG):
                        random_weight[iloop] = del_g[ig] * del_g[jg]
                        random_tau[iloop] = k_gas_g[igas,:][ig] * amount[igas] \
                            + k_gas_g[igas+1,:][jg] * amount[igas+1]
                        iloop = iloop + 1
                tau_g = rank(random_weight,random_tau,del_g)
        # subsequent gases, add amount*k to previous summed k
        else:
            # if opacity due to next gas is negligible
            if k_gas_g[igas+1,:][-1] * amount[igas+1] < cutoff:
                pass
            # if opacity due to previous gases is negligible
            elif tau_g[-1] < cutoff:
                tau_g = k_gas_g[igas+1,:] * amount[igas+1]
            # else resort-rebin with random overlap approximation
            else:
                iloop = 0
                for ig in range(NG):
                    for jg in range(NG):
                        random_weight[iloop] = del_g[ig] * del_g[jg]

                        random_tau[iloop] = tau_g[ig] \
                            + k_gas_g[igas+1,:][jg] * amount[igas+1]
                        iloop = iloop + 1
                tau_g = rank(random_weight,random_tau,del_g)
    return tau_g


@jit(nopython=True)
def interp_k_lta(P_grid, T_grid, P_layer, T_layer, k_w_g_p_t, k_wave_grid, wave_grid):
    
    NWAVEK, NG, NPRESS, NTEMP = k_w_g_p_t.shape
    NWAVE = len(wave_grid)
    NLAYER = len(P_layer)
    k_w_g_l = np.zeros((NWAVE,NG,NLAYER))
    P_grid = np.log(P_grid)
    # Interpolate the k values at the layer temperature and pressure
    for ilayer in range(NLAYER):
        p_l = np.log(P_layer[ilayer])
        if p_l < np.min(P_grid):
            p_l = np.min(P_grid)
        if p_l > np.max(P_grid):
            p_l = np.max(P_grid)
   
        t_l = T_layer[ilayer]
        
        if t_l < np.min(T_grid):
            t_l = np.min(T_grid)
        if t_l > np.max(T_grid):
            t_l = np.max(T_grid) 
        
        
        ip = np.argmin(np.abs(P_grid-p_l))
        if ip >= len(P_grid)-1:
            ip = len(P_grid)-2
        v = (p_l-P_grid[ip])/(P_grid[ip+1]-P_grid[ip])
        Tn = T_grid[ip]
        it = np.argmin(np.abs(Tn-t_l))
        
        if it >= len(Tn)-1:
            it = len(Tn)-2
        u = (t_l-Tn[it])/(Tn[it+1]-Tn[it])
        Tn = T_grid[ip+1]
        it2 = np.argmin(np.abs(Tn-t_l))
        if it2 >= len(Tn)-1:
            it2 = len(Tn)-2
       
        u2 = (t_l-Tn[it2])/(Tn[it2+1]-Tn[it2])
#         u2 = min(max(u2,0),1)
#         u = min(max(u,0),1)
#         v = min(max(v,0),1)
        for iwave in range(NWAVE):
            wave = wave_grid[iwave]
            # Find indices of the two closest wavenumbers in k_wave_grid
            iw_closest = np.searchsorted(k_wave_grid, wave)  # Find insertion point
            iw_low = max(iw_closest - 1, 0)
            iw_high = min(iw_closest, len(k_wave_grid) - 1)
            if iw_high == iw_low:
                iw_high = min(iw_high + 1, len(k_wave_grid) - 1)
            # Calculate weights for wavenumber interpolation
            wave_low = k_wave_grid[iw_low]
            wave_high = k_wave_grid[iw_high]
            w = (wave - wave_low) / (wave_high - wave_low) if wave_high != wave_low else 0   
#             w = min(max(w,0),1)
            f111 = np.log(k_w_g_p_t[iw_low,  :, ip, it])
            f112 = np.log(k_w_g_p_t[iw_low,  :, ip, it+1])
            f121 = np.log(k_w_g_p_t[iw_low,  :, ip+1, it2])
            f122 = np.log(k_w_g_p_t[iw_low,  :, ip+1, it2+1])
            f211 = np.log(k_w_g_p_t[iw_high, :, ip, it])
            f212 = np.log(k_w_g_p_t[iw_high, :, ip, it+1])
            f221 = np.log(k_w_g_p_t[iw_high, :, ip+1, it2])
            f222 = np.log(k_w_g_p_t[iw_high, :, ip+1, it2+1])
            k_interpolated_1 = np.exp((1 - v) * (1 - u) * f111 + v * (1 - u2) * f121 + (1 - v) * u * f112 + v * u2 * f122)
            k_interpolated_2 = np.exp((1 - v) * (1 - u) * f211 + v * (1 - u2) * f221 + (1 - v) * u * f212 + v * u2 * f222)
            
            k_w_g_l[iwave, :, ilayer] = (1-w)*k_interpolated_1 + w*k_interpolated_2
        
    
    return k_w_g_l
