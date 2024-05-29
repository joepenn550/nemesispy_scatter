#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate collision-induced-absorption optical path.
"""
import numpy as np
from numba import jit

@jit(nopython=True)
def bilinear_xy(Q, x1, x2, y1, y2, x, y):
    fxy1 = ((x2 - x + 1e-30) / (x2 - x1 + 2e-30)) * Q[0] + ((x - x1 + 1e-30) / (x2 - x1 + 2e-30)) * Q[1]
    fxy2 = ((x2 - x + 1e-30) / (x2 - x1 + 2e-30)) * Q[2] + ((x - x1 + 1e-30) / (x2 - x1 + 2e-30)) * Q[3]
    return ((y2 - y + 1e-30) / (y2 - y1 + 2e-30)) * fxy1 + ((y - y1 + 1e-30) / (y2 - y1 + 2e-30)) * fxy2

@jit(nopython=True)
def trilinear_interpolation(grid, x_values, y_values, z_values, x_array, y_array, z_array):
    """
    Performs trilinear interpolation on a 3D grid for arrays of x, y, and z coordinates.
    Points outside the grid are assigned a value of 0.
    
    :param grid: 3D array of grid values.
    :param x_values: 1D array of x-axis values in the grid.
    :param y_values: 1D array of y-axis values in the grid.
    :param z_values: 1D array of z-axis values in the grid.
    :param x_array: 1D array of x-coordinates for interpolation.
    :param y_array: 1D array of y-coordinates for interpolation.
    :param z_array: 1D array of z-coordinates for interpolation.
    :return: 3D array of interpolated values.
    """
    # Initialize the resulting 3D array
    result = np.zeros((z_array.size, x_array.size))

    # Iterate over all x, y, and z values
    for k in range(z_array.size):
        for i in range(x_array.size):
            x = x_array[i]
            if grid.shape[0] == 1:
                x = 0
            y = y_array[i]
            z = z_array[k]
#             print('x',x, x_values[0],x_values[-1],flush=True)
#             print('y',y, y_values[0],y_values[-1],flush=True)
#             print('z',z, z_values[0],z_values[-1],flush=True)
            # Check if x, y, and z are within the bounds of the grid
            if x < x_values[0] or x > x_values[-1]\
            or y < y_values[0] or y > y_values[-1]\
            or z < z_values[0] or z > z_values[-1]:
                result[k, i] = 0
            else:
                # Find the indices of the nearest x, y, z values
                ix = np.searchsorted(x_values, x) - 1
                iy = np.searchsorted(y_values, y) - 1
                iz = np.searchsorted(z_values, z) - 1

                ix = max(min(ix, grid.shape[0] - 2), 0)
                # fix for special case
                if ix == 0 and grid.shape[0] == 1:
                    ix = -1

                iy = max(min(iy, grid.shape[1] - 2), 0)
                iz = max(min(iz, grid.shape[2] - 2), 0)

                # Extract the corner values of the grid cell
                x1, x2 = x_values[ix], x_values[ix + 1]
                y1, y2 = y_values[iy], y_values[iy + 1]
                z1, z2 = z_values[iz], z_values[iz + 1]

                Q000, Q100, Q010, Q110 = grid[ix, iy, iz], grid[ix+1, iy, iz], \
                                         grid[ix, iy+1, iz], grid[ix+1, iy+1, iz]

                Q001, Q101, Q011, Q111 = grid[ix, iy, iz+1], grid[ix+1, iy, iz+1], \
                                         grid[ix, iy+1, iz+1], grid[ix+1, iy+1, iz+1]
                fz1 = bilinear_xy(np.array([Q000, Q100, Q010, Q110]), x1, x2, y1, y2, x, y)
                fz2 = bilinear_xy(np.array([Q001, Q101, Q011, Q111]), x1, x2, y1, y2, x, y)

                # Linear interpolation in z
                result[k, i] = ((z2 - z + 1e-30) / (z2 - z1 + 2e-30)) * fz1\
                             + ((z - z1 + 1e-30) / (z2 - z1 + 2e-30)) * fz2
    return result


@jit(nopython=True, cache = False)
def calc_tau_cia(wave_grid, K_CIA, ISPACE,
    ID, TOTAM, T_layer, P_layer, VMR_layer, PARA_layer, DELH,
    cia_nu_grid, TEMPS, FRACS, INORMAL=0):
    """
    Calculates
    Parameters
    ----------
    wave_grid : ndarray
        Wavenumber (cm-1) or wavelength array (um) at which to compute
        CIA opacities.
    ID : ndarray
        Gas ID
    # ISO : ndarray
    #     Isotop ID.
    VMR_layer : TYPE
        DESCRIPTION.
    ISPACE : int
        Flag indicating whether the calculation must be performed in
        wavenumbers (0) or wavelength (1)
    K_CIA(NPAIR,NTEMP,NWAVE) : ndarray
         CIA cross sections for each pair at each temperature level and wavenumber.
    cia_nu_grid : TYPE
        DESCRIPTION.
    INORMAL : int


    Returns
    -------
    tau_cia_layer(NWAVE,NLAY) : ndarray
        CIA optical depth in each atmospheric layer.
    """
      
    # Need to pass NLAY from a atm profile
    NPAIR = K_CIA.shape[0]

    NLAY,NVMR = VMR_layer.shape
    ISO = np.zeros((NVMR))

    # mixing ratios of the relevant gases
    qh2 = np.zeros((NLAY))
    qhe = np.zeros((NLAY))
    qn2 = np.zeros((NLAY))
    qch4 = np.zeros((NLAY))
    qco2 = np.zeros((NLAY))
    # IABSORB = np.ones(5,dtype='int32') * -1

    # get mixing ratios from VMR grid
    for iVMR in range(NVMR):
        if ID[iVMR] == 39: # hydrogen
            qh2[:] += VMR_layer[:,iVMR]
            # IABSORB[0] = iVMR
        if ID[iVMR] == 40: # helium
            qhe[:] += VMR_layer[:,iVMR]
            # IABSORB[1] = iVMR
        if ID[iVMR] == 22: # nitrogen
            qn2[:] += VMR_layer[:,iVMR]
            # IABSORB[2] = iVMR
        if ID[iVMR] == 6: # methane
            qch4[:] += VMR_layer[:,iVMR]
            # IABSORB[3] = iVMR
        if ID[iVMR] == 2: # co2
            qco2[:] += VMR_layer[:,iVMR]
            # IABSORB[4] = iVMR
    # calculating the opacity
    XLEN = DELH * 1.0e2 # cm
    TOTAM = TOTAM * 1.0e-4 # cm-2

    ### back to FORTRAN ORIGINAL
    P0=101325
    T0=273.15
    AMAGAT = 2.68675E19 #mol cm-3
    KBOLTZMANN = 1.381E-23
    MODBOLTZA = 10.*KBOLTZMANN/1.013

    tau = (P_layer/P0)**2 * (T0/T_layer)**2 * DELH
    height1 = P_layer * MODBOLTZA * T_layer

    height = XLEN * 1e2
    amag1 = TOTAM /height/AMAGAT
    tau = height*amag1**2

    AMAGAT = 2.68675E19 #mol cm-3
    amag1 = TOTAM / XLEN / AMAGAT # number density
    tau = XLEN*amag1**2# optical path, why fiddle around with XLEN

    # define the calculatiion wavenumbers
    if ISPACE == 0: # input wavegrid is already in wavenumber (cm^-1)
        WAVEN = wave_grid
    elif ISPACE == 1:
        WAVEN = 1.e4/wave_grid
        isort = np.argsort(WAVEN)
        WAVEN = WAVEN[isort] # ascending wavenumbers

    # if WAVEN.min() < cia_nu_grid.min() or WAVEN.max()>cia_nu_grid.max():
    #     print('warning in CIA :: Calculation wavelengths expand a larger range than in .cia file')

    # calculate the CIA opacity at the correct temperature and wavenumber
    NWAVEC = len(wave_grid)  # Number of calculation wavelengths
    tau_cia_layer = np.zeros((NWAVEC,NLAY))
    
    k_cia = np.zeros((NWAVEC,NPAIR,NLAY))

    
    for ipair in range(NPAIR):
        k_cia[:,ipair,:] = trilinear_interpolation(K_CIA[ipair], FRACS, TEMPS, cia_nu_grid, PARA_layer, T_layer, WAVEN)
    for ilay in range(NLAY):  
        if len(FRACS)==1:
            #Combining the CIA absorption of the different pairs (included in .cia file)
            sum1 = np.zeros(NWAVEC)
            if INORMAL==0: # equilibrium hydrogen (1:1)
                sum1[:] = sum1[:] + k_cia[:,0,ilay] * qh2[ilay] * qh2[ilay] \
                    + k_cia[:,1,ilay] * qhe[ilay] * qh2[ilay]
            elif INORMAL==1: # normal hydrogen (3:1)
                sum1[:] = sum1[:] + k_cia[:,2,ilay] * qh2[ilay] * qh2[ilay]\
                    + k_cia[:,3,ilay] * qhe[ilay] * qh2[ilay]

            sum1[:] = sum1[:] + k_cia[:,4,ilay] * qh2[ilay] * qn2[ilay]
            sum1[:] = sum1[:] + k_cia[:,5,ilay] * qn2[ilay] * qch4[ilay]
            sum1[:] = sum1[:] + k_cia[:,6,ilay] * qn2[ilay] * qn2[ilay]
            sum1[:] = sum1[:] + k_cia[:,7,ilay] * qch4[ilay] * qch4[ilay]
            sum1[:] = sum1[:] + k_cia[:,8,ilay] * qh2[ilay] * qch4[ilay]

            # look up CO2-CO2 CIA coefficients (external)
            """
            TO BE DONE
            """
            k_co2 = sum1*0
            # k_co2 = co2cia(WAVEN)

            sum1[:] = sum1[:] + k_co2[:] * qco2[ilay] * qco2[ilay]

            #Look up N2-N2 NIR CIA coefficients
            k_n2 = np.interp(WAVEN,wnn,nn_abs)
            k_n2[WAVEN<np.min(wnn)] = 0.0
            k_n2[WAVEN>np.max(wnn)] = 0.0
            sum1[:] = sum1[:] + k_n2[:] * qn2[ilay] * qn2[ilay] * 1e-5

            #Look up N2-H2 NIR CIA coefficients
            k_n2h2 = np.interp(WAVEN,wnh,nh_abs)
            k_n2h2[WAVEN<np.min(wnh)] = 0.0
            k_n2h2[WAVEN>np.max(wnh)] = 0.0
            sum1[:] = sum1[:] + k_n2h2[:] * qn2[ilay] * qh2[ilay] * 1e-5
            # TO BE DONE


        else:
            sum1 = np.zeros(NWAVEC)
            sum1[:] = sum1[:] + k_cia[:,0,ilay] * qh2[ilay] * qh2[ilay]
            sum1[:] = sum1[:] + k_cia[:,1,ilay] * qh2[ilay] * qhe[ilay]

        tau_cia_layer[:,ilay] = sum1[:] * tau[ilay]     
    if ISPACE==1:
        tau_cia_layer[:,:] = tau_cia_layer[isort,:]
    return tau_cia_layer


wnn = np.array([\
 4500.0,4505.0,4510.0,4515.0,4520.0,4525.0,4530.0,4535.0,
 4540.0,4545.0,4550.0,4555.0,4560.0,4565.0,4570.0,4575.0,
 4580.0,4585.0,4590.0,4595.0,4600.0,4605.0,4610.0,4615.0,
 4620.0,4625.0,4630.0,4635.0,4640.0,4645.0,4650.0,4655.0,
 4660.0,4665.0,4670.0,4675.0,4680.0,4685.0,4690.0,4695.0,
 4700.0,4705.0,4710.0,4715.0,4720.0,4725.0,4730.0,4735.0,
 4740.0,4745.0,4750.0,4755.0,4760.0,4765.0,4770.0,4775.0,
 4780.0,4785.0,4790.0,4795.0,4800.0,4805.0,4810.0,4815.0,
 4820.0,4825.0])  
wnh = np.array([\
 3995.00,4000.00,4005.00,4010.00,4015.00,4020.00,
 4025.00,4030.00,4035.00,4040.00,4045.00,4050.00,
 4055.00,4060.00,4065.00,4070.00,4075.00,4080.00,
 4085.00,4090.00,4095.00,4100.00,4105.00,4110.00,
 4115.00,4120.00,4125.00,4130.00,4135.00,4140.00,
 4145.00,4150.00,4155.00,4160.00,4165.00,4170.00,
 4175.00,4180.00,4185.00,4190.00,4195.00,4200.00,
 4205.00,4210.00,4215.00,4220.00,4225.00,4230.00,
 4235.00,4240.00,4245.00,4250.00,4255.00,4260.00,
 4265.00,4270.00,4275.00,4280.00,4285.00,4290.00,
 4295.00,4300.00,4305.00,4310.00,4315.00,4320.00,
 4325.00,4330.00,4335.00,4340.00,4345.00,4350.00,
 4355.00,4360.00,4365.00,4370.00,4375.00,4380.00,
 4385.00,4390.00,4395.00,4400.00,4405.00,4410.00,
 4415.00,4420.00,4425.00,4430.00,4435.00,4440.00,
 4445.00,4450.00,4455.00,4460.00,4465.00,4470.00,
 4475.00,4480.00,4485.00,4490.00,4495.00,4500.00,
 4505.00,4510.00,4515.00,4520.00,4525.00,4530.00,
 4535.00,4540.00,4545.00,4550.00,4555.00,4560.00,
 4565.00,4570.00,4575.00,4580.00,4585.00,4590.00,
 4595.00,4600.00,4605.00,4610.00,4615.00,4620.00,
 4625.00,4630.00,4635.00,4640.00,4645.00,4650.00,
 4655.00,4660.00,4665.00,4670.00,4675.00,4680.00,
 4685.00,4690.00,4695.00,4700.00,4705.00,4710.00,
 4715.00,4720.00,4725.00,4730.00,4735.00,4740.00,
 4745.00,4750.00,4755.00,4760.00,4765.00,4770.00,
 4775.00,4780.00,4785.00,4790.00,4795.00,4800.00,
 4805.00,4810.00,4815.00,4820.00,4825.00,4830.00,
 4835.00,4840.00,4845.00,4850.00,4855.00,4860.00,
 4865.00,4870.00,4875.00,4880.00,4885.00,4890.00,
 4895.00,4900.00,4905.00,4910.00,4915.00,4920.00,
 4925.00,4930.00,4935.00,4940.00,4945.00,4950.00,
 4955.00,4960.00,4965.00,4970.00,4975.00,4980.00,
 4985.00,4990.00,4995.00])

nn_abs = np.array([\
 1.5478185E-05,3.4825567E-05,5.4172953E-05,7.3520343E-05,
 9.2867725E-05,1.1221511E-04,1.3156250E-04,1.5090988E-04,
 1.7025726E-04,1.8960465E-04,2.0895203E-04,2.3593617E-04,
 2.9850862E-04,3.6948317E-04,4.4885988E-04,5.4001610E-04,
 6.4105232E-04,7.5234997E-04,8.7262847E-04,9.9942752E-04,
 1.1362602E-03,1.2936132E-03,1.5176521E-03,1.7954395E-03,
 2.1481151E-03,2.6931590E-03,3.1120952E-03,2.7946872E-03,
 2.5185575E-03,2.4253442E-03,2.4188559E-03,2.4769977E-03,
 2.4829037E-03,2.3845681E-03,2.2442993E-03,2.1040305E-03,
 1.9726211E-03,1.8545000E-03,1.7363789E-03,1.6182578E-03,
 1.5128252E-03,1.4635258E-03,1.2099572E-03,1.0359654E-03,
 9.1723543E-04,7.5135247E-04,6.0498451E-04,5.0746030E-04,
 4.0987082E-04,3.2203691E-04,2.5376283E-04,2.0496233E-04,
 1.5671484E-04,1.1761552E-04,9.7678370E-05,7.8062728E-05,
 5.8552457E-05,4.8789554E-05,4.1275161E-05,3.9085765E-05,
 3.9056369E-05,3.5796973E-05,3.0637581E-05,2.5478185E-05,
 2.0318790E-05,5.1593952E-06])

nh_abs = np.array([\
 3.69231E-04,3.60000E-03,6.83077E-03,1.00615E-02,
 1.36610E-02,1.84067E-02,2.40000E-02,3.18526E-02,
 3.97052E-02,4.75578E-02,4.88968E-02,7.44768E-02,
 9.08708E-02,0.108070,0.139377,0.155680,0.195880,0.228788,
 0.267880,0.324936,0.367100,0.436444,0.500482,0.577078,
 0.656174,0.762064,0.853292,0.986708,1.12556,1.22017,
 1.33110,1.65591,1.69356,1.91446,1.75494,1.63788,
 1.67026,1.62200,1.60460,1.54774,1.52408,1.48716,
 1.43510,1.42334,1.34482,1.28970,1.24494,1.16838,
 1.11038,1.06030,0.977912,0.924116,0.860958,0.807182,
 0.759858,0.705942,0.680112,0.619298,0.597530,0.550046,
 0.512880,0.489128,0.454720,0.432634,0.404038,0.378780,
 0.359632,0.333034,0.317658,0.293554,0.277882,0.262120,
 0.240452,0.231128,0.210256,0.202584,0.192098,0.181876,
 0.178396,0.167158,0.171314,0.165576,0.166146,0.170206,
 0.171386,0.181330,0.188274,0.205804,0.223392,0.253012,
 0.292670,0.337776,0.413258,0.490366,0.600940,0.726022,
 0.890254,1.14016,1.21950,1.45480,1.35675,1.53680,
 1.50765,1.45149,1.38065,1.19780,1.08241,0.977574,
 0.878010,0.787324,0.708668,0.639210,0.578290,0.524698,
 0.473266,0.431024,0.392020,0.357620,0.331398,0.299684,
 0.282366,0.260752,0.242422,0.234518,0.217008,0.212732,
 0.204464,0.198802,0.199584,0.188652,0.195038,0.191616,
 0.200324,0.213712,0.224948,0.252292,0.276978,0.318584,
 0.369182,0.432017,0.527234,0.567386,0.655152,0.660094,
 0.739228,0.698344,0.662759,0.663277,0.584378,0.535622,
 0.481566,0.443086,0.400727,0.364086,0.338196,0.303834,
 0.289236,0.262176,0.247296,0.231594,0.211104,0.205644,
 0.185118,0.178470,0.170610,0.152406,0.153222,0.132552,
 0.131400,0.122286,0.109758,0.107472,9.21480E-02,9.09240E-02,
 8.40520E-02,7.71800E-02,7.03080E-02,6.34360E-02,5.76892E-02,
 5.32345E-02,4.90027E-02,4.49936E-02,4.12073E-02,3.76437E-02,
 3.43029E-02,3.11848E-02,2.80457E-02,2.49195E-02,2.19570E-02,
 1.91581E-02,1.65230E-02,1.40517E-02,1.17440E-02,9.60000E-03,
 8.40000E-03,7.20000E-03,6.00000E-03,4.80000E-03,3.60000E-03,
 2.40000E-03,1.20000E-03]) 