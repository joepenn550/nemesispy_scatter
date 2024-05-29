from nemesispy_scatter.common.parameterisations import *
import numpy as np
import os

def parse_spx_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    
    FWHM = float(lines[0].split()[0])
    latitude = float(lines[0].split()[1])
    longitude = float(lines[0].split()[2])
    ngeom = int(float(lines[0].split()[3]))
    
    spectra = [[] for i in range(ngeom)]
    
    sol_angs = []
    emiss_angs = []
    aphis = []
    
    igeom = -1
    for line in lines[3:]:
        if len(line.split()) == 6:
            igeom +=1
            sol_angs.append(float(line.split()[2]))
            emiss_angs.append(float(line.split()[3]))
            aphis.append(float(line.split()[4]))
            
        elif len(line.split()) == 3:
            spectra[igeom].append([float(x) for x in line.split()])
    
    return spectra, FWHM, latitude, longitude, sol_angs, emiss_angs, aphis

def parse_sha_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    return int(lines[0])

def parse_vpf_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    vpdict = {}
    for line in lines[1:]:
        gas_id, iso_id, vp, svpflag = line.split()
        gas_id = int(gas_id)
        iso_id = int(iso_id)
        vp = float(vp)
        svpflag = int(svpflag)
        vpdict[(gas_id,iso_id)] = (vp,svpflag)
    return vpdict
    
def parse_ref_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    data = []
    gas_ids = []
    iso_ids = []
    start_parsing = False
    planet = int(lines[2].split()[0])
    ngas = int(lines[2].split()[3])
    
    for line in lines:
        elements = line.split()
        
        # Parse gas_ids and iso_ids before the header
        if not start_parsing:
            try:
                gas_id = int(elements[0])
                iso_id = int(elements[1])
                gas_ids.append(gas_id)
                iso_ids.append(iso_id)
            except:
                if len(gas_ids) > 0:
                    start_parsing = True
                continue

        # Parse data lines after the header
        if start_parsing:
            try:
                height = float(elements[0]) * 1e3  # Convert to meters
                pressure = float(elements[1]) * 101325  # Convert to Pascals
                temperature = float(elements[2])
                vmr_values = [float(vmr) for vmr in elements[3:3+ngas]]  # Convert VMR values
                data.append([height, pressure, temperature] + vmr_values)
            except:
                # Stop parsing if the line doesn't contain valid data
                break

    return np.array(data), np.array(gas_ids), np.array(iso_ids),planet

def parse_other_ref_file(filepath, buffer = 2):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    data = [list(map(float, line.split())) for line in lines[buffer:] if len(line.strip()) > 2]

    # Convert data into an array with columns
    # Transpose the list of lists to get columns
    data_array = np.stack(data).transpose()
    return np.array(data_array)[1:].transpose()

def parse_fla_file(file_path):
    parameters = [0] * 9  

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Parse available lines
    for index, line in enumerate(lines):
        if index < len(parameters):  # Check to avoid index out of range if more lines than expected
            parameters[index] = int(line.split()[0].strip())

    inormal, iray, ih20, ich4, io3, inh3, iptf, imie, iuv = parameters

    return inormal, iray, ih20, ich4, io3, inh3, iptf, imie, iuv


def parse_lay_file(file_path):
    with open(file_path, "r") as f:
        f.readline() # header
        npro = int(f.readline().strip())
        custom_p_base = []
        for i in range(npro):
            custom_p_base.append(float(f.readline().strip()))

    return np.array(custom_p_base)
    
def parse_set_file(file_path):
    with open(file_path, "r") as f:
        f.readline() # header
        nmu = int(f.readline().split()[-1].strip())

        mu = []
        wtmu = []

        for i in range(nmu):
            mui, wtmui = f.readline().split()
            mu.append(float(mui))
            wtmu.append(float(wtmui))

        mu = np.array(mu)
        wtmu = np.array(wtmu)
        nf = int(f.readline().split()[-1].strip())
        nphi = int(f.readline().split()[-1].strip())
        insol = int(f.readline().split()[-1].strip())
        soldist = float(f.readline().split()[-1].strip())
        lowbc = int(f.readline().split()[-1].strip())
        galb = float(f.readline().split()[-1].strip())
        tsurf = float(f.readline().split()[-1].strip())
        f.readline()
        H0 = float(f.readline().split()[-1].strip())*1000
        npro = int(f.readline().split()[-1].strip())
        laytype = int(f.readline().split()[-1].strip())
        layint = int(f.readline().split()[-1].strip())
    return mu, wtmu, nf, nphi, insol, soldist, lowbc, galb, tsurf, H0, npro, laytype, layint
            
def parse_table_file(file_path,subfolder, cia=False): # NEED TO IMPLEMENT READING DNU, NPARA, ETC
    with open(file_path, "r") as f:
        table_paths = []
        line = f.readline().replace('\n', '')
        if cia:
            dnu = float(f.readline().split('!')[0].split()[0])
            npara = int(f.readline().split('!')[0].split()[0])
        else:
            dnu = 0.0
            npara = 0
        while line:
            if not os.path.exists(line):
                if os.path.exists(subfolder+line):
                    line = subfolder+line
            table_paths.append(line)
            line = f.readline().replace('\n', '')  
    return table_paths, dnu, npara

def parse_solspec_file(file_path, wave_grid, soldist, ispace):
    # Reading solar file data - set this up properly!
    with open(file_path, 'r') as file:
        header = [next(file) for _ in range(4)]
        xsolar = np.loadtxt(file).transpose()

    # Constants
    AU = 1.49598E13

    # Convert solar spectrum
    area = 4 * np.pi * (soldist * AU) ** 2
    solar = xsolar
    solar[1, :] /= area
    if ispace == 1:
        sol = np.interp(wave_grid, np.array(solar[0]),np.array(solar[1]))
    else:
        sol = np.interp(wave_grid, np.array(solar[0])[::-1],np.array(solar[1])[::-1])
    return sol
            
            
def parse_hazef_file(file_path):
    with open(file_path, "r") as haze_f:
        params = []
        param_errs = []
        dists = []
        dist_mults = []
        haze_waves = []
        calc_waves = []
        for i in range(2):
            line = haze_f.readline().split()
            xai = line[0] # need to sort out 3-parameter distributions?
            xa_erri = line[1]
            params.append(float(xai))
            param_errs.append(float(xa_erri))
            try:
                dists.append(float(line[2]))
                dist_mults.append(float(line[3]))
            except:
                dists.append(0)
                dist_mults.append(1)
                    
        nwave, clen = haze_f.readline().split('!')[0].split()
        vref, nreal_ref = haze_f.readline().split('!')[0].split()
        v_od_norm = haze_f.readline().split('!')[0]

        stop = False
        for i in range(int(nwave)):
            line = haze_f.readline().split()
            v, xai, xa_erri = line[:3]
            
            if not stop:
                params.append(float(xai))
                param_errs.append(float(xa_erri))
                haze_waves.append(float(v))
                try:
                    dists.append(float(line[3]))
                    dist_mults.append(float(line[4]))
                except:
                    dists.append(0)
                    dist_mults.append(1)
                
            if float(clen) < 0:
                stop = True
            calc_waves.append(float(v))
    
    return params, param_errs, dists, dist_mults, haze_waves, calc_waves,\
            float(clen), float(vref), float(nreal_ref), float(v_od_norm)
            
            
def parse_inp_file(file_path):
    with open(file_path, "r") as f:
        ispace, iscat, ilbl = map(int, f.readline().split('!')[0].strip().split()) 
        woff = float(f.readline().split('!')[0].strip())
        ename = f.readline().strip()
        niter = int(f.readline().split('!')[0].strip())
        philimit = float(f.readline().split('!')[0].strip())
        nspec, ioff = map(int, f.readline().split('!')[0].strip().split()) 
        lin = int(f.readline().split('!')[0].strip())
    return ispace, iscat, ilbl, woff, ename, niter, philimit, nspec, ioff, lin
    
            
            
            
            
            
            
            
            
            