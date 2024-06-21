import os
import numpy as np
from nemesispy_scatter.common.read_nemesis import *
from nemesispy_scatter.common.read_apr import *
from mpi4py import MPI
from tqdm import tqdm
from scipy.stats import norm
from scipy.sparse import csc_array
from scipy.sparse.linalg import inv, bicgstab, splu
import ultranest
import pymultinest

os.environ["OMP_NUM_THREADS"] = "1"


class NEMESIS:
    def __init__(self, filepath, safe_load = False):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.safe_load = safe_load
        
        for j in range(self.size+1):
            if self.safe_load:
                self.comm.barrier()
                if j!=self.rank:
                    continue
            else:
                if j > 0: 
                    break
            self.read_nemesispy_input(filepath)
        
        
    def read_nemesispy_input(self, filepath):
        with open(filepath, "r") as f:
            f.readline()  # header
            self.folder_flag = int(f.readline().split()[0].strip())
            self.directory = f.readline().split()[0].strip()
            self.radrepo = os.environ['RADREPO'] + 'raddata/'
            self.project = f.readline().split()[0].strip()
            self.makephase_downsampling = int(f.readline().split()[0].strip())
            self.latitude_clen = float(f.readline().split()[0].strip())
            self.latitude_smoothing = self.latitude_clen > 0

            try:
                self.output_style = int(f.readline().split()[0].strip())
                if self.output_style == 1 and self.rank == 0:
                    self.mre_out = self.directory[:-1] + '_out/'
                    os.makedirs(self.mre_out, exist_ok=True)
                self.use_numba_cache = int(f.readline().split()[0].strip())
                os.environ['USE_NUMBA_CACHE'] = 'True' if self.use_numba_cache else 'False'
            except:
                self.output_style = 0
                os.environ['USE_NUMBA_CACHE'] = 'True'
            
            from nemesispy_scatter.radtran.forward_model import ForwardModel
            self.FM = ForwardModel()
            
    def run_optimal_estimation(self):
        self.load_nemesis() 
        self.optimal_estimation_setup()
        self.optimal_estimation()
        self.optimal_estimation_wrapup()

    def run_nested_sampling(self,frac_remain = 0.001,maxdiff = 5, downsampling = 1):
        self.load_nemesis() 
        self.nested_sampling(frac_remain = frac_remain,maxdiff = maxdiff, downsampling = downsampling)
        
    def load_nemesis(self):
        if self.folder_flag:
            directories = [entry for entry in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, entry))]
        else:
            self.directory = self.directory[:-1]
            directories = ['']
        sorted_directories = sorted(directories)
        count = -1
        total_ngeom = 0
        angles = []
        spectra = []
        self.lats = []
        self.lons = []
        self.output_folders = []
        project = self.project

        for j in range(self.size + 1):
            if self.safe_load:
                self.comm.barrier()
                if j != self.rank:
                    continue
            else:
                if j > 0: 
                    break
                
            for subfolder in sorted_directories:
                subfolder = self.directory + subfolder + '/'
                try:
                    spx_data, FWHM, latitude, longitude,\
                    sol_angs, emiss_angs, aphis = parse_spx_file(subfolder + project + '.spx')
                    if self.rank == 0:
                        print(f'Successfully read spectrum from {subfolder}', flush=True)
                    self.lats.append(latitude)
                    self.lons.append(longitude)
                except Exception as e:
                    if self.rank == 0:
                        print(e, flush=True)
                        print(f'Could not read spectrum from folder {subfolder}', flush=True)
                    continue

                self.output_folders.append(subfolder)
                count += 1

                if count == 0:
                    ins_shape = 0
                    if FWHM != 0:
                        try:
                            ins_shape = parse_sha_file(subfolder + project + '.sha')
                        except Exception as e:
                            if self.rank == 0:
                                print(e, flush=True)
                                print('FWHM != 0 but could not read .sha file.', flush=True)
                                print('Continuing with square instrument shape', flush=True)

                    try:
                        ref_data, gas_id, iso_id, planet = parse_ref_file(subfolder + project + '.ref')
                    except Exception as e:
                        if self.rank == 0:
                            print(e, flush=True)
                            print('Error reading .ref file. Aborting.', flush=True)
                        self.comm.Abort(1)

                    npro = ref_data.shape[0]
                    H_model, P_model, T_model = ref_data[:, 0], ref_data[:, 1], ref_data[:, 2]
                    VMR_model = ref_data[:, 3:]

                    try:
                        A_model = parse_other_ref_file(subfolder + 'aerosol.ref', buffer=2)
                    except Exception as e:
                        A_model = np.array([[]])

                    try:
                        PARA_model = parse_other_ref_file(subfolder + 'parah2.ref', buffer=1)[:, 0]
                    except Exception as e:
                        PARA_model = np.zeros_like(P_model)

                    try:
                        vpdict = parse_vpf_file(subfolder + project + '.vpf')
                    except:
                        vpdict = {}

                    try:
                        ispace, iscat, ilbl, woff, ename, niter, philimit, nspec, ioff, lin\
                                                = parse_inp_file(subfolder + project + '.inp')
                    except Exception as e:
                        if self.rank == 0:
                            print(e, flush=True)
                            print('Could not read .inp file. Aborting.', flush=True)
                        self.comm.Abort(1)

                    try:
                        inormal, iray, ih20, ich4, io3, inh3, iptf, imie, iuv = parse_fla_file(subfolder + project + '.fla')
                    except Exception as e:
                        if self.rank == 0:
                            print(e, flush=True)
                            print('Could not read .fla file. Aborting.', flush=True)
                        self.comm.Abort(1)

                    try:
                        mu, wtmu, nf, nphi, insol, soldist, lowbc, galb,\
                        tsurf, H0, nlay, laytype, layint = parse_set_file(subfolder + project + '.set')
                    except Exception as e:
                        if self.rank == 0:
                            print(e, flush=True)
                            print('Could not read .set file. Aborting.', flush=True)
                        self.comm.Abort(1)

                    if laytype not in [1, 2, 3, 4]:
                        if self.rank == 0:
                            print(laytype, flush=True)
                            print('Layer type not implemented. Aborting.', flush=True)
                        self.comm.Abort(1)

                    custom_p_base = parse_lay_file(subfolder + 'pressure.lay') if laytype == 4 else None
                    angles = np.array(emiss_angs)
                    total_ngeom += len(angles)
                    spectrum = np.stack(spx_data).swapaxes(1, 2)
                    spectrum[:, 0] += woff
                    spectra.append(spectrum)
                    single_wave_grid = spectrum[0, 0]

                    try:
                        xa, xa_err, get_profile, contflags, varflags, haze_wave_grids,\
                        size_flags, n_real_flags, dists, dist_mults = parse_apr_file(subfolder + project + '.apr', npro, vpdict)
                    except:
                        self.comm.Abort(1)

                    if ilbl == 0:
                        try:
                            kta_file_path, _, _ = parse_table_file(subfolder + project + '.kls', subfolder)
                        except Exception as e:
                            if self.rank == 0:
                                print(e, flush=True)
                                print('ILBL = 0, but .kls file is not present. Trying to read .lls file.', flush=True)
                            try:
                                kta_file_path, _, _ = parse_table_file(subfolder + project + '.lls', subfolder)
                            except Exception as e:
                                if self.rank == 0:
                                    print(e, flush=True)
                                    print('Could not find .lls file either. Aborting', flush=True)
                                self.comm.Abort(1)

                    elif ilbl == 1:
                        print("ILBL = 1 is not implemented yet. Aborting.")
                        self.comm.Abort(1)

                    elif ilbl == 2:
                        try:
                            kta_file_path, _, _ = parse_table_file(subfolder + project + '.lls', subfolder)
                        except Exception as e:
                            if self.rank == 0:
                                print(e, flush=True)
                                print('ILBL = 2, but .lls file is not present. Trying to read .kls file.', flush=True)
                            try:
                                kta_file_path, _, _ = parse_table_file(subfolder + project + '.kls', subfolder)
                            except Exception as e:
                                if self.rank == 0:
                                    print(e, flush=True)
                                    print('Could not find .kls file either. Aborting', flush=True)
                                self.comm.Abort(1)

                    try:
                        cia_file_path, dnu, npara = parse_table_file(subfolder + project + '.cia', subfolder, cia=True)
                    except Exception as e:
                        if self.rank == 0:
                            print(e, flush=True)
                            print('Could not find .cia file', flush=True)

                    try:
                        sol_file_path, _, _ = parse_table_file(subfolder + project + '.sol', subfolder)
                    except Exception as e:
                        if self.rank == 0:
                            print(e, flush=True)
                            print('Could not find .sol file. Aborting.', flush=True)
                        self.comm.Abort(1)

                    self.FM.set_planet_model(planet=planet, gas_id_list=gas_id, 
                                             iso_id_list=iso_id, NLAYER=nlay, radrepo=self.radrepo, latitude=latitude)
                    self.FM.set_angles(mu=mu, wtmu=wtmu)
                    self.FM.set_opacity_data(kta_file_paths=kta_file_path, 
                                             cia_file_path=self.radrepo + cia_file_path[0], 
                                             wave_grid=single_wave_grid, dnu=dnu, npara=npara)
                    self.FM.makephase_downscaling = self.makephase_downsampling
                    self.FM.FWHM = FWHM
                    self.FM.ins_shape = ins_shape
                    self.FM.layer_type = laytype
                    self.FM.iray = iray
                    self.FM.imie = imie
                    self.FM.ispace = ispace
                    self.FM.inormal = inormal
                    self.FM.custom_p_base = custom_p_base

                    try:
                        self.sol = parse_solspec_file(self.radrepo + sol_file_path[0], self.FM.wave_grid, soldist, ispace)
                    except Exception as e:
                        try: 
                            self.sol = parse_solspec_file(sol_file_path[0], self.FM.wave_grid, soldist, ispace)
                        except:
                            if self.rank == 0:
                                print(e, flush=True)
                                print(f'Could not read solar file: {self.radrepo + sol_file_path[0]}. Aborting.', flush=True)
                            self.comm.Abort(1)

                elif count > 0:
                    total_ngeom += len(angles)
                    spectrum = np.stack(spx_data).swapaxes(1, 2)
                    spectrum[:, 0] += woff
                    spectra.append(spectrum)

        self.niter = niter
        self.ngeom = len(angles)
        self.nspec = int(total_ngeom / self.ngeom)
        self.nvar = len(xa)
        
        self.angles = angles
        self.sol_angles = sol_angs
        self.aphis = aphis
        
        xa = np.concatenate([xa for _ in range(self.nspec)])
        self.xa = np.log(xa)
        xa_err = np.concatenate([xa_err for _ in range(self.nspec)])
        self.xa_err = xa_err / np.exp(self.xa)
        
        self.contflags = contflags
        self.varflags = varflags
        self.size_flags = size_flags
        self.n_real_flags = n_real_flags
        
        self.haze_wave_grids = haze_wave_grids
        
        self.H0 = H0
        
        self.philimit = philimit
        
        self.single_wave_grid = single_wave_grid
        self.spectra = np.concatenate(spectra, axis=0)
        self.wave_grid = np.concatenate([self.spectra[i, 0] for i in range(self.spectra.shape[0])])
        self.y = np.concatenate([self.spectra[i, 1] for i in range(self.spectra.shape[0])])
        self.y_err = np.concatenate([self.spectra[i, 2] for i in range(self.spectra.shape[0])])
        self.lenspec = int(len(self.y) / self.nspec)
        
        self.get_profile = get_profile
        
        self.H_model = H_model
        self.P_model = P_model
        self.T_model = T_model
        self.VMR_model = VMR_model
        self.A_model = A_model
        self.PARA_model = PARA_model
        self.gas_id = gas_id
        self.iso_id = iso_id
        
        self.first_run = True
        
        self.dists = dists
        self.dist_mults = dist_mults
            
    def optimal_estimation_setup(self):
        """
        Creates matrices needed for an optimal estimation run. Most are scipy.sparse objects 
        for speed when doing large spatially smoothed runs.
        """
        
        
        self.xn = self.xa.copy()
        self.xn_err = self.xa_err.copy()

        ix = np.arange(len(self.xa))                             
        self.ix_gradreq = ix[np.where(self.xa_err>1e-6)]
        self.dx = 0.05*self.xa
        self.dx[self.dx == 0] = 0.1
        
        if self.rank == 0: # only the master node needs covariance matrices
            self.sei = 1/self.y_err[:,None]**2
            sa = (np.eye(len(self.xa))*self.xa_err[:,None])**2

            nvar = self.nvar
            nspec = self.nspec

            if len(self.contflags)>0:
                for flag in self.contflags:
                    startidx, endidx, typeflag, clen, hwidx = flag
                    for ispec in range(self.nspec):
                        for i in range(startidx,endidx):
                            for j in range(startidx,endidx):
                                if i==j:
                                    continue
                                if typeflag==0:
                                    sa[i,j] = np.sqrt(sa[i,i]*sa[j,j])*np.exp(-np.abs(np.log(self.P_model[i-startidx]\
                                                                                            /self.P_model[j-startidx]))/clen)
                                elif typeflag==1:
                                    haze_wave_grid = self.haze_wave_grids[hwidx]
                                    sa[i,j] = np.sqrt(sa[i,i]*sa[j,j])\
                                                *np.exp(-np.abs(haze_wave_grid[i-startidx]-haze_wave_grid[j-startidx])/clen)
                        startidx += nvar
                        endidx += nvar

            if self.latitude_smoothing:
                for n in range(0, nspec * nvar, nvar):
                    for m in range(0, nspec * nvar, nvar):
                        if n != m:  
                            distance = np.sqrt((self.lats[n//nvar]-self.lats[m//nvar])**2\
                                             + (self.lons[n//nvar]-self.lons[m//nvar])**2)
                            sa[n:n+nvar,m:m+nvar] = np.sqrt(sa[n:n+nvar,n:n+nvar]*sa[m:m+nvar,m:m+nvar])\
                                                    *np.exp(-np.abs(distance)/self.latitude_clen)


            SAMINFAC = 0.001
            diag_prod = np.diag(sa)[:, None] * np.diag(sa)  
            scaled_diag_prod = diag_prod * SAMINFAC
            sa[sa < scaled_diag_prod] = 0

            self.sa = sa
            nonzeros = sa.nonzero()
            self.sac = csc_array((sa[nonzeros], nonzeros), sa.shape)
            self.lu = splu(self.sac)

            eye = np.eye(sa.shape[0])
            nonzeros = eye.nonzero()
            self.eye = csc_array((eye[nonzeros],nonzeros),eye.shape)
        self.yn = self.get_y(self.xa)
        
                          
    def optimal_estimation(self):
        """
        Runs optimal estimation. Very similar in implementation to the original FORTRAN.
        """
        
                          
        if self.niter > 0:
            kks = self.get_kks(self.yn,self.xn,self.dx,self.ix_gradreq, self.angles)
            if self.rank==0:
                self.kks = csc_array(kks)
                self.kks.eliminate_zeros()
                kks1 = self.kks.copy()
                kks1.eliminate_zeros()
        xn1 = self.xn.copy()
        yn1 = self.yn.copy()
        
        stopflag = False
        contflag = False
        alambda = 1.0

        for i in range(self.niter):
            if self.niter < 0:
                break
            if self.rank == 0:
                if not contflag:

                    mat_product = self.sac @ (kks1 * self.sei).transpose()
                    A = mat_product @ kks1 + self.eye
                    B = mat_product @ csc_array((self.y-self.yn)[:, None]) \
                      - mat_product @ kks1 @ csc_array((self.xa-self.xn)[:, None])

                    xplus, info = bicgstab(A, B.toarray(), x0 =(self.xn-self.xa)[:,None])

                    if info != 0:
                        print(f'''X+ CALCULATION FAILURE, ERROR CODE: {info}. 
                                  RETRYING WITH LOWER TOLERANCE. RESULTS MAY BE INACCURATE''',flush=True)
                        xplus, info = bicgstab(A, B.toarray(), x0 =(self.xn-self.xa)[:,None], maxiter = info*5, tol = 1e-3)
                        if info != 0:
                            print(f'''REPEATED X+ CALCULATION FAILURE. EXITING''',flush=True)
                            break

                    y_cost_old = ((self.y-self.yn)*self.sei[:,0]*(self.y-self.yn)).sum()

                    z = self.lu.solve(self.xn - self.xa)
                    x_cost_old = (self.xn - self.xa).transpose() @ z

                    costf_old = y_cost_old + x_cost_old

                    xn_out = self.xa + xplus

            contflag = False
            
            if self.rank == 0:
                xn1 = self.xn + (xn_out-self.xn)/(1+alambda)
            xn1 = self.comm.bcast(xn1,root=0)    
            yn1 = self.get_y(xn1)
            
            if self.rank == 0 :
                y_cost_new = ((self.y-yn1)*self.sei[:,0]*(self.y-yn1)).sum()

                z = self.lu.solve(xn1 - self.xa)
                x_cost_new = (xn1 - self.xa).transpose() @ z

                costf_new = y_cost_new + x_cost_new

                print("Iter, New cost, Old cost, alambda:  ", i, costf_new, costf_old, alambda, flush = True)
                print("Change in cost, Convergence limit:  ", 1 - costf_new/costf_old, self.philimit, flush = True)
                print("Cost from spectra, Cost from apriori:  ", y_cost_new,x_cost_new)
                print('Chisq old: ', self.rchs(self.y,self.yn,self.y_err),flush=True)
                print('Chisq new: ', self.rchs(self.y,yn1,self.y_err),flush=True)

                if costf_new < costf_old:

                    self.xn = xn1.copy()
                    self.yn = yn1.copy()
                    self.kks = kks1.copy()

                    if 1 - costf_new/costf_old < self.philimit and alambda < 1:
                        stopflag = True

                    else:
                        alambda*=0.3

                elif i > 4 and 1 - np.abs(costf_new/costf_old) < self.philimit:
                    if alambda < 0.1:
                        alambda*=10
                        contflag = True

                    else:
                        stopflag = True
                else:
                    alambda*=10
                    contflag = True
                    
            contflag = self.comm.bcast(contflag,root=0)
            if contflag: 
                continue
            
            stopflag = self.comm.bcast(stopflag,root=0)
            if stopflag:
                break

            self.xn = self.comm.bcast(self.xn,root=0)
            self.yn = self.comm.bcast(self.yn,root=0)

            self.dx = self.xn*0.05
            self.dx[self.dx == 0] = 0.1

            if i == self.niter - 1:
                break
            
            kks1 = self.get_kks(self.yn,self.xn,self.dx,self.ix_gradreq, self.angles)

            if self.rank==0:
                kks1 = csc_array(kks1)
                kks1.eliminate_zeros()

            self.comm.barrier()

    def optimal_estimation_wrapup(self):
                      
        """
        Calculates errors (which can take a long time for smoothed runs!). Then outputs all the .mre files.
        
        """
            
        lenspec = self.lenspec  
        nvar = self.nvar
        if self.rank == 0:
            print('Iteration finished. Calculating errors.',flush=True)
            if self.niter > 0:
                A = self.sac @ (self.kks * self.sei).transpose() @ self.kks + self.eye 
                A.eliminate_zeros()
                lu = splu(A)
                for i in range(len(self.xn_err)):
                    if i in self.ix_gradreq:
                        err = np.sqrt(lu.solve(self.sa[i])[i])
                        self.xn_err[i] = err
                    else:
                        self.xn_err[i] = self.xa_err[i]

            print('Writing to .mre files',flush=True)
            for ispec in range(self.nspec):
                
                output_file = self.output_folders[ispec] +self.project+'.mre'
                if self.output_style == 1:
                    output_file = self.mre_out +self.project+str(int(self.lats[ispec]))+'.mre'
                
                with open(output_file, 'w') as file:

                    # Writing the header information
                    file.write("        1  ! Total number of retrievals\n")
                    file.write(f"    1  {self.ngeom}  {len(self.y)}  {len(self.xa)}  {len(self.y)}   ! ispec,ngeom,ny,nx,ny\n")
                    file.write(f"  {float(self.lats[ispec])}  {float(self.lons[ispec])} Latitude, Longitude\n")
                    file.write(" Radiances expressed as uW cm-2 sr-1 um-1\n")
                    file.write("   i  lambda  R_meas     error   %err  R_fit     Diff%\n")

                    # Iterating through the data arrays and writing each line
                    for i, (lambda_val, r_meas, err, r_fit) in \
                                        enumerate(zip(self.wave_grid[ispec*lenspec:(ispec+1)*lenspec], \
                                                      self.y[ispec*lenspec:(ispec+1)*lenspec]*1e6,     \
                                                      self.y_err[ispec*lenspec:(ispec+1)*lenspec]*1e6, \
                                                      self.yn[ispec*lenspec:(ispec+1)*lenspec]*1e6), start=1):

                        percent_err = (err / r_meas) * 100
                        diff_percent = np.abs((r_meas - r_fit) / r_meas) * 100
                        line = f"{i:5d} {lambda_val:10.8f} {r_meas:.8E} {err:.8E} {percent_err:7.2f} {r_fit:.8E} {diff_percent:7.2f}\n"
                          
                        file.write(line)

                    file.write(f"\n")
                    file.write(f" nvar =            {self.varflags.shape[0]*self.nspec}\n")
                    for group in self.varflags:
                        start_idx, end_idx, varnum, varid, isoid, paramid = group 
                        # Writing the header for each variable group
                        file.write(f" Variable            {varnum}\n")
                        file.write(f"{varid} {isoid} {paramid} \n") 
                        file.write(" 0 0 0 0 0 !placeholder\n")  
                        file.write("    i,   ix,   xa          sa_err       xn          xn_err\n")

                        # Writing the data for each variable group
                        for i, ix in enumerate(range(start_idx, end_idx), start=1):
                            xa_val = np.exp(self.xa[ix+ispec*nvar])
                            xa_err_val = xa_val*self.xa_err[ix+ispec*nvar]
                            xn_val = np.exp(self.xn[ix+ispec*nvar])
                            xn_err_val = xn_val*self.xn_err[ix+ispec*nvar]
                            line = f"{i:5d} {ix+1:5d} {xa_val:10.5E} {xa_err_val:10.5E} {xn_val:10.5E} {xn_err_val:10.5E}\n"
                            file.write(line)
        if self.rank == 0:
            print('Done!',flush=True)
        self.comm.barrier()
                          
                          
    def rchs(self, a,b,err):
        return np.sum(((a - b)**2)/(len(a.flatten())*(err**2)))

    def LogLikelihood(self, cube, _=0,__=0):
        xn = self.xa.copy()
        for i, var in enumerate(self.vars_to_vary):
            xn[var] = cube[i]

        start_index = np.random.randint(self.downsampling)
        indices = np.arange(len(self.single_wave_grid))#[start_index::downsampling]

        yn = np.array([])
        for j in range(self.ngeom):
            yn = np.concatenate([yn, self.get_spec(xn,self.angles[j], 
                                                       self.sol_angles[j], 
                                                       self.aphis[j],False if j else True, indices)])
#         y_indices = np.concatenate([indices,indices+len(self.y)//2])
        chisq = self.rchs(yn,self.y,self.y_err)

        like = -np.log(chisq)
        return like
    
    def Prior(self, cube):
        cube1 = cube.copy()
        
        for i in range(len(self.vars_to_vary)):
              cube1[i] = self.priors[i](cube1[i])

        return cube1
    
    def nested_sampling(self,frac_remain=0.001,maxdiff = 5, downsampling = 1):
        self.downsampling = downsampling
        self.vars_to_vary = [i for i in range(len(self.xa)) if self.xa_err[i]>1e-7]
        
        self.priors = []
        for i in self.vars_to_vary:
            dist_code = self.dists[i]
            if dist_code == 0:
                self.priors.append(norm(self.xa[i], self.xa_err[i] * self.dist_mults[i]).ppf)
            elif dist_code == 1:
                self.priors.append(lambda x, i=i: x * (self.xa[i] + self.dist_mults[i]*self.xa_err[i] - \
                                                  self.xa[i] + self.dist_mults[i]*self.xa_err[i]) + \
                                                  self.xa[i] - self.dist_mults[i]*self.xa_err[i])
            else:
                print('DISTRIBUTION ID NOT DEFINED!', flush = True)

        sampler = ultranest.ReactiveNestedSampler([str(i) for i in self.vars_to_vary], self.LogLikelihood, self.Prior,
                  log_dir=self.directory+'/ultranest_output', resume='resume')
        self.result = sampler.run(frac_remain=frac_remain,max_num_improvement_loops=-1)
        sampler.print_results()
        sampler.plot()
#         import corner
#         import matplotlib.pyplot as plt
#         if sampler.log:
#             sampler.logger.debug('Making corner plot ...')
#         results = sampler.results
#         paramnames = results['paramnames']
#         data = results['weighted_samples']['points']
#         weights = results['weighted_samples']['weights']

#         corner.corner(
#             results['weighted_samples']['points'],
#             weights=results['weighted_samples']['weights'],
#             labels=results['paramnames'],smooth = 1.0)
            
#         if sampler.log_to_disk:
#             plt.savefig(os.path.join(sampler.logs['plots'], 'corner.pdf'), bbox_inches='tight')
#             plt.close()
#             sampler.logger.debug('Making corner plot ... done')
        sampler.plot_trace()
        
        
    def get_kks(self, y, xn, dxs, ixs, angles, tasks_per_worker=1):
        local_kks = np.zeros((len(y), len(xn)))
        kks = None
        lenspec = int(len(y)/self.nspec)
        ixs = list(ixs)
        
        if self.size == 1:
            # Handle everything in the main process (no parallelisation)
            kks = np.zeros((len(y), len(xn)))
            for i in tqdm(range(len(ixs))):
                ispec = ixs[i] // self.nvar
                kks_section = self.get_kk(y[ispec * lenspec:(ispec + 1) * lenspec],
                                          xn[ispec * self.nvar:(ispec + 1) * self.nvar],
                                          dxs[ixs[i]], ixs[i] % self.nvar)
                kks[ispec * lenspec:(ispec + 1) * lenspec, ixs[i]] += kks_section
        
        else:
            if self.rank == 0:
                total_tasks = len(ixs)
                kks = np.zeros((len(y), len(xn)))
                active_requests = []
                active_receives = {}  # Map: worker_rank -> (receive_request, task_info)

                # Initially set up non-blocking receives for expected data
                for worker in range(1, self.size):
                    tasks_for_worker = [ixs.pop(0) for _ in range(tasks_per_worker) if ixs]
                    if tasks_for_worker:
                        send_req = self.comm.isend(tasks_for_worker, dest=worker, tag=0)
                        active_requests.append((send_req, worker, tasks_for_worker))
                        # Prepare for receiving data back from this worker
                        recv_req = self.comm.irecv(source=worker, tag=2)
                        active_receives[worker] = recv_req  # Use worker rank as key

                with tqdm(total=total_tasks) as pbar:
                    while active_requests or active_receives or ixs:
                        # Check send requests
                        for req, worker, tasks in active_requests[:]:
                            if req.test():
                                active_requests.remove((req, worker, tasks))

                        # Check receive requests
                        for worker, recv_req in list(active_receives.items()):
                            data_received = recv_req.test()
                            if data_received[0]:  # If the receive operation has completed
                                for data, task, ispec in data_received[1]:
                                    kks[ispec*lenspec:(ispec+1)*lenspec, task] += data
                                del active_receives[worker]  # Remove completed receive request
                                pbar.update(len(data_received[1]))
                                # Optionally, send next tasks to this worker
                                tasks_for_worker = [ixs.pop(0) for _ in range(tasks_per_worker) if ixs]
                                if tasks_for_worker:
                                    next_send_req = self.comm.isend(tasks_for_worker, dest=worker, tag=0)
                                    active_requests.append((next_send_req, worker, tasks_for_worker))
                                    # Setup receive for the next tasks' results
                                    next_recv_req = self.comm.irecv(source=worker, tag=2)
                                    active_receives[worker] = next_recv_req

                # Terminate workers
                for worker in range(1, self.size):
                    self.comm.send(None, dest=worker, tag=1)
                print('Jacobian calculation complete', flush=True)

            else:
                while True:
                    tasks = self.comm.recv(source=0, tag=MPI.ANY_TAG)
                    if tasks is None:
                        break
                    message = []
                    for task in tasks:
                        ispec = task // self.nvar
                        local_kks_section = self.get_kk(y[ispec*lenspec:(ispec+1)*lenspec],
                                                   xn[ispec*self.nvar:(ispec+1)*self.nvar],
                                                   dxs[task], task % self.nvar)
                        message.append((local_kks_section, task, ispec))
                    self.comm.send(message, dest=0, tag=2)
                    self.comm.isend(None, dest=0, tag=0)
        return kks

    def get_kk(self,y,xn_init,dx,ix):
        xn = xn_init.copy()
        xn[ix] = xn_init[ix] + dx
        spec = np.array([])
        for j in range(self.ngeom):
            spec = np.concatenate([spec, self.get_spec(xn,self.angles[j], 
                                                       self.sol_angles[j], 
                                                       self.aphis[j],False if j else True)])
        kk = (spec - y)/(dx)
        return kk

    def get_spec(self,xn,emiss_angle, sol_angle, aphi, remake_phase=True, indices = None):
        if indices is not None:
            self.FM.wave_grid = self.single_wave_grid[indices]
            sol = self.sol[indices]
        else:
            sol = self.sol
            
        xn = np.exp(xn)
        T_model_new, VMR_model_new, A_model_new,\
        A_info, PARA_model_new, size_model, n_real_model = self.get_profile(self.H_model.copy(), self.P_model.copy(),
                                                                            self.T_model.copy(), \
                                                                            self.VMR_model.copy(), self.A_model.copy(), 
                                                                            self.PARA_model.copy(),\
                                                                            self.gas_id.copy(), self.iso_id.copy(),
                                                                            self.FM.plt_params.copy(), xn)
        
        spec = self.FM.calc_point_spectrum(self.H_model.copy(), self.P_model,T_model_new,VMR_model_new,
                                 A_model_new,A_info,PARA_model_new,size_model,self.size_flags,n_real_model,self.n_real_flags,
                                 self.H0,emiss_angle,sol_angle,aphi,solspec = sol,remake_phase = remake_phase)

        return spec

    def get_y(self, x):
        if self.rank == 0:
            print('Calculating spectrum', flush=True)

        ys = np.array([])

        specs_per_process = self.nspec // self.size + (1 if self.rank < self.nspec % self.size else 0)

        start_spec = sum(self.nspec // self.size + (1 if r < self.nspec % self.size else 0) for r in range(self.rank))
        end_spec = start_spec + specs_per_process

        for spec_index in range(start_spec, end_spec):
            xi = x[spec_index * self.nvar:(spec_index + 1) * self.nvar]
            for j in range(self.ngeom):
                remake_phase = j == 0
                ys = np.concatenate([ys, self.get_spec(xi,self.angles[j], 
                                                       self.sol_angles[j], 
                                                       self.aphis[j],remake_phase)])
            self.first_run = False

        if self.first_run:
            xi = x[0:self.nvar]
            _ = self.get_spec(xi, self.angles[0], self.sol_angles[0], self.aphis[0], remake_phase=True)
            self.first_run = False

            
        if self.rank == 0:
            print('Spectrum calculated', flush=True)

        gathered_ys = self.comm.gather(ys, root=0)
        if self.rank == 0:
            ys = np.concatenate(gathered_ys)

        ys = self.comm.bcast(ys, root=0)
        self.comm.barrier()
        return ys
