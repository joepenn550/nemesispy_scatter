import numpy as np
from functools import partial
from nemesispy_scatter.common.constants import K_B, N_A, ATM
from nemesispy_scatter.radtran.calc_mmw import calc_mmw
from nemesispy_scatter.common.calc_hydrostat import calc_grav
from nemesispy_scatter.common.calc_hydrostat import calc_hydrostat


def parameteriser(pname, npro, H_model, P_model, T_model, VMR_model, A_model, PARA_model,\
                               gas_id, iso_id, plt_params, cont_profile, x):
    
    """
    Defines parameterisations. Most inputs are self-explanatory.
    
    pname: Name of parameterisation
    cont_profile: Reference profile 
    x: Relevant section of state vector
    
    Returns a profile of length npro, which depends on the parameterisation chosen.
    
    
    """
    
    # Calculating molwt, gravity, scale height
    
    molwt = np.zeros(len(P_model))
    for i in range(len(molwt)):
        molwt[i] = calc_mmw(gas_id, VMR_model[i], iso_id)
        
    H_model = calc_hydrostat(P_model, T_model, molwt, plt_params[0], plt_params[4], 
                         plt_params, H=H_model)
    
    molwt = molwt * 1000
    gravity = calc_grav(plt_params,H_model)
    scale = K_B*T_model/(molwt*gravity)
    
    # Also adjusting units of H and P
    
    H_model = H_model/1000
    P_model = P_model/ATM
    
    
    # Parameterisations:
    
    if pname == 'continuous':
        return x
    
    elif pname == 'simple_scaling':
        return x*cont_profile
    
    elif pname == 'simple_knee': # untested
        
        pknee = x[0]
        xdeep = x[1]
        xfsh = x[2]

        xfac = (1-xfsh)/xfsh

        hknee = np.interp(pknee, P_model[::-1], H_model[::-1])
        jfsh = 0

        x1 = np.ones(npro)*xdeep 
        for j in range(npro):
            if P_model[j] < pknee:
                if jfsh == 0:
                    delh = H_model[j] - hknee
                else:
                    delh = H_model[j] - H_model[j-1]
                x1[j] = x1[j-1]*np.exp(-delh*xfac/scale[j])
                jfsh = 1

        x1 = np.clip(x1,1e-36,np.inf)
        return x1

    elif pname == 'double_knee': # untested
    
        pknee = x[0]
        xdeep = x[1]
        xfsh = x[2]


        xfac = (1-xfsh)/xfsh

        hknee = np.interp(pknee, P_model[::-1], H_model[::-1])
        jfsh = 0

        delh = 1.0

        for j in range(npro):
            pmin = np.abs(P_model[j]-pknee)
            if pmin < delh:
                jfsh = j
                delh = pmin

        x1 = np.zeros(npro)
        x1[jfsh] = xdeep

        for j in range(jfsh+1,npro):
            delh = H_model[j] - H_model[j-1]
            x1[j] = x1[j-1]*np.exp(-delh*xfac/scale[j])

        for j in range(j-1,0,-1):
            delh = H_model[j] - H_model[j+1]
            x1[j] = x1[j+1]*np.exp(-delh*xfac/scale[j])

        x1 = np.clip(x1,1e-36,1e10)

        return x1
    
    elif pname == 'var_press_exp_knee':  # ~5% off sometimes - why?
    
        pknee = x[0]
        xdeep = x[1]
        xfsh = x[2]

        hknee = np.interp(pknee, P_model[::-1], H_model[::-1])
        # Initialize ND, Q, OD
        nd = np.zeros(npro)
        od = np.zeros(npro)
        q = np.zeros(npro)

        jknee = -1
        xf = 1.0 

        # Find levels in atmosphere that span pknee
        for j in range(npro - 1):
            if P_model[j] >= pknee and P_model[j + 1] < pknee:
                jknee = j
                break
        if jknee < 0:
            print("subprofretg: Error in model 32.",flush=True)
            print("XDEEP,XFSH,PKNEE",flush=True)
            print(xdeep, xfsh, pknee,flush=True)

        delh = H_model[jknee + 1] - hknee
        xfac = 0.5 * (scale[jknee] + scale[jknee + 1]) * xfsh
        nd[jknee + 1] = np.exp(-delh / xfac)

        delh = hknee - H_model[jknee]
        
        xfac = xf
        nd[jknee] = np.exp(-delh / xfac)

        for j in range(jknee + 2, npro):
            delh = H_model[j] - H_model[j - 1]
            xfac = scale[j] * xfsh
            nd[j] = nd[j - 1] * np.exp(-np.abs(delh) / xfac)


        for j in range(jknee):
            delh = H_model[jknee] - H_model[j]
            xfac = xf
            nd[j] = np.exp(-np.abs(delh) / xfac)


        xmolwt = molwt
        for j in range(npro):
            # Calculate density of atmosphere (g/cm3)
            rho = 0.1013*molwt[j]/(K_B)*(P_model[j]/T_model[j])
            # Calculate initial particles/gram
            q[j] = nd[j] / rho

        # Now integrate optical thickness
        od[npro - 1] = nd[npro - 1] * scale[npro - 1] * xfsh * 1e5

        for j in range(npro - 2, -1, -1):
            if j > jknee:
                delh = H_model[j + 1] - H_model[j]
                xfac = scale[j] * xfsh
                od[j] = od[j + 1] + (nd[j] - nd[j + 1]) * xfac * 1e5
            else:
                if j == jknee:
                    delh = H_model[j + 1] - hknee
                    xfac = 0.5 * (scale[j] + scale[j + 1]) * xfsh
                    od[j] = od[j + 1] + (1. - nd[j + 1]) * xfac * 1e5
                    delh = hknee - H_model[j]
                    xfac = xf
                    od[j] = od[j] + (1. - nd[j]) * xfac * 1e5
                else:
                    delh = H_model[j + 1] - H_model[j]
                    xfac = xf
                    od[j] = od[j + 1] + (nd[j + 1] - nd[j]) * xfac * 1e5
        odx = od[0]
        # Now normalise specific density profile
        for j in range(npro):
            q[j] = q[j] * xdeep / odx
            if q[j] > 1e10:
                q[j] = 1e10
            elif q[j] < 1e-36:
                q[j] = 1e-36
            ntest = np.isnan(q[j])
            if ntest:
                q[j] = 1e-36
        return q

    elif pname == 'gaussian':
        xdeep = x[0]
        pknee = x[1]
        xwid = x[2]

        y0 = np.log(pknee)
        x1 = np.zeros(npro)

        for j in range(npro):
            y = np.log(P_model[j])
            x1[j] = xdeep*np.exp(-((y-y0)/xwid)**2)

        return x1

    elif pname == 'lorentzian':

        xdeep = x[0]
        pknee = x[1]
        xwid = x[2]

        y0 = np.log(pknee)
        x1 = np.zeros(npro)

        for j in range(npro):
            y = np.log(P_model[j])
            xx = (y-y0)**2 + xwid**2

            x1[j] = xdeep*(xwid**2)/xx

        return x1
    
    elif pname == 'modifych4irwin':

        ch4tropvmr, ch4stratvmr, RH = x

        SCH40 = 10.6815
        SCH41 = -1163.83
        # psvp is in bar

        xnew = np.zeros(npro)
        xnewgrad = np.zeros(npro)
        pch4 = np.zeros(npro)
        pbar = np.zeros(npro)
        psvp = np.zeros(npro)

        for i in range(npro):
            pbar[i] = P_model[i] * 1.013
            tmp = SCH40 + SCH41 / T_model[i]
            psvp[i] = 1e-30 if tmp < -69.0 else np.exp(tmp)

            pch4[i] = ch4tropvmr * pbar[i]
            if pch4[i] / psvp[i] > 1.0:
                pch4[i] = psvp[i] * RH

            if pbar[i] < 0.1 and pch4[i] / pbar[i] > ch4stratvmr:
                pch4[i] = pbar[i] * ch4stratvmr

            if pbar[i] > 0.5 and pch4[i] / pbar[i] > ch4tropvmr:
                pch4[i] = pbar[i] * ch4tropvmr
                xnewgrad[i] = 1.0

            xnew[i] = pch4[i] / pbar[i]

        return xnew

    elif pname == 'integrated_gaussian':
    
        xdeep,pknee,xwid = x

        y0 = np.log(pknee)

        rho = 0.1013*molwt/(K_B)*(P_model/T_model)

        y = np.log(P_model)

        q = 1/(xwid*np.sqrt(np.pi))*np.exp(-((y-y0)/xwid)**2)
        nd = q*rho
        od = nd*scale*1e5

        xod = np.sum(od) * 0.25

        x1 = q*xdeep/xod

        return x1
    

def get_parameterisation(pid, npro):
    """
    Takes a parameterisation id, and returns a tuple of the form 
    (parameterisation, corresponding length of state vector, normalisation index)
    If normalisation index >-1, the sum of the returned profile is later normalised to 
    x[normalisation_index]. Used for parameterisations which take in an integrated opacity.
    """
    pid_registry = {0:  ('continuous', npro, -1),
                    1:  ('simple_knee', 3, -1),
                    2:  ('simple_scaling', 1, -1),
                    3:  ('simple_scaling', 1, -1),
                    4:  ('simple_knee', 3, -1),
                    7:  ('double_knee', 3, -1),
                    12: ('gaussian', 3, -1),
                    13: ('lorentzian', 3, -1),
                    32: ('var_press_exp_knee', 3, 1),
                    45: ('modifych4irwin', 3, -1),
                    47: ('integrated_gaussian',3, 0),}
    try:
        pname,xlen,normidx = pid_registry[pid]
        return partial(parameteriser,pname),xlen,normidx    
                    
    except:
        print(f'PARAMETERISATION {pid} NOT DEFINED!', flush=True)
        return
