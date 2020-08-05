import numpy as np
import emcee
from scipy.interpolate import griddata
import scipy.optimize as op
import matplotlib.pylab as pl
import sys
import os
# SET PATH TO SPECTRAL SIMULATOR
#ss_path = "../simulate_lte_ra/"
#sys.path.append(ss_path)
from classes import *
from constants import *
from numba import njit
from tqdm import tqdm

from multiprocessing import Pool

h = 6.626*10**(-34) #Planck's constant in J*s
k = 1.381*10**(-23) #Boltzmann's constant in J/K
kcm = 0.69503476 #Boltzmann's constant in cm-1/K
ckm = 2.998*10**5 #speed of light in km/s
ccm = 2.998*10**10 #speed of light in cm/s
cm = 2.998*10**8 #speed of light in m/s

# calculates the rms noise in a given spectrum, should be robust to interloping lines, etc.

@njit
def calc_noise_std(y,sigma=3):
    tmp_y = np.copy(y)
    i = np.nanmax(tmp_y)
    rms = np.sqrt(np.nanmean(np.square(tmp_y)))
    while i > sigma*rms:
        tmp_y = tmp_y[tmp_y<sigma*rms]
        noise_mean = np.nanmean(tmp_y)
        rms = np.sqrt(np.nanmean(np.square(tmp_y)))
        # noise std as the rms with mean subtracted away
        noise_std = rms - noise_mean
        i = np.nanmax(tmp_y)
    return noise_mean, noise_std

# reads in the data and returns the data which has coverage of a given species (from the simulated intensities)
def read_file(filename, restfreqs, int_sim, oldformat=False, shift=0.0, GHz=False, plot=False, block_interlopers=True):
    data = np.load(filename, allow_pickle=True)

    if oldformat:
        freqs = data[:,1]
        intensity = data[:,2]
    else:
        freqs = data[0]
        intensity = data[1]

    if GHz:
        freqs *= 1000.

    relevant_freqs = np.zeros(freqs.shape)
    relevant_intensity = np.zeros(intensity.shape)
    relevant_yerrs = np.zeros(freqs.shape)

    covered_trans = []

    for i, rf in enumerate(restfreqs):
        thresh = 0.05
        if int_sim[i] > thresh*np.max(int_sim):
            vel = (rf - freqs)/rf*300000 + shift
            locs = np.where((vel<(.3+6.)) & (vel>(-.3+5.6)))

            if locs[0].size != 0:
                noise_mean, noise_std = calc_noise_std(intensity[locs])
                if block_interlopers and (np.max(intensity[locs]) > 6*noise_std):
                    print("interloping line detected " + str(rf))
                    if plot:
                        pl.plot(freqs[locs], intensity[locs])
                        pl.show()
                else:
                    covered_trans.append(i)
                    print("Found_line at: " + str(rf))
                    relevant_freqs[locs] = freqs[locs]
                    relevant_intensity[locs] = intensity[locs]
                    relevant_yerrs[locs] = np.sqrt(noise_std**2 + (intensity[locs]*0.1)**2)

                if plot:
                    pl.plot(freqs[locs], intensity[locs])
                    pl.show()
            else:
                print("No data covering " + str(rf))

    mask = relevant_freqs > 0

    relevant_freqs = relevant_freqs[mask]
    relevant_intensity = relevant_intensity[mask]
    relevant_yerrs = relevant_yerrs[mask]
    print(relevant_yerrs)

    return(relevant_freqs, relevant_intensity, relevant_yerrs, covered_trans)




def predict_intensities(source_size, Ncol, Tex, dV, mol_cat):
    obs_params = ObsParams("test", source_size=source_size)
    sim = MolSim("mol sim", mol_cat, obs_params, [0.0], [Ncol], [dV], [Tex], ll=[7000], ul=[30000], gauss=False)
    freq_sim = sim.freq_sim
    int_sim = sim.int_sim
    tau_sim = sim.tau_sim

    return freq_sim, int_sim, tau_sim



# Apply a beam dilution correction factor
@njit(fastmath=True)
def apply_beam(frequency, intensity, source_size, dish_size):
    #create a wave to hold wavelengths, fill it to start w/ frequencies
    wavelength = cm/(frequency*1e6)
    
    #fill it with beamsizes
    beam_size = wavelength * 206265 * 1.22 / dish_size
    
    #create an array to hold beam dilution factors
    dilution_factor = source_size**2/(beam_size**2 + source_size**2)
    
    intensity_diluted = intensity*dilution_factor
    
    return intensity_diluted



@njit(fastmath=True)
def make_model(freqs1, freqs2, freqs3, freqs4, ints1, ints2, ints3, ints4, ss1, ss2, ss3, ss4, datagrid0, datagrid1, vlsr1, vlsr2, vlsr3, vlsr4, dV, Tex):
    curr_model = np.zeros(datagrid1.shape)
    model1 = np.zeros(datagrid1.shape)
    model2 = np.zeros(datagrid1.shape)
    model3 = np.zeros(datagrid1.shape)
    model4 = np.zeros(datagrid1.shape)
    nlines = freqs1.shape[0]

    for i in range(nlines):
        vel_grid = (freqs1[i]-datagrid0)/freqs1[i]*ckm
        mask = np.abs(vel_grid-5.8) < dV*10
        model1[mask] += ints1[i]*np.exp(-0.5*((vel_grid[mask] - vlsr1)/(dV/2.355))**2.)
        model2[mask] += ints2[i]*np.exp(-0.5*((vel_grid[mask] - vlsr2)/(dV/2.355))**2.)
        model3[mask] += ints3[i]*np.exp(-0.5*((vel_grid[mask] - vlsr3)/(dV/2.355))**2.)
        model4[mask] += ints4[i]*np.exp(-0.5*((vel_grid[mask] - vlsr4)/(dV/2.355))**2.)

    J_T = (h*datagrid0*10**6/k)*(np.exp(((h*datagrid0*10**6)/(k*Tex))) -1)**-1
    J_Tbg = (h*datagrid0*10**6/k)*(np.exp(((h*datagrid0*10**6)/(k*2.7))) -1)**-1

    model1 = apply_beam(datagrid0, (J_T - J_Tbg)*(1 - np.exp(-model1)), ss1, 100)
    model2 = apply_beam(datagrid0, (J_T - J_Tbg)*(1 - np.exp(-model2)), ss2, 100)
    model3 = apply_beam(datagrid0, (J_T - J_Tbg)*(1 - np.exp(-model3)), ss3, 100)
    model4 = apply_beam(datagrid0, (J_T - J_Tbg)*(1 - np.exp(-model4)), ss4, 100)

    curr_model = model1 + model2 + model3 + model4

    return curr_model



def lnlike(theta, datagrid, mol_cat):
    tot_lnlike = 0.
    yerrs = datagrid[2]
    line_ids = datagrid[3]

    source_size1, source_size2, source_size3, source_size4, Ncol1, Ncol2, Ncol3, Ncol4, Tex, vlsr1, vlsr2, vlsr3, vlsr4, dV = theta

    freqs1, ints1, taus1 = predict_intensities(source_size1, Ncol1, Tex, dV, mol_cat)
    freqs2, ints2, taus2 = predict_intensities(source_size2, Ncol2, Tex, dV, mol_cat)
    freqs3, ints3, taus3 = predict_intensities(source_size3, Ncol3, Tex, dV, mol_cat)
    freqs4, ints4, taus4 = predict_intensities(source_size4, Ncol4, Tex, dV, mol_cat)

    freqs1 = np.array(freqs1)[line_ids]
    freqs2 = np.array(freqs2)[line_ids]
    freqs3 = np.array(freqs3)[line_ids]
    freqs4 = np.array(freqs4)[line_ids]

    taus1 = np.array(taus1)[line_ids]
    taus2 = np.array(taus2)[line_ids]
    taus3 = np.array(taus3)[line_ids]
    taus4 = np.array(taus4)[line_ids]

    ints1 = np.array(ints1)[line_ids]
    ints2 = np.array(ints2)[line_ids]
    ints3 = np.array(ints3)[line_ids]
    ints4 = np.array(ints4)[line_ids]

    curr_model = make_model(freqs1, freqs2, freqs3, freqs4, taus1, taus2,
            taus3, taus4, source_size1, source_size2, source_size3,
            source_size4, datagrid[0], datagrid[1], vlsr1, vlsr2, vlsr3, vlsr4,
            dV, Tex)

    inv_sigma2 = 1.0/(yerrs**2)
    tot_lnlike = np.sum((datagrid[1] - curr_model)**2*inv_sigma2 - np.log(inv_sigma2))
    #print(f"Log likelihood: {np.log(inv_sigma2)}")

    return -0.5*tot_lnlike



def lnprior(theta, prior_stds, prior_means):
    source_size1, source_size2, source_size3, source_size4, Ncol1, Ncol2, Ncol3, Ncol4, Tex, vlsr1, vlsr2, vlsr3, vlsr4, dV = theta
    #s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13 = prior_stds
    #m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13 = prior_means

    # in several cases the standard deviations on some of the parameters are too restrictive (e.g. vlsr and dV). Relaxing slightly
    #s9 = m13*0.8
    #s10 = m13*0.8
    #s11 = m13*0.8
    #s12 = m13*0.8
    #s13 = m13*0.3


    # set custom priors and limits here
    if (0. < source_size1 < 400) and (0. < source_size2 < 400) and (0. < source_size3 < 400) and (0. < source_size4 < 400) and (0. < Ncol1 < 10**16.) and (0. < Ncol2 < 10**16.) and (0. < Ncol3 < 10**16.) and (0. < Ncol4 < 10**16.) and (vlsr1 < (vlsr2-0.05)) and (vlsr2 < (vlsr3-0.05)) and (vlsr3 < (vlsr4-0.05)) and (vlsr2 < (vlsr1+0.3)) and (vlsr3 < (vlsr2+0.3)) and (vlsr4 < (vlsr3+0.3)) and (0. < dV < 0.3:
        return 0.

    return -np.inf



def lnprob(theta, datagrid, mol_cat, prior_stds, prior_means):
    lp = lnprior(theta, prior_stds, prior_means)
    if not np.isfinite(lp):
        return -np.inf
    prob = lnlike(theta, datagrid, mol_cat)
    return lp + prob



def fit_multi_gaussian(datafile, fit_folder, prior_path, catalogue, nruns, restart=True):
    """
    This version of the code is modified from the usual scripts, as it is the proto-MCMC
    sampler; we use uniform priors instead of a previous simulation.
    """
    print("Fitting column densities")
    datagrid = np.load(datafile, allow_pickle=True)

    ndim, nwalkers = 14, 200

    initial = [9.18647134e+01, 7.16006254e+01, 2.32811179e+02, 2.47626564e+02,
            1.76929473e+11*1.1, 5.50609449e+11*1.1, 2.85695178e+11*1.1,
            4.63340654e+11*1.1, 6.02600284e+00, 5.59259457e+00, 5.76263295e+00,
            5.88341792e+00, 6.01574809e+00, 1.22574843e-01]

    if restart==True:
        pos = [initial + np.array([1.e-1,1.e-1,1.e-1,1.e-1, 1.e10,1.e10,1.e10,1.e10, 1.e-3, 1.e-3,1.e-3,1.e-3,1.e-3, 1.e-3])*np.random.randn(ndim) for i in range(nwalkers)]
    else:
        initial = np.percentile(np.load(fit_folder + "/chain.npy")[:,-200:,:].reshape(-1,14).T, 50, axis=1)
        pos = [initial + np.array([1.e-1,1.e-1,1.e-1,1.e-1, 1.e10,1.e10,1.e10,1.e10, 1.e-3, 1.e-3,1.e-3,1.e-3,1.e-3, 1.e-3])*np.random.randn(ndim) for i in range(nwalkers)]

    mol_cat = MolCat("mol", catalogue)

    # load priors
    #psamples = np.load(prior_path).T

    prior_stds = 0.
    prior_means = 0.

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(datagrid, mol_cat, prior_stds, prior_means), pool=pool)
        file_name = fit_folder + "/chain.npy"

        for i in tqdm(range(nruns)):
            print(i)
            sampler.run_mcmc(pos, 1)
            np.save(file_name, np.array(sampler.chain))
            pos = sampler.chain[:,-1,:]

    return


def init_setup(fit_folder, cat_folder, data_path, mol_name, block_interlopers, **kwargs):
    print("Running setup for: " + mol_name + ", block_interlopers = " + str(block_interlopers))
    try:
        os.mkdir(fit_folder + "/" + mol_name)
    except:
        pass

    # Predict frequencies
    obs_params = ObsParams("init", source_size=40)
    catfile = cat_folder + "/"+mol_name+".cat"
    mol_cat = MolCat(mol_name, catfile)
    sim = MolSim(mol_name+" sim 8K", mol_cat, obs_params, [0.0], [7.e11],
            [0.37], [8.], ll=[7000], ul=[30000], gauss=False)
    freq_sim = np.array(sim.freq_sim)
    int_sim = np.array(sim.int_sim)

    # Read data
    print("Reading in data")
    freqs_gotham, ints_gotham, yerrs_gotham, covered_trans_gotham = read_file(data_path, freq_sim, int_sim, block_interlopers=block_interlopers, plot=False)

    # Compile all data
    datagrid = [freqs_gotham, ints_gotham, yerrs_gotham, covered_trans_gotham]

    # save the data
    datafile = fit_folder + "/" + mol_name + "/all_"+mol_name+"_lines_GOTHAM_freq_space.npy"
    print("Saving data to: " + datafile)
    np.save(datafile, datagrid)
    return datafile, catfile



###################################################################################
# assuming an input dictionary of name input_dict.py
from input_dict import input_dict

#mol_name = input_dict.mol_name
#fit_folder = input_dict.fit_folder
#cat_folder = input_dict.cat_folder
#data_path = input_dict.data_path
#prior_path  = input_dict.prior_path
#block_interlopers = input_dict.block_interlopers
#nruns = input_dict.nruns
#restart = input_dict.restart
fit_folder = input_dict.get("fit_folder")
prior_path = input_dict.get("prior_path")
nruns = input_dict.get("nruns")
restart = input_dict.get("restart")

datafile, catfile = init_setup(**input_dict)
fit_multi_gaussian(datafile, fit_folder, prior_path, catfile, nruns, restart)
