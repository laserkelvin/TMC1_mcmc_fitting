import numpy as np
import emcee
from scipy.interpolate import griddata
import scipy.optimize as op
import matplotlib.pylab as pl
import sys
import os
sys.path.append("/lustre/cv/users/rloomis/TMC-1/TMC-1/simulate_lte_RAL")
from classes import *
from constants import *
import time
from scipy.stats import gaussian_kde
from numba import njit

# calculates the rms noise in a given spectrum, should be robust to interloping lines, etc.
def calc_noise_std(spectrum):
    dummy_ints = np.copy(spectrum)
    noise = np.copy(spectrum)
    dummy_mean = np.nanmean(dummy_ints)
    dummy_std = np.nanstd(dummy_ints)

    # repeats 3 times to make sure to avoid any interloping lines
    for chan in np.where(dummy_ints < (-dummy_std*3.5))[0]:
        noise[chan-10:chan+10] = np.nan
    for chan in np.where(dummy_ints > (dummy_std*3.5))[0]:
        noise[chan-10:chan+10] = np.nan
    noise_mean = np.nanmean(noise)
    noise_std = np.nanstd(np.real(noise))

    for chan in np.where(dummy_ints < (-noise_std*3.5))[0]:
        noise[chan-10:chan+10] = np.nan
    for chan in np.where(dummy_ints > (noise_std*3.5))[0]:
        noise[chan-10:chan+10] = np.nan
    noise_mean = np.nanmean(noise)
    noise_std = np.nanstd(np.real(noise))

    for chan in np.where(dummy_ints < (-dummy_std*3.5))[0]:
        noise[chan-10:chan+10] = np.nan
    for chan in np.where(dummy_ints > (dummy_std*3.5))[0]:
        noise[chan-10:chan+10] = np.nan
    noise_mean = np.nanmean(noise)
    noise_std = np.nanstd(np.real(noise))

    return noise_mean, noise_std



# reads in the data and returns the data which has coverage of a given species (from the simulated intensities)
def read_file(filename, restfreqs, int_sim, oldformat=False, shift=0.0, GHz=False, plot=False, block_interlopers=True):
    data = np.load(filename)

    if oldformat:
        freqs = data[:,1]
        intensity = data[:,2]
    else:
        freqs = data[:,0]
        intensity = data[:,1]

    if GHz:
        freqs *= 1000.

    relevant_freqs = np.zeros(freqs.shape)
    relevant_intensity = np.zeros(intensity.shape)
    relevant_yerrs = np.zeros(freqs.shape)

    covered_trans = []

    for i, rf in enumerate(restfreqs):
        if block_interlopers:
            thresh = 0.1
        else:
            thresh = 0.05
        if int_sim[i] > thresh*np.max(int_sim):
            print(rf, int_sim[i])
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
    nan_mask = np.isnan(relevant_yerrs)
    logger.info(f"There are {nan_mask.sum()} baddies!")
    logger.info(f"{relevant_freqs[nan_mask]}")

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

    curr_model = model1+model2+model3+model4

    return curr_model



def lnlike(theta, datagrid, mol_cat):
    tot_lnlike = 0.
    yerrs = datagrid[2]
    line_ids = datagrid[3]

    source_size1, source_size2, source_size3, source_size4, Ncol1, Ncol2, Ncol3, Ncol4, Tex, vlsr1, vlsr2, vlsr3, vlsr4, dV = theta
    t0 = time.time()
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

    t1 = time.time()
    print("predict_time")
    print(t1-t0)

    t0 = time.time()
    curr_model = make_model(freqs1, freqs2, freqs3, freqs4, taus1, taus2, taus3, taus4, source_size1, source_size2, source_size3, source_size4, datagrid[0], datagrid[1], vlsr1, vlsr2, vlsr3, vlsr4, dV, Tex)
    t1 = time.time()
    print("convolve_time")
    print(t1-t0)

    inv_sigma2 = 1.0/(yerrs**2)
    tot_lnlike = np.sum((datagrid[1] - curr_model)**2*inv_sigma2 - np.log(inv_sigma2))
    #pl.plot(datagrid[0], curr_model, color='darkslateblue')
    #pl.plot(datagrid[0], datagrid[1], color='firebrick')
    #pl.fill_between(datagrid[0], datagrid[1]-yerrs, datagrid[1]+yerrs, color='firebrick', alpha=0.5)
    #pl.show()

    #print("Likelihood function call, tot_lnlike = " + str(tot_lnlike) + ", Ncol guesses = " + str(theta))

    return -0.5*tot_lnlike



def lnprior(theta, prior_stds, prior_means):
    source_size1, source_size2, source_size3, source_size4, Ncol1, Ncol2, Ncol3, Ncol4, Tex, vlsr1, vlsr2, vlsr3, vlsr4, dV = theta

    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13 = prior_stds
    m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13 = prior_means

    s13 = m13*0.1

    if (0. < source_size1 < 400) and (0. < source_size2 < 400) and (0. < source_size3 < 400) and (0. < source_size4 < 400) and (0. < Ncol1 < 10**16.) and (0. < Ncol2 < 10**16.) and (0. < Ncol3 < 10**16.) and (0. < Ncol4 < 10**16.) and (vlsr1 < (vlsr2-0.05)) and (vlsr2 < (vlsr3-0.05)) and (vlsr3 < (vlsr4-0.05)) and (vlsr2 < (vlsr1+0.3)) and (vlsr3 < (vlsr2+0.3)) and (vlsr4 < (vlsr3+0.3)) and dV < 0.3:
        p0 = np.log(1.0/(np.sqrt(2*np.pi)*s0))-0.5*(source_size1-m0)**2/s0**2
        p1 = np.log(1.0/(np.sqrt(2*np.pi)*s1))-0.5*(source_size2-m1)**2/s1**2
        p2 = np.log(1.0/(np.sqrt(2*np.pi)*s2))-0.5*(source_size3-m2)**2/s2**2
        p3 = np.log(1.0/(np.sqrt(2*np.pi)*s3))-0.5*(source_size4-m3)**2/s3**2

        p8 = np.log(1.0/(np.sqrt(2*np.pi)*s8))-0.5*(Tex-m8)**2/s8**2

        p9 = np.log(1.0/(np.sqrt(2*np.pi)*s9))-0.5*(vlsr1-m9)**2/s9**2
        p10 = np.log(1.0/(np.sqrt(2*np.pi)*s10))-0.5*(vlsr2-m10)**2/s10**2
        p11 = np.log(1.0/(np.sqrt(2*np.pi)*s11))-0.5*(vlsr3-m11)**2/s11**2
        p12 = np.log(1.0/(np.sqrt(2*np.pi)*s12))-0.5*(vlsr4-m12)**2/s12**2

        p13 = np.log(1.0/(np.sqrt(2*np.pi)*s13))-0.5*(dV-m13)**2/s13**2

        return p0 + p1 + p2 + p3 + p8 + p9 + p10 + p11 + p12 + p13
    
    #if (1. < source_size1 < 2.e2) and (1. < source_size2 < 2.e2) and (1. < source_size3 < 2.e2) and (1. < source_size4 < 2.e2) and (8. < Ncol1 < 15.) and (8. < Ncol2 < 15.) and (8. < Ncol3 < 15.) and (8. < Ncol4 < 15.) and (2.7 < Tex < 25.) and (5.3 < vlsr1 < 6.3) and (5.3 < vlsr2 < 6.3) and (5.3 < vlsr3 < 6.3) and (5.3 < vlsr4 < 6.3) and (vlsr1 < vlsr2) and (vlsr2 < vlsr3) and (vlsr3 < vlsr4) and (0.1 < dV < 1.0):
    #    return 0.0

    return -np.inf



def lnprob(theta, datagrid, mol_cat, prior_stds, prior_means):
    lp = lnprior(theta, prior_stds, prior_means)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, datagrid, mol_cat)



def fit_multi_gaussian(datafile, catalogue, nruns, restart=True):
    print("Fitting column densities")
    datagrid = np.load(datafile, allow_pickle=True)

    ndim, nwalkers = 14, 100

    initial = [9.97766019e+01, 6.53763803e+01, 2.65877815e+02, 2.62112227e+02, 10**11.20793551, 10**11.3842407 , 10**10.9991768 , 10**11.30212397, 6.14379612e+00, 5.59469316e+00, 5.76406900e+00, 5.88574970e+00, 6.01717941e+00, 1.20822176e-01]


    if restart==True:
        pos = [initial + np.array([1.e-1,1.e-1,1.e-1,1.e-1, 1.e10,1.e10,1.e10,1.e10, 1.e-3, 1.e-3,1.e-3,1.e-3,1.e-3, 1.e-3])*np.random.randn(ndim) for i in range(nwalkers)]
    else:
        #pos = np.vstack((np.load("1-cyanonapthalene_fit/chain_pt1.npy")[:,-1,:], np.load("1-cyanonapthalene_fit/chain_pt1.npy")[:,-1,:] + 0.0001, np.load("1-cyanonapthalene_fit/chain_pt1.npy")[:,-1,:] + 0.0002, np.load("1-cyanonapthalene_fit/chain_pt1.npy")[:,-1,:] - 0.0002))
        initial = np.percentile(np.load("1-cyanonapthalene_fit/chain_pt1.npy")[:,-200:,:].reshape(-1,14).T, 50, axis=1)
        pos = [initial + np.array([1.e-1,1.e-1,1.e-1,1.e-1, 1.e10,1.e10,1.e10,1.e10, 1.e-3, 1.e-3,1.e-3,1.e-3,1.e-3, 1.e-3])*np.random.randn(ndim) for i in range(nwalkers)]

    mol_cat = MolCat("mol", catalogue)

    # load priors
    psamples = np.load("benzonitrile_fit/chain_pt2.npy")[:,:-400,:].reshape((-1,14)).T

    prior_stds = (np.abs(np.percentile(psamples, 16, axis=1)-np.percentile(psamples, 50, axis=1)) + np.abs(np.percentile(psamples, 84, axis=1) - np.percentile(psamples, 50, axis=1)))/2.
    prior_means = np.percentile(psamples, 50, axis=1)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(datagrid, mol_cat, prior_stds, prior_means), threads=16)

    for i in range(nruns):
        print(i)
        sampler.run_mcmc(pos, 1)
        file_name = "1-cyanonapthalene_fit/chain_pt2.npy"
        np.save(file_name, np.array(sampler.chain))
        pos = sampler.chain[:,-1,:]

    return


def init_setup(mol_name, block_interlopers):
    print("Running setup for: " + mol_name + ", block_interlopers = " + str(block_interlopers))
    os.mkdir("1-cyanonapthalene_fit/" + mol_name)

    # Predict frequencies
    obs_params = ObsParams("init", source_size=40)
    catfile = "/lustre/cv/users/rloomis/TMC-1/TMC-1/GOTHAM/gotham_catalogs_trimmed/"+mol_name+".cat"
    mol_cat = MolCat(mol_name, catfile)
    sim = MolSim(mol_name+" sim 8K", mol_cat, obs_params, [0.0], [7.e11], [0.12], [8.], ll=[7000], ul=[30000], gauss=False)
    freq_sim = np.array(sim.freq_sim)
    int_sim = np.array(sim.int_sim)

    # Read data
    print("Reading in data")
    freqs_gotham, ints_gotham, yerrs_gotham, covered_trans_gotham = read_file('/lustre/cv/users/rloomis/TMC-1/TMC-1/GOTHAM_data/tmc_all_gbt.npy', freq_sim, int_sim, block_interlopers=block_interlopers, plot=False)

    # Compile all data
    datagrid = [freqs_gotham, ints_gotham, yerrs_gotham, covered_trans_gotham]

    # save the data
    datafile = "1-cyanonapthalene_fit/" + mol_name + "/all_"+mol_name+"_lines_GOTHAM_freq_space.npy"
    print("Saving data to: " + datafile)
    np.save(datafile, datagrid)
    return datafile, catfile




###################################################################################
mol_data = np.genfromtxt('1-cyanonapthalene.txt',dtype='str')
print(mol_data)
mols = [mol_data[0]]
blocks = [eval(mol_data[1])]

for i, mol in enumerate(mols):
    mols[i] = mol[:-4]
for i, mol in enumerate(mols):
    datafile, catfile = init_setup(mol, block_interlopers=blocks[i])
    fit_multi_gaussian(datafile, catfile, 4000, restart=False)
    #outfile = "1-cyanonapthalene_fit/" + mol + "/best_params.txt"
    #np.savetxt(outfile, best_params)
    #plot_best(best_params, fixed_params, datafile, catfile)
