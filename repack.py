from typing import Type, Tuple
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import ruamel.yaml as yaml
import emcee
from loguru import logger
from numba import njit

from constants import *
from classes import MolSim, ObsParams, MolCat


@njit(fastmath=True)
def calc_noise_std(y: np.ndarray, sigma=3):
    tmp_y = np.copy(y)
    i = np.nanmax(tmp_y)
    rms = np.sqrt(np.nanmean(np.square(tmp_y)))
    while i > sigma * rms:
        tmp_y = tmp_y[tmp_y < sigma * rms]
        noise_mean = np.nanmean(tmp_y)
        noise_std = np.nanstd(np.real(tmp_y))
        rms = np.sqrt(np.nanmean(np.square(tmp_y)))
        # noise std as the rms with mean subtracted away
        i = np.nanmax(tmp_y)
    return noise_mean, noise_std


# reads in the data and returns the data which has coverage of a given species (from the simulated intensities)
def read_file(
    filename: str,
    restfreqs: np.ndarray,
    int_sim: np.ndarray,
    oldformat=False,
    shift=0.0,
    GHz=False,
    plot=False,
    block_interlopers=True,
):
    data = np.load(filename, allow_pickle=True)

    if oldformat:
        freqs = data[:, 1]
        intensity = data[:, 2]
    else:
        freqs = data[0]
        intensity = data[1]

    if GHz:
        freqs *= 1000.0

    relevant_freqs = np.zeros(freqs.shape)
    relevant_intensity = np.zeros(intensity.shape)
    relevant_yerrs = np.zeros(freqs.shape)

    covered_trans = []

    # loop over the rest frequencies
    for i, rf in enumerate(restfreqs):
        thresh = 0.05
        # if the simulated intensity is greater than a threshold, we process it
        if int_sim[i] > thresh * np.max(int_sim):
            # convert to VLSR
            vel = (rf - freqs) / rf * 300000 + shift
            locs = np.where((vel < (0.3 + 6.0)) & (vel > (-0.3 + 5.6)))
            # if there are values that satisfy the condition, go further
            if locs[0].size != 0:
                noise_mean, noise_std = calc_noise_std(intensity[locs])
                if block_interlopers and (np.max(intensity[locs]) > 6 * noise_std):
                    logger.info(f"Interloping line detected @ {rf}.")
                    if plot:
                        pl.plot(freqs[locs], intensity[locs])
                        pl.show()
                else:
                    covered_trans.append(i)
                    logger.info(f"Found_line at @ {rf}")
                    relevant_freqs[locs] = freqs[locs]
                    relevant_intensity[locs] = intensity[locs]
                    relevant_yerrs[locs] = np.sqrt(
                        noise_std ** 2 + (intensity[locs] * 0.1) ** 2
                    )

                if plot:
                    pl.plot(freqs[locs], intensity[locs])
                    pl.show()
            else:
                logger.info(f"No data covering {rf}")

    mask = relevant_freqs > 0

    relevant_freqs = relevant_freqs[mask]
    relevant_intensity = relevant_intensity[mask]
    relevant_yerrs = relevant_yerrs[mask]

    return (relevant_freqs, relevant_intensity, relevant_yerrs, covered_trans)


def predict_intensities(
    source_size: float, Ncol: float, Tex: float, dV: float, mol_cat: Type[MolCat]
):
    """
    Compute the simulated frequency/intensities using a set of model parameters. This function is
    called by each individual model to compute its contribution to the log likelihood.

    Parameters
    ----------
    source_size : float
        Size of the source
    Ncol : float
        Column density
    Tex : float
        Excitation temperature
    dV : float
        Linewidth
    mol_cat : Type[MolCat]
        Transitions catalog represented by a `MolCat` object

    Returns
    -------
    np.ndarray
        Arrays corresponding to frequency, intensity, and opacity
    """
    obs_params = ObsParams("test", source_size=source_size)
    sim = MolSim(
        "mol sim",
        mol_cat,
        obs_params,
        [0.0],
        [Ncol],
        [dV],
        [Tex],
        ll=[7000],
        ul=[30000],
        gauss=False,
    )
    freq_sim = sim.freq_sim
    int_sim = sim.int_sim
    tau_sim = sim.tau_sim

    return freq_sim, int_sim, tau_sim


# Apply a beam dilution correction factor
@njit(fastmath=True)
def apply_beam(frequency, intensity, source_size, dish_size):
    # create a wave to hold wavelengths, fill it to start w/ frequencies
    wavelength = cm / (frequency * 1e6)

    # fill it with beamsizes
    beam_size = wavelength * 206265 * 1.22 / dish_size

    # create an array to hold beam dilution factors
    dilution_factor = source_size ** 2 / (beam_size ** 2 + source_size ** 2)

    intensity_diluted = intensity * dilution_factor

    return intensity_diluted


@njit(fastmath=True)
def make_model(
    freqs1,
    freqs2,
    freqs3,
    freqs4,
    ints1,
    ints2,
    ints3,
    ints4,
    ss1,
    ss2,
    ss3,
    ss4,
    datagrid0,
    datagrid1,
    vlsr1,
    vlsr2,
    vlsr3,
    vlsr4,
    dV,
    Tex,
):
    curr_model = np.zeros(datagrid1.shape)
    model1 = np.zeros(datagrid1.shape)
    model2 = np.zeros(datagrid1.shape)
    model3 = np.zeros(datagrid1.shape)
    model4 = np.zeros(datagrid1.shape)
    nlines = freqs1.shape[0]

    for i in range(nlines):
        vel_grid = (freqs1[i] - datagrid0) / freqs1[i] * ckm
        mask = np.abs(vel_grid - 5.8) < dV * 10
        model1[mask] += ints1[i] * np.exp(
            -0.5 * ((vel_grid[mask] - vlsr1) / (dV / 2.355)) ** 2.0
        )
        model2[mask] += ints2[i] * np.exp(
            -0.5 * ((vel_grid[mask] - vlsr2) / (dV / 2.355)) ** 2.0
        )
        model3[mask] += ints3[i] * np.exp(
            -0.5 * ((vel_grid[mask] - vlsr3) / (dV / 2.355)) ** 2.0
        )
        model4[mask] += ints4[i] * np.exp(
            -0.5 * ((vel_grid[mask] - vlsr4) / (dV / 2.355)) ** 2.0
        )

    J_T = (h * datagrid0 * 10 ** 6 / k) * (
        np.exp(((h * datagrid0 * 10 ** 6) / (k * Tex))) - 1
    ) ** -1
    J_Tbg = (h * datagrid0 * 10 ** 6 / k) * (
        np.exp(((h * datagrid0 * 10 ** 6) / (k * 2.7))) - 1
    ) ** -1

    model1 = apply_beam(datagrid0, (J_T - J_Tbg) * (1 - np.exp(-model1)), ss1, 100)
    model2 = apply_beam(datagrid0, (J_T - J_Tbg) * (1 - np.exp(-model2)), ss2, 100)
    model3 = apply_beam(datagrid0, (J_T - J_Tbg) * (1 - np.exp(-model3)), ss3, 100)
    model4 = apply_beam(datagrid0, (J_T - J_Tbg) * (1 - np.exp(-model4)), ss4, 100)

    curr_model = model1 + model2 + model3 + model4

    return curr_model


"""
Likelihood definitions

This section of the code defines all of the ways that the log likelihood and prior
is computed, given a set of parameters.
"""


def lnlike(theta: Tuple[float], datagrid: np.ndarray, mol_cat: Type[MolCat]) -> float:
    """
    Compute the log likelihood, given model parameters, a molecule, and observations.
    
    This function first calls `predict_intensities` to compute the contribution from
    each model, and applies beam corrections and sums over models with `make_model`.
    
    The log likelihood is then given by `tot_lnlike`.

    Parameters
    ----------
    theta : Tuple[float]
        Tuple of model parameters that are unpacked and used to compute
        the contribution from each model
    datagrid : np.ndarray
        Set of observational data to condition the model with
    mol_cat : Type[MolCat]
        `MolCat` object used to compute intensities

    Returns
    -------
    float
        The total log likelihood
    """
    tot_lnlike = 0.0
    yerrs = datagrid[2]
    line_ids = datagrid[3]

    (
        source_size1,
        source_size2,
        source_size3,
        source_size4,
        Ncol1,
        Ncol2,
        Ncol3,
        Ncol4,
        Tex,
        vlsr1,
        vlsr2,
        vlsr3,
        vlsr4,
        dV,
    ) = theta

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

    curr_model = make_model(
        freqs1,
        freqs2,
        freqs3,
        freqs4,
        taus1,
        taus2,
        taus3,
        taus4,
        source_size1,
        source_size2,
        source_size3,
        source_size4,
        datagrid[0],
        datagrid[1],
        vlsr1,
        vlsr2,
        vlsr3,
        vlsr4,
        dV,
        Tex,
    )

    inv_sigma2 = 1.0 / (yerrs ** 2)
    tot_lnlike = np.sum(
        (datagrid[1] - curr_model) ** 2 * inv_sigma2 - np.log(inv_sigma2)
    )

    return -0.5 * tot_lnlike


def base_lnprior(theta):
    """
    Compute the prior for a set of parameters. This is used for simulations that do
    not have a prior, and instead uses uniform, uninformative priors.

    Parameters
    ----------
    theta : [type]
        [description]

    Returns
    -------
    float
        Prior likelihood
    """
    (
        source_size1,
        source_size2,
        source_size3,
        source_size4,
        Ncol1,
        Ncol2,
        Ncol3,
        Ncol4,
        Tex,
        vlsr1,
        vlsr2,
        vlsr3,
        vlsr4,
        dV,
    ) = theta

    # set custom priors and limits here
    if (
        (0.0 < source_size1 < 400)
        and (0.0 < source_size2 < 400)
        and (0.0 < source_size3 < 400)
        and (0.0 < source_size4 < 400)
        and (0.0 < Ncol1 < 10 ** 16.0)
        and (0.0 < Ncol2 < 10 ** 16.0)
        and (0.0 < Ncol3 < 10 ** 16.0)
        and (0.0 < Ncol4 < 10 ** 16.0)
        and (vlsr1 < (vlsr2 - 0.05))
        and (vlsr2 < (vlsr3 - 0.05))
        and (vlsr3 < (vlsr4 - 0.05))
        and (vlsr2 < (vlsr1 + 0.3))
        and (vlsr3 < (vlsr2 + 0.3))
        and (vlsr4 < (vlsr3 + 0.3))
        and (0.0 < dV < 0.3)
    ):
        return 0.0
    return -np.inf


def lnprior(theta, prior_stds, prior_means):
    """
    The regular modeling prior, that is based off of an actual template, and computed
    by Gaussian likelihoods.

    Parameters
    ----------
    theta : [type]
        [description]
    prior_stds : [type]
        [description]
    prior_means : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    (
        source_size1,
        source_size2,
        source_size3,
        source_size4,
        Ncol1,
        Ncol2,
        Ncol3,
        Ncol4,
        Tex,
        vlsr1,
        vlsr2,
        vlsr3,
        vlsr4,
        dV,
    ) = theta
    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13 = prior_stds
    m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13 = prior_means

    # in several cases the standard deviations on some of the parameters are too restrictive (e.g. vlsr and dV). Relaxing slightly
    s9 = m13 * 0.8
    s10 = m13 * 0.8
    s11 = m13 * 0.8
    s12 = m13 * 0.8
    s13 = m13 * 0.3

    # set custom priors and limits here
    if (
        (0.0 < source_size1 < 200)
        and (0.0 < source_size2 < 200)
        and (0.0 < source_size3 < 200)
        and (0.0 < source_size4 < 200)
        and (0.0 < Ncol1 < 10 ** 16.0)
        and (0.0 < Ncol2 < 10 ** 16.0)
        and (0.0 < Ncol3 < 10 ** 16.0)
        and (0.0 < Ncol4 < 10 ** 16.0)
        and (vlsr1 < (vlsr2 - 0.05))
        and (vlsr2 < (vlsr3 - 0.05))
        and (vlsr3 < (vlsr4 - 0.05))
        and (vlsr2 < (vlsr1 + 0.3))
        and (vlsr3 < (vlsr2 + 0.2))
        and (vlsr4 < (vlsr3 + 0.2))
        and dV < 0.3
    ):

        p0 = (
            np.log(1.0 / (np.sqrt(2 * np.pi) * s0))
            - 0.5 * (source_size1 - m0) ** 2 / s0 ** 2
        )
        p1 = (
            np.log(1.0 / (np.sqrt(2 * np.pi) * s1))
            - 0.5 * (source_size2 - m1) ** 2 / s1 ** 2
        )
        p2 = (
            np.log(1.0 / (np.sqrt(2 * np.pi) * s2))
            - 0.5 * (source_size3 - m2) ** 2 / s2 ** 2
        )
        p3 = (
            np.log(1.0 / (np.sqrt(2 * np.pi) * s3))
            - 0.5 * (source_size4 - m3) ** 2 / s3 ** 2
        )

        p8 = np.log(1.0 / (np.sqrt(2 * np.pi) * s8)) - 0.5 * (Tex - m8) ** 2 / s8 ** 2

        p9 = np.log(1.0 / (np.sqrt(2 * np.pi) * s9)) - 0.5 * (vlsr1 - m9) ** 2 / s9 ** 2
        p10 = (
            np.log(1.0 / (np.sqrt(2 * np.pi) * s10))
            - 0.5 * (vlsr2 - m10) ** 2 / s10 ** 2
        )
        p11 = (
            np.log(1.0 / (np.sqrt(2 * np.pi) * s11))
            - 0.5 * (vlsr3 - m11) ** 2 / s11 ** 2
        )
        p12 = (
            np.log(1.0 / (np.sqrt(2 * np.pi) * s12))
            - 0.5 * (vlsr4 - m12) ** 2 / s12 ** 2
        )

        p13 = (
            np.log(1.0 / (np.sqrt(2 * np.pi) * s13)) - 0.5 * (dV - m13) ** 2 / s13 ** 2
        )

        return p0 + p1 + p2 + p3 + p8 + p9 + p10 + p11 + p12 + p13

    return -np.inf


def lnprob(theta, datagrid, mol_cat, prior_stds=None, prior_means=None):
    """
    Compute the total log probability as the sum of the prior
    and the log likelihood.

    Parameters
    ----------
    theta : [type]
        [description]
    datagrid : [type]
        [description]
    mol_cat : [type]
        [description]
    prior_stds : [type]
        [description]
    prior_means : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if (prior_stds is None) and (prior_means is None):
        lp = base_lnprior(theta)
    # else:
    #     lp = lnprior(theta, prior_stds, prior_means)
    if not np.isfinite(lp):
        return -np.inf
    prob = lnlike(theta, datagrid, mol_cat)
    return lp + prob


def load_input_file(yml_path):
    with open(yml_path, "r") as read_file:
        input_dict = yaml.load(read_file)
    return input_dict


def init_setup(
    fit_folder, cat_folder, data_path, mol_name, block_interlopers, **kwargs
):
    fit_folder = Path(fit_folder)
    cat_folder = Path(cat_folder)
    data_path = Path(data_path)
    output_path = fit_folder.joinpath(mol_name)

    # make the output path
    output_path.mkdir(parents=True, exist_ok=True)
    # point to the catalog
    catalog_path = cat_folder.joinpath(mol_name).with_suffix(".cat")
    # set up logging
    logger.add(output_path.joinpath("LOG"))
    logger.info(
        f"Running setup for: {mol_name}, with block_interplopers={block_interlopers}"
    )

    # Predict frequencies
    obs_params = ObsParams("init", source_size=40)
    mol_cat = MolCat(mol_name, str(catalog_path))
    sim = MolSim(
        mol_name + " sim 8K",
        mol_cat,
        obs_params,
        [0.0],
        [7.0e11],
        [0.37],
        [8.0],
        ll=[7000],
        ul=[30000],
        gauss=False,
    )
    freq_sim = np.array(sim.freq_sim)
    int_sim = np.array(sim.int_sim)

    # Read data
    logger.info("Reading in data")
    freqs_gotham, ints_gotham, yerrs_gotham, covered_trans_gotham = read_file(
        data_path, freq_sim, int_sim, block_interlopers=block_interlopers, plot=False
    )

    # Compile all data
    datagrid = [freqs_gotham, ints_gotham, yerrs_gotham, covered_trans_gotham]

    # save the data
    npy_path = output_path.joinpath(f"all_{mol_name}_lines_GOTHAM_freq_space.npy")
    logger.info(f"Saving data to: {npy_path}")
    np.save(npy_path, datagrid)
    return npy_path, str(catalog_path), output_path


def fit_multi_gaussian(
    datafile,
    output_path,
    catalogue,
    nruns,
    restart=True,
    prior_path=None,
    workers=8,
    progressbar=False,
    **kwargs,
):
    """
    This version of the code is modified from the usual scripts, as it is the proto-MCMC
    sampler; we use uniform priors instead of a previous simulation.
    """
    logger.info("Fitting column densities.")
    datagrid = np.load(datafile, allow_pickle=True)

    ndim, nwalkers = 14, 200

    initial = [
        9.18647134e01,
        7.16006254e01,
        2.32811179e02,
        2.47626564e02,
        1.76929473e11 * 1.1,
        5.50609449e11 * 1.1,
        2.85695178e11 * 1.1,
        4.63340654e11 * 1.1,
        6.02600284e00,
        5.59259457e00,
        5.76263295e00,
        5.88341792e00,
        6.01574809e00,
        1.22574843e-01,
    ]

    if restart:
        pos = [
            initial
            + np.array(
                [
                    1.0e-1,
                    1.0e-1,
                    1.0e-1,
                    1.0e-1,
                    1.0e10,
                    1.0e10,
                    1.0e10,
                    1.0e10,
                    1.0e-3,
                    1.0e-3,
                    1.0e-3,
                    1.0e-3,
                    1.0e-3,
                    1.0e-3,
                ]
            )
            * np.random.randn(ndim)
            for i in range(nwalkers)
        ]
    else:
        initial = np.percentile(
            np.load(output_path.joinpath("chain.npy"))[:, -200:, :].reshape(-1, 14).T,
            50,
            axis=1,
        )
        pos = [
            initial
            + np.array(
                [
                    1.0e-1,
                    1.0e-1,
                    1.0e-1,
                    1.0e-1,
                    1.0e10,
                    1.0e10,
                    1.0e10,
                    1.0e10,
                    1.0e-3,
                    1.0e-3,
                    1.0e-3,
                    1.0e-3,
                    1.0e-3,
                    1.0e-3,
                ]
            )
            * np.random.randn(ndim)
            for i in range(nwalkers)
        ]
    # initialize a catalog object
    mol_cat = MolCat("mol", catalogue)

    # if prior path is specified, then we load in the chain and compute
    # the statistics
    if prior_path:
        # load priors
        prior_samples = np.load(prior_path).T

        prior_stds = (
            np.abs(
                np.percentile(prior_samples, 16, axis=1)
                - np.percentile(prior_samples, 50, axis=1)
            )
            + np.abs(
                np.percentile(prior_samples, 84, axis=1)
                - np.percentile(prior_samples, 50, axis=1)
            )
        ) / 2.0
        prior_means = np.percentile(prior_samples, 50, axis=1)
    else:
        prior_stds = prior_means = None
    logger.info(f"Using prior {prior_path}, mean: {prior_means}, std: {prior_stds}")

    # run the MCMC sampling
    output_chain = Path(output_path).joinpath("chain.npy")
    logger.info("Entering MCMC sampling routine.")
    with Pool(workers) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=(datagrid, mol_cat, prior_stds, prior_means),
            pool=pool,
        )
        iterator = range(nruns)
        if progressbar:
            from tqdm import tqdm

            iterator = tqdm(iterator)
        for iteration in iterator:
            sampler.run_mcmc(pos, 1)
            if iteration % 10 == 0:
                np.save(output_chain, np.array(sampler.chain))
            pos = sampler.chain[:, -1, :]
    logger.info("Completed MCMC sampling routine.")
    return
