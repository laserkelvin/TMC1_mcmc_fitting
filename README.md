# TMC-1 MCMC analysis

This folder builds off of Ryan's scripts to perform the MCMC
analysis of TMC-1 using the GOTHAM DR1.

The modifications made here by Kelvin comprising a significant refactoring
and organization of the code into effectively a single module:
all the functionality from reading in input, setting up the file
structure, computing beam corrections, model likelihoods, and
running the MCMC using `emcee` is included here. The exceptions
to this are legacy `simulate_lte` functions, such as `MolCat`
and `ObsParams`.

The way that the simulations are run in this version is through
YAML input files, contained in `yml/`. The parameter names are
the same as in the previous version. You can then create a short
Python script that calls two functions: `init_setup` and
`fit_multi_gaussian`; the former creates the file structures,
and the latter performs the MCMC analysis.

## Set up and workflow

1. Clone this repository
2. Use `conda env create -f conda.yml` to reproduce the software environment
3. Generate priors using benzonitrile or other
4. Perform MCMC analysis of other molecules, using precomputed Gaussian priors

Corner plots are generated with `seaborn`, although in the first iteration `corner`
was used. Using `seaborn` was a little bit more extensible.

## YAML input structure

```yaml
mol_name:   # name of the molecule, will be used to lookup catalog and generate folder
fit_folder: # name of the folder to save data to; {fit_folder}/{mol_name}
data_path:  # filename/path to the file containing X/Y data
cat_folder: # folder containing all the SPCAT catalogs
prior_path: # if not `null`, will use statistics of this a previously run chain to use as parameters for a Gaussian prior
block_interlopers: # if true, we block out mask strong features in the noise/yerr calculation
nruns:      # number of iterations (in batches of 10) to sample
restart:    # if true, sampling is continued from a previous run by reading in `chain.npy` in the output folder
initial:    # if not `nul`, load in a NumPy array containing parameters to start sampling from
workers:    # number of workers to parallelize sampling over
```

## Generating priors with benzonitrile

To start a run from scratch (i.e. uniform, uninformative priors)
the `prior_path` keyword in the YAML file can be given as `null`,
which corresponds to `None` in Python. The effective parameters
can be found in `yml/benzonitrile.yml`. To generate the priors,
we use some initial guess parameters that are hard coded, corresponding
to early preliminary fits. 

Other relevant parameters:

- `restart` is set to `True` because we will always start from scratch
- `initial` is set to `null`, same as above
- `block_interlopers` is set to `False`, because benzonitrile has strong lines

## Production runs

For subsequent molecules, be sure to:

- Use the benzonitrile posterior mean as an initial starting point with `initial`. The parameters will have to be calculated separately with `chain.npy`
- Make sure `block_interlopers` is set to `True`
- Make sure `prior_path` is set to the Benzonitrile `chain.npy`

To calculate initial parameters, we've been using the 50th percentile (i.e. the median) to represent the posterior mean:

```python
# shape of chain is n_samples, n_walkers, n_parameters
bn_chain = np.load("prior/benzonitrile/chain.npy")
# flatten walker dimension after taking last 1000 samples with 14 parameters
bn_chain = bn_chain[-1000:,:,:].reshape(-1,14)
median = np.percentile(bn_chain, 50, axis=0)
np.save("initial_parameters.npy", median)
```

The `1-cnn.yml` and `2-cnn.yml` files in the `yml` directory demonstrate the correct input format and parameters for 1- and 2-cyanonaphthalene respectively.

