# nemesispy_scatter

## Installation:

To install, clone the repository and then type:

pip install .

## Usage:

There's an example script "optimal_estimation_example.py" - it's very simple. It imports the NEMESIS class, initialises it with an input file, and runs optimal estimation. The input file is of the following form (there's one in the folder as well):

HEADER

0 or 1 for multiple folder input - this tells the code whether you're about to give it the path to a single folder to run a retrieval in (0) or whether you are giving it the path to a folder containing multiple folders to run retrievals in simultaneously.

The next line is just the path to the retrieval folder.

The next line is the project (the name of your .spx, .inp files)

The Mie scattering calculations work a little differently in this version, so the next line tells Nemesis how spaced out the Makephase calculation wavelengths should be. To replicate Fortran results, leave this at 0.

The final line is the correlation length in degrees for the spatial smoothing term. Just like the vertical smoothing for continuous profiles, this correlates the state vectors between retrievals if you are running in multiple folder mode. The degree of correlation is dependent on the locations in the .spx files, and the correlation length. Setting this to 0 gives the same results as running the retrievals separately.

Once you've filled out the input file, you can run the script with:

mpiexec -np N python ./optimal_estimation_example.py

to run on N cores. The speed of retrievals scales (roughly) linearly with the number of processors.

## Notes:

· You'll need a RADREPO environment variable set - the code will look in $RADREPO/raddata/ for the data files it needs.

· The first time you run it, the code takes some time (~2-3 min) to initialise and compile. After this, it should store a cache somewhere, and will be a lot quicker (~10 s) to compile on subsequent runs. 

· An example slurm script is also provided for running the code on the cluster.

· Most parameterisations are not implemented. A main priority right now is implementing more of these to ensure compatibility with Nemesis. I've mainly just been using continuous profiles for most things.

· Errors can often be obscure. I've tried to put in error traps for most issues with reading files, so hopefully most of the time you should be able to tell what's wrong. However, if there is an issue further along in the code, it may be difficult to trace.

· For what I've been doing, the forward model matches up with the Fortran version to a high degree of accuracy. However, some combinations of inputs seem to cause deviations from this - I think I've fixed most of them, but I'm sure some still remain.

· If you're using spatial smoothing, the matrix operations needed for optimal estimation take longer with larger spatial correlation lengths. In particular, calculating the errors on xn at the end of a retrieval can take a long time with long state vectors and a large number of locations.

## Nested Sampling

- The current way to do nested sampling is by adding a prior distribution code (0 for a log-gaussian, 1 for a log-uniform distribution) and a prior distribution width (in terms of the apriori fractional error) to lines in the .apr file. It defaults to a gaussian with a standard deviation of 1*(apriori error)
- For example, consider this cloud:

-1 0 32
  
1.0 0.2 1 5

1e-2 4e-3 0 3

0.05 0.05e-8
- The first parameter, the knee pressure, has a log-uniform prior distribution, with a mean of 1, an upper bound of exp(log(1) + (0.2/1)*5) = exp(1) and a lower bound of exp(log(1) - (0.2/1)*5) = exp(-1)
- The second parameter, the peak opacity, has a log-gaussian prior distribution with a mean of 1e-2, and a standard deviation in log space of (4e-3/1e-2)*3 = 1.2
- The third parameter has a fractional error less than 1e-7 and so is constant.

There is an example file, "nested_sampling_example.py", which will run nested sampling on some synthetic data (with gaussian noise added) in the example_nested_sampling/ folder. You should run this in parallel with mpiexec - it needs around 100k samples to complete!
